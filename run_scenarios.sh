#!/usr/bin/env bash
# Runs a sequence of loss-flag ablation scenarios through train.py, one at a
# time. Each scenario reuses the on-disk patch cache (see dataset.py /
# config.py CACHE_DIR) — only the first scenario in the queue pays the full
# preload cost, later ones with the same data config load in seconds.
#
# Edit the SCENARIOS array below to add/remove/reorder scenarios. Flags are
# train.py's existing --use_X / --no_X CLI overrides (see `python train.py
# --help`); leave the flag list empty for the L1-only baseline.
#
# Usage:
#   ./run_scenarios.sh                    # run every scenario in order
#   ./run_scenarios.sh diff_v hetero_nll  # run only the named scenario(s)
#   SEEDS="42 43 44" ./run_scenarios.sh diff_v
#
# ORDER MATTERS. Run the gates before the model they gate:
#   0. python infer_volume.py --scenario_dir <any run> --dry_run
#        -> tile count per volume + total forward passes. The whole sampling
#           budget depends on it. Needs run_config.json and the data only, NOT a
#           checkpoint, so it can be run before/while a scenario trains.
#   1. python scripts/vae_fidelity.py ...      Gate D1 (§4)
#   2. hetero_nll                              the bar (§6 step 2)
#   3. diff_v                                  the model (§6 steps 4-5)
#
# ONE SEED FIRST. The featHU 2-sigma gate is 0.84 HU and sampling variance stacks
# on top of seed variance; go to 3 seeds only for a configuration that has
# already moved the variance ratio off 0.176.
#
# SAMPLE COUNT. --n_samples 8 to explore (CRPS and the variance ratio are valid
# at any N), but the final calibration pass needs --n_samples 20: a 90% interval
# cannot be expressed by fewer than 19 samples, so below that the coverage column
# is NaN rather than low. 2.5x the sampling cost, once, at the end.

set -uo pipefail

BASE_OUT="../out_synthesis_train/literature_baseline"
STOP_ON_ERROR=1   # set to 0 to keep going after a scenario fails

SCENARIOS=(
  # ── The reference point ────────────────────────────────────────────────────
  # The deterministic best, re-trainable here so every comparison in this repo is
  # against a model trained by THIS code rather than against a number copied out
  # of analysis/BASELINE_REFERENCE.md. Should reproduce featHU ~13.46 at 3 seeds;
  # if it does not, nothing below is comparable to the recorded baseline and that
  # is the first thing to fix.
  "l1_organ_groupnorm|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group"

  # ── Step 2: the calibration baseline diffusion has to beat ─────────────────
  # Same config, plus a (mu, log sigma^2) head and Gaussian NLL instead of L1.
  # ~1% of diffusion's cost. Its per-VOXEL CRPS/coverage is a real bar; its
  # per-ORGAN numbers will be near zero by construction (models_hetero.py says
  # why, in advance). Both are reported — the gap is the point.
  #
  #   python infer_volume.py --scenario_dir <run> --split test
  #   python scripts/calibration_eval.py --mode gaussian --dir <run>/phase_infer
  "hetero_nll|--use_hetero --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group"

  # ── Steps 4-5: conditional diffusion ───────────────────────────────────────
  # group norm throughout, so the comparison against l1_organ_groupnorm changes
  # one thing. v-prediction is the default; x0 is the parameterisation ablation,
  # and reporting both is cheaper than asserting the choice.
  #
  # EXPECTED, stated before running (DIFFUSION_PLAN.md §3, §7):
  #   single DDIM sample  featHU ~1.41x WORSE than 13.46. Not a failure.
  #   mean of N samples   featHU ~13.46, i.e. it collapses to the conditional mean.
  #   var ratio           0.176 -> 1.0. THIS is the result.
  "diff_v|--use_diffusion --parameterisation v --generator_norm group"
  "diff_x0|--use_diffusion --parameterisation x0 --generator_norm group"

  # The organ losses that fixed the deterministic model's levels, applied to the
  # predicted x0 (§5.4). If they help featHU but flatten the variance ratio, they
  # are pulling the sampler back toward the conditional mean — which is worth
  # knowing and is exactly what the var-ratio column will show.
  "diff_v_organ|--use_diffusion --parameterisation v --generator_norm group --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile"

  # No classifier-free guidance. Isolates what the guidance dial costs at
  # training time; a guidance SWEEP is an inference-time flag
  # (infer_volume.py --guidance), not a scenario.
  "diff_v_nocfg|--use_diffusion --parameterisation v --generator_norm group --cfg_drop_prob 0.0"

  # ── The level channel (IMPLEMENTATION.md §2a) ──────────────────────────────
  # BEFORE running any of these, run Gate 0: infer_volume + benchmark.py on the
  # EXISTING diff_* checkpoints. Without raps_hf / grad_w1 / var-ratio for the
  # old runs there is nothing to compare against, and the pixel columns cannot
  # tell "learned texture" from "copied the input" — the samples of diff_v,
  # diff_x0 and diff_v_nocfg are all closer to the NCCT than the NCCT is to the
  # CECT, which is the reading these scenarios exist to change.
  #
  # ONE KEY AT A TIME. These are stacked deliberately in this order so each row
  # differs from the row above it in exactly one thing.
  "diff_v_ablate|--use_diffusion --parameterisation v --generator_norm group --diffusion_offset_noise 0 --min_snr_gamma 0 --ema_decay 0 --diffusion_beta1 0.5 --augment none --diffusion_selection val_loss"
  "diff_v_ema|--use_diffusion --parameterisation v --generator_norm group --diffusion_offset_noise 0 --min_snr_gamma 0 --diffusion_beta1 0.5 --augment none --diffusion_selection val_loss"
  "diff_v_snrw|--use_diffusion --parameterisation v --generator_norm group --diffusion_offset_noise 0 --augment none --diffusion_selection val_loss"
  "diff_v_offset|--use_diffusion --parameterisation v --generator_norm group --augment none --diffusion_selection val_loss"
  "diff_v_detailsel|--use_diffusion --parameterisation v --generator_norm group --augment none"
  "diff_v_aug|--use_diffusion --parameterisation v --generator_norm group"

  # The data lever, which is the largest one identified and has never been
  # exercised: at organ_focus_frac 0 most training patches are body wall, limb
  # and air, where the correct output IS a copy of the input. Labels are the
  # contrast organs — aorta 52, IVC 63, portal+splenic vein 64, plus liver 5,
  # spleen 1, pancreas 7, kidneys 2/3. Confirm against
  # orgFeatXGB_CTPhase/.../ts_label_map_total.json before trusting the ids.
  "diff_v_organfocus|--use_diffusion --parameterisation v --generator_norm group --organ_focus_frac 0.5 --organ_focus_labels 52 63 64 5 1 7 2 3"

  # Everything on, plus the organ losses on the predicted x0 (now with
  # clip=False, so saturated voxels actually contribute a gradient).
  "diff_v_full|--use_diffusion --parameterisation v --generator_norm group --organ_focus_frac 0.5 --organ_focus_labels 52 63 64 5 1 7 2 3 --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile"
  # add more scenarios here, format: "name|--flag1 --flag2 ..."
)

SEEDS="${SEEDS:-42}"

run_one() {
  local name="$1" flags="$2"
  for seed in $SEEDS; do
    local out="${BASE_OUT}_${name}"
    [[ "$seed" != "42" || "$SEEDS" != "42" ]] && out="${out}_s${seed}"
    mkdir -p "$out"
    echo "=== [$(date '+%F %T')] Scenario: $name  seed=$seed  ->  $out ==="
    # shellcheck disable=SC2086
    python train.py --output_dir "$out" --seed "$seed" $flags \
      2>&1 | tee -a "$out/run_scenarios.log"
    local status="${PIPESTATUS[0]}"
    if [[ "$status" -ne 0 ]]; then
      echo "!!! Scenario '$name' seed $seed FAILED (exit $status)"
      [[ "$STOP_ON_ERROR" -eq 1 ]] && exit "$status"
    fi
  done
}

if [[ $# -gt 0 ]]; then
  for want in "$@"; do
    found=0
    for entry in "${SCENARIOS[@]}"; do
      name="${entry%%|*}"; flags="${entry#*|}"
      if [[ "$name" == "$want" ]]; then
        run_one "$name" "$flags"
        found=1
      fi
    done
    [[ "$found" -eq 0 ]] && echo "!!! No scenario named '$want' in SCENARIOS" >&2
  done
else
  for entry in "${SCENARIOS[@]}"; do
    name="${entry%%|*}"; flags="${entry#*|}"
    run_one "$name" "$flags"
  done
fi
