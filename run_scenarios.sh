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
#        -> tile count per volume. The whole sampling budget depends on it.
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
  "diff_l1_organ_groupnorm|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group"
  "diff_l1_organ_groupnorm_adv|--use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group  --use_adversarial --use_cond_disc --adv_warmup_epochs 15"

  # ── Step 2: the calibration baseline diffusion has to beat ─────────────────
  # Same config, plus a (mu, log sigma^2) head and Gaussian NLL instead of L1.
  # ~1% of diffusion's cost. Its per-VOXEL CRPS/coverage is a real bar; its
  # per-ORGAN numbers will be near zero by construction (models_hetero.py says
  # why, in advance). Both are reported — the gap is the point.
  #
  #   python infer_volume.py --scenario_dir <run> --split test
  #   python scripts/calibration_eval.py --mode gaussian --dir <run>/phase_infer
  "diff_hetero_nll|--use_hetero --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_l1_decay --generator_norm group"

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
  # Adversarial critic on the one-step x0 estimate (DIFFUSION_PLAN.md §11).
  # RUN THE LAMBDA SWEEP, NOT JUST THE DEFAULT. lambda_adv=2.0 is inherited from
  # the GAN baseline, where it competed with an L1 term at lambda=100. Here it is
  # added to a diffusion MSE of order 1e-2..1, so at 2.0 the adversarial term is
  # the DOMINANT gradient after warmup — a strictly bigger intervention than the
  # same number was in the baseline. Sweep DOWN first; _lam05 is the one to run
  # if only one adversarial run is affordable.
  "diff_v_organ_adv|--use_diffusion --parameterisation v --generator_norm group --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile --use_adversarial --use_cond_disc --adv_warmup_epochs 15"
  "diff_v_organ_adv_lam05|--use_diffusion --parameterisation v --generator_norm group --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lambda_adv 0.5"
  "diff_v_organ_adv_lam01|--use_diffusion --parameterisation v --generator_norm group --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lambda_adv 0.1"

  # ── 2.5-D input (DIFFUSION_PLAN.md §11) ────────────────────────────────────
  # 2k+1 adjacent axial slices as CHANNELS of the conditioning input; the target
  # is still the centre slice. z is the one geometric deficiency that survived
  # measurement (config.py N_INPUT_SLICES): patch_depth=1 shows the model 1.5 mm
  # of an aorta that runs 258 mm. k=2 buys 7.5 mm for ~no extra parameters.
  #
  # TWO THINGS BEFORE RUNNING THESE:
  #  * They REBUILD THE PATCH CACHE. n_input_slices is part of the cache key
  #    (dataset.py), so the first 2.5-D scenario pays the full preload, and the
  #    2-D cache stays valid for everything above.
  #  * RAM scales with n. The source cache is 20k x n x 128 x 128 x f32
  #    = ~1.3 GB x n (targets and masks do not scale). n=5 is ~6.6 GB, n=11 is
  #    ~14.4 GB. Lower --max_train_patches if the preload gets OOM-killed.
  #
  # The non-adversarial twin comes first — an adversarial 2.5-D run on its own
  # cannot separate "2.5-D helped" from "the critic helped".
  "diff_v_organ_slices5|--use_diffusion --parameterisation v --generator_norm group --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile --n_input_slices 5"
  "diff_v_organ_adv_slices5|--use_diffusion --parameterisation v --generator_norm group --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile --n_input_slices 5 --use_adversarial --use_cond_disc --adv_warmup_epochs 15 --lambda_adv 0.5"

  # The z-extent sweep. Only worth it if slices5 moved something above the
  # featHU gate; it is the most expensive row here (cache RAM and preload time).
  "diff_v_organ_slices11|--use_diffusion --parameterisation v --generator_norm group --use_organ --use_per_organ_weights --organ_weight_preset tiered --use_hu_profile --n_input_slices 11"

  # No classifier-free guidance. Isolates what the guidance dial costs at
  # training time; a guidance SWEEP is an inference-time flag
  # (infer_volume.py --guidance), not a scenario.
  "diff_v_nocfg|--use_diffusion --parameterisation v --generator_norm group --cfg_drop_prob 0.0"
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
