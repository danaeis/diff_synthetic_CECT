#!/usr/bin/env bash
# Generate FULL multi-label TotalSegmentator masks for every volume in the
# synthesis deeds-aligned data directory.
#
# Replaces the old per-volume `vindr_ts_get_stats.sh` (which dumped stats pkls
# with an incomplete organ set). Here we save complete `total`-task masks that
# INCLUDE the aorta/heart/IVC, next to each volume as `<vol>_deeds_seg_full.nii.gz`.
#
# Resumable: skips any volume whose output mask already exists, so you can Ctrl-C
# and re-run. Sequential by default (one TS per GPU — safest); set JOBS>1 only if
# you know the GPU/RAM can take concurrent TotalSegmentator processes.
#
# Usage:
#   ./run_ts_masks.sh <DATA_DIR> [FILE_TAG] [SEG_SUFFIX] [FULL] [JOBS]
# Examples:
#   ./run_ts_masks.sh ../../../sample_data_reg/ncct_cect/vindr_ds/all_baseline_algorithms/B2_deeds__aligned
#   ./run_ts_masks.sh /abs/deeds_dir _deeds _seg_full 0 1
#
# Args:
#   DATA_DIR    (required) per-case subdirs with *<FILE_TAG>.nii.gz volumes
#   FILE_TAG    default _deeds   (volume suffix before .nii.gz)
#   SEG_SUFFIX  default _seg_full (output = <vol-without-.nii.gz>+SEG_SUFFIX+.nii.gz)
#   FULL        default 0        (1 = full-res TS; 0 = fast/3mm, matches the model)
#   JOBS        default 1        (parallel TS processes — keep 1 on a single GPU)

set -uo pipefail

DATA_DIR="${1:?usage: ./run_ts_masks.sh <DATA_DIR> [FILE_TAG] [SEG_SUFFIX] [FULL] [JOBS]}"
FILE_TAG="${2:-_deeds}"
SEG_SUFFIX="${3:-_seg_full}"
FULL="${4:-0}"
JOBS="${5:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

full_flag=""
[[ "$FULL" == "1" ]] && full_flag="--full"

# Discover volumes: *<FILE_TAG>.nii.gz, excluding segmentation masks and DVFs.
# (while-read instead of mapfile, so it also works on bash 3.2.)
VOLS=()
while IFS= read -r _v; do VOLS+=("$_v"); done < <(
  find "$DATA_DIR" -type f -name "*${FILE_TAG}.nii.gz" \
       ! -name "*_seg*" ! -name "*_dvf*" | sort)
total="${#VOLS[@]}"
if [[ "$total" -eq 0 ]]; then
  echo "No *${FILE_TAG}.nii.gz volumes under $DATA_DIR" >&2
  exit 1
fi
echo "Found $total volumes under $DATA_DIR (FULL=$FULL, JOBS=$JOBS, out suffix=${FILE_TAG}${SEG_SUFFIX})"

out_path() {   # <vol>_deeds.nii.gz -> <vol>_deeds_seg_full.nii.gz
  echo "${1%${FILE_TAG}.nii.gz}${FILE_TAG}${SEG_SUFFIX}.nii.gz"
}

if [[ "$JOBS" -gt 1 ]] && command -v parallel >/dev/null 2>&1; then
  echo "Running with GNU parallel -j $JOBS …"
  export SCRIPT_DIR full_flag FILE_TAG SEG_SUFFIX
  printf '%s\n' "${VOLS[@]}" | parallel -j "$JOBS" --bar '
    vol={};
    out="${vol%'"${FILE_TAG}"'.nii.gz}'"${FILE_TAG}${SEG_SUFFIX}"'.nii.gz";
    [ -f "$out" ] && exit 0;
    python "'"$SCRIPT_DIR"'/ts_get_mask.py" "$vol" "$out" '"$full_flag"''
else
  [[ "$JOBS" -gt 1 ]] && echo "GNU parallel not found — running sequentially."
  done=0; made=0; skipped=0; failed=0
  for vol in "${VOLS[@]}"; do
    done=$((done+1))
    out="$(out_path "$vol")"
    if [[ -f "$out" ]]; then
      skipped=$((skipped+1))
    else
      echo "[$done/$total] $(basename "$vol")"
      if python "$SCRIPT_DIR/ts_get_mask.py" "$vol" "$out" $full_flag; then
        made=$((made+1))
      else
        failed=$((failed+1)); echo "  !! failed" >&2
      fi
    fi
  done
  echo "Done: $made generated, $skipped already existed, $failed failed (of $total)."
fi
