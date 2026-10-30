#!/usr/bin/env bash
# Batch convenience wrapper around the original single-experiment launcher.
set -euo pipefail
STUDY_REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  3090)
    STUDY_PYTHON="${OFFICIAL_PYTHON:-/home/le/miniconda3/envs/mslau-p51/bin/python}"
    STUDY_DATASET="${OFFICIAL_DATASET:-/data/dataset/zsy/mslau-net/Kvasir-SEG}"
    STUDY_HIERA="${OFFICIAL_HIERA:-/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors}"
    STUDY_OUTPUT="${OFFICIAL_OUTPUT:-/data/models/le/mslau-net/official_kvasir_880_120_original20}"
    STUDY_RUN_NAME="${OFFICIAL_RUN_NAME:-rtx3090_original20_v1}"
    ;;
  5070ti)
    STUDY_WORKSPACE="$(dirname -- "$STUDY_REPOSITORY")"
    STUDY_PYTHON="${OFFICIAL_PYTHON:-$STUDY_WORKSPACE/work/local_train_env/bin/python}"
    STUDY_DATASET="${OFFICIAL_DATASET:-$STUDY_WORKSPACE/datasets/Kvasir-SEG}"
    STUDY_HIERA="${OFFICIAL_HIERA:-$STUDY_WORKSPACE/checkpoints/sam2_hiera_large.fb_r1024.safetensors}"
    STUDY_OUTPUT="${OFFICIAL_OUTPUT:-$STUDY_WORKSPACE/work/official_kvasir_880_120_original20}"
    STUDY_RUN_NAME="${OFFICIAL_RUN_NAME:-rtx5070ti_original20_v1}"
    ;;
  *)
    printf '%s\n' 'Usage: bash tools/run_official_kvasir.sh {3090|5070ti} [--run] [additional Python options]'
    exit 2
    ;;
esac
shift
STUDY_RUN_ROOT="$STUDY_OUTPUT/$STUDY_RUN_NAME"
if [[ " $* " == *" --run "* && -e "$STUDY_RUN_ROOT" ]]; then
  printf '%s\n' "Run already exists: $STUDY_RUN_ROOT" >&2
  exit 1
fi
for STUDY_SEED in 1234 42 2026; do
  for STUDY_EXPERIMENT in exp00_baseline exp03_naf_p12 exp09_naf_discrete \
      exp00_baseline_paired_strict_v2 exp03_naf_p12_paired_strict_v2 exp09_naf_discrete_paired_strict_v2; do
    export P51_OUTPUT_ROOT="$STUDY_RUN_ROOT/$STUDY_EXPERIMENT"
    "$STUDY_PYTHON" -u "$STUDY_REPOSITORY/tools/run_experiment.py" "$STUDY_EXPERIMENT" \
      --split-dir "$STUDY_REPOSITORY/configs/splits/kvasir_medico_official_880_120_v1" \
      --seed "$STUDY_SEED" --gpu "${OFFICIAL_GPU:-0}" \
      --dataset "$STUDY_DATASET" --hiera-checkpoint "$STUDY_HIERA" "$@"
  done
done
