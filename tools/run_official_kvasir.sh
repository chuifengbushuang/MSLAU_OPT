#!/usr/bin/env bash
# Default is validation/preview. Append --run only after the experiment plan is confirmed.
set -euo pipefail
STUDY_REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  3090)
    STUDY_PYTHON="${OFFICIAL_PYTHON:-/home/le/miniconda3/envs/mslau-p51/bin/python}"
    STUDY_DATASET="${OFFICIAL_DATASET:-/data/dataset/zsy/mslau-net/Kvasir-SEG}"
    STUDY_HIERA="${OFFICIAL_HIERA:-/data/models/zsy/mslau-net/pretrained/sam2_hiera_large.fb_r1024.safetensors}"
    STUDY_OUTPUT="${OFFICIAL_OUTPUT:-/data/models/le/mslau-net/official_kvasir_880_120_fp32}"
    STUDY_RUN_NAME="${OFFICIAL_RUN_NAME:-rtx3090_v1}"
    ;;
  5070ti)
    STUDY_WORKSPACE="$(dirname -- "$STUDY_REPOSITORY")"
    STUDY_PYTHON="${OFFICIAL_PYTHON:-$STUDY_WORKSPACE/work/local_train_env/bin/python}"
    STUDY_DATASET="${OFFICIAL_DATASET:-$STUDY_WORKSPACE/datasets/Kvasir-SEG}"
    STUDY_HIERA="${OFFICIAL_HIERA:-$STUDY_WORKSPACE/checkpoints/sam2_hiera_large.fb_r1024.safetensors}"
    STUDY_OUTPUT="${OFFICIAL_OUTPUT:-$STUDY_WORKSPACE/work/official_kvasir_880_120_fp32}"
    STUDY_RUN_NAME="${OFFICIAL_RUN_NAME:-rtx5070ti_v1}"
    ;;
  *)
    printf '%s\n' 'Usage: bash tools/run_official_kvasir.sh {3090|5070ti} [--run] [additional Python options]'
    exit 2
    ;;
esac
shift
exec "$STUDY_PYTHON" -u "$STUDY_REPOSITORY/tools/run_official_kvasir_study.py" \
  --paired-protocol "${OFFICIAL_PAIRED_PROTOCOL:-strict-v2}" \
  --dataset "$STUDY_DATASET" --hiera-checkpoint "$STUDY_HIERA" \
  --output-root "$STUDY_OUTPUT" --run-name "$STUDY_RUN_NAME" \
  --gpu "${OFFICIAL_GPU:-0}" --epochs 100 --patience 10 "$@"
