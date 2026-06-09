#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RESULT_ROOT="${RESULT_ROOT:-checkpoint}"
RAY_CONFIG="${RAY_CONFIG:-TrainRayEncoder}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:0}"
ENCODER_RUN_MANIFEST="${ENCODER_RUN_MANIFEST:-${ENCODER_SWEEP_MANIFEST:-${RESULT_ROOT}/encoder_runs.tsv}}"
ENCODER_NOTE="${ENCODER_NOTE:-ray_encoder_vitb_lr1e4}"

COMMON_OVERRIDE_ARGS=()
ENCODER_COMMON_ARGS=()

if [[ -n "${COMMON_OVERRIDES:-}" ]]; then
  read -r -a COMMON_OVERRIDE_ARGS <<< "${COMMON_OVERRIDES}"
fi
if [[ -n "${ENCODER_COMMON_OVERRIDES:-}" ]]; then
  read -r -a ENCODER_COMMON_ARGS <<< "${ENCODER_COMMON_OVERRIDES}"
elif [[ -n "${RAY_ARGS:-}" ]]; then
  read -r -a ENCODER_COMMON_ARGS <<< "${RAY_ARGS}"
elif [[ -n "${RAY_OVERRIDES:-}" ]]; then
  read -r -a ENCODER_COMMON_ARGS <<< "${RAY_OVERRIDES}"
fi

normalize_overrides() {
  local -n values_ref="$1"
  local value
  for i in "${!values_ref[@]}"; do
    value="${values_ref[$i]}"
    case "${value}" in
      +dataset.train.max_pairs_per_scene=*|+dataset.val.max_pairs_per_scene=*|+dataset.test.max_pairs_per_scene=*)
        values_ref[$i]="${value#+}"
        ;;
    esac
  done
}

checkpoint_score() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import math
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
score = checkpoint.get("best_score", checkpoint.get("score"))
if score is None or not math.isfinite(float(score)):
    raise SystemExit(f"checkpoint has no finite best score: {path}")
print(float(score))
PY
}

run_ddp() {
  local config_name="$1"
  shift
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN_BIN}" \
    --rdzv-backend="${RDZV_BACKEND}" \
    --rdzv-endpoint="${RDZV_ENDPOINT}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py -cn "${config_name}" "$@"
}

normalize_overrides COMMON_OVERRIDE_ARGS
normalize_overrides ENCODER_COMMON_ARGS

mkdir -p "${RESULT_ROOT}"
: > "${ENCODER_RUN_MANIFEST}"

echo "[encoder] TrainRayEncoder pretraining"
echo "  config: ${RAY_CONFIG}"
echo "  note: ${ENCODER_NOTE}"

run_ddp "${RAY_CONFIG}" \
  "result_root=${RESULT_ROOT}" \
  "note=${ENCODER_NOTE}" \
  "${COMMON_OVERRIDE_ARGS[@]}" \
  "${ENCODER_COMMON_ARGS[@]}"

pointer="${RESULT_ROOT}/latest_RayEncoder.txt"
if [[ ! -f "${pointer}" ]]; then
  echo "RayEncoder latest pointer was not created: ${pointer}" >&2
  exit 1
fi
run_dir="$(tr -d '[:space:]' < "${pointer}")"
checkpoint="${run_dir}/checkpoint/ray_encoder_best.pth"
if [[ ! -f "${checkpoint}" ]]; then
  echo "RayEncoder checkpoint was not created: ${checkpoint}" >&2
  exit 1
fi
score="$(checkpoint_score "${checkpoint}")"
printf "%s\t%s\t%s\t%s\t%s\n" "single" "${score}" "${run_dir}" "${checkpoint}" "config=${RAY_CONFIG} note=${ENCODER_NOTE}" >> "${ENCODER_RUN_MANIFEST}"

echo "RayEncoder run: ${run_dir}"
echo "Best score: ${score}"
echo "Encoder run manifest: ${ENCODER_RUN_MANIFEST}"
