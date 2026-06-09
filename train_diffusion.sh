#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RESULT_ROOT="${RESULT_ROOT:-checkpoint}"
SIT_CONFIG="${SIT_CONFIG:-TrainCPLightSiT_Minimal}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:0}"
DIFFUSION_RUN_MANIFEST="${DIFFUSION_RUN_MANIFEST:-${DIFFUSION_SWEEP_MANIFEST:-${RESULT_ROOT}/diffusion_runs.tsv}}"
DIFFUSION_NOTE="${DIFFUSION_NOTE:-cplightsit_lr1e4}"

if [[ -z "${RAY_CHECKPOINT:-}" ]]; then
  echo "RAY_CHECKPOINT must point to the selected ray_encoder_best.pth." >&2
  exit 1
fi
if [[ ! -f "${RAY_CHECKPOINT}" ]]; then
  echo "RAY_CHECKPOINT does not exist: ${RAY_CHECKPOINT}" >&2
  exit 1
fi

COMMON_OVERRIDE_ARGS=()
DIFFUSION_COMMON_ARGS=()

if [[ -n "${COMMON_OVERRIDES:-}" ]]; then
  read -r -a COMMON_OVERRIDE_ARGS <<< "${COMMON_OVERRIDES}"
fi
if [[ -n "${DIFFUSION_COMMON_OVERRIDES:-}" ]]; then
  read -r -a DIFFUSION_COMMON_ARGS <<< "${DIFFUSION_COMMON_OVERRIDES}"
elif [[ -n "${SIT_ARGS:-}" ]]; then
  read -r -a DIFFUSION_COMMON_ARGS <<< "${SIT_ARGS}"
elif [[ -n "${SIT_OVERRIDES:-}" ]]; then
  read -r -a DIFFUSION_COMMON_ARGS <<< "${SIT_OVERRIDES}"
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
normalize_overrides DIFFUSION_COMMON_ARGS

mkdir -p "${RESULT_ROOT}"
: > "${DIFFUSION_RUN_MANIFEST}"

echo "[diffusion] CP-LightSiT fine-tuning"
echo "  ray encoder: ${RAY_CHECKPOINT}"
echo "  config: ${SIT_CONFIG}"
echo "  note: ${DIFFUSION_NOTE}"

run_ddp "${SIT_CONFIG}" \
  "result_root=${RESULT_ROOT}" \
  "ray_encoder_checkpoint=${RAY_CHECKPOINT}" \
  "note=${DIFFUSION_NOTE}" \
  "${COMMON_OVERRIDE_ARGS[@]}" \
  "${DIFFUSION_COMMON_ARGS[@]}"

pointer="${RESULT_ROOT}/latest_CPLightSiT.txt"
if [[ ! -f "${pointer}" ]]; then
  echo "CP-LightSiT latest pointer was not created: ${pointer}" >&2
  exit 1
fi
run_dir="$(tr -d '[:space:]' < "${pointer}")"
checkpoint="${run_dir}/checkpoint/best.pth"
if [[ ! -f "${checkpoint}" ]]; then
  echo "CP-LightSiT checkpoint was not created: ${checkpoint}" >&2
  exit 1
fi
score="$(checkpoint_score "${checkpoint}")"
printf "%s\t%s\t%s\t%s\t%s\n" "single" "${score}" "${run_dir}" "${checkpoint}" "config=${SIT_CONFIG} note=${DIFFUSION_NOTE}" >> "${DIFFUSION_RUN_MANIFEST}"

echo "CP-LightSiT run: ${run_dir}"
echo "Best score: ${score}"
echo "Diffusion run manifest: ${DIFFUSION_RUN_MANIFEST}"
