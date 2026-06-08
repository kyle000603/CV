#!/usr/bin/env bash
set -euo pipefail

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CUDA_DEVICES="${CUDA_DEVICES-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RESULT_ROOT="${RESULT_ROOT:-checkpoint}"
RAY_CONFIG="${RAY_CONFIG:-TrainRayEncoder}"
SIT_CONFIG="${SIT_CONFIG:-TrainCPLightSiT_Minimal}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:0}"

COMMON_OVERRIDE_ARGS=()
RAY_OVERRIDE_ARGS=()
SIT_OVERRIDE_ARGS=()

if [[ -n "${COMMON_OVERRIDES:-}" ]]; then
  read -r -a COMMON_OVERRIDE_ARGS <<< "${COMMON_OVERRIDES}"
fi
if [[ -n "${RAY_ARGS:-}" ]]; then
  read -r -a RAY_OVERRIDE_ARGS <<< "${RAY_ARGS}"
elif [[ -n "${RAY_OVERRIDES:-}" ]]; then
  read -r -a RAY_OVERRIDE_ARGS <<< "${RAY_OVERRIDES}"
fi
if [[ -n "${SIT_ARGS:-}" ]]; then
  read -r -a SIT_OVERRIDE_ARGS <<< "${SIT_ARGS}"
elif [[ -n "${SIT_OVERRIDES:-}" ]]; then
  read -r -a SIT_OVERRIDE_ARGS <<< "${SIT_OVERRIDES}"
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

normalize_overrides COMMON_OVERRIDE_ARGS
normalize_overrides RAY_OVERRIDE_ARGS
normalize_overrides SIT_OVERRIDE_ARGS

run_ddp() {
  local config_name="$1"
  shift
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN_BIN}" \
    --rdzv-backend="${RDZV_BACKEND}" \
    --rdzv-endpoint="${RDZV_ENDPOINT}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py -cn "${config_name}" "$@"
}

echo "[1/2] Pretraining RayEncoder with DDP"
run_ddp "${RAY_CONFIG}" \
  "result_root=${RESULT_ROOT}" \
  "${COMMON_OVERRIDE_ARGS[@]}" \
  "${RAY_OVERRIDE_ARGS[@]}"

RAY_POINTER="${RESULT_ROOT}/latest_RayEncoder.txt"
if [[ ! -f "${RAY_POINTER}" ]]; then
  echo "RayEncoder latest pointer was not created: ${RAY_POINTER}" >&2
  exit 1
fi

RAY_RUN_DIR="$(tr -d '[:space:]' < "${RAY_POINTER}")"
RAY_CHECKPOINT="${RAY_RUN_DIR}/checkpoint/ray_encoder_best.pth"
if [[ ! -f "${RAY_CHECKPOINT}" ]]; then
  echo "RayEncoder checkpoint was not created: ${RAY_CHECKPOINT}" >&2
  exit 1
fi

echo "[2/2] Finetuning CP-LightSiT with DDP"
run_ddp "${SIT_CONFIG}" \
  "result_root=${RESULT_ROOT}" \
  "ray_encoder_checkpoint=${RAY_CHECKPOINT}" \
  "${COMMON_OVERRIDE_ARGS[@]}" \
  "${SIT_OVERRIDE_ARGS[@]}"

SIT_POINTER="${RESULT_ROOT}/latest_CPLightSiT.txt"
if [[ -f "${SIT_POINTER}" ]]; then
  SIT_RUN_DIR="$(tr -d '[:space:]' < "${SIT_POINTER}")"
  echo "CP-LightSiT run: ${SIT_RUN_DIR}"
fi
echo "RayEncoder checkpoint: ${RAY_CHECKPOINT}"
