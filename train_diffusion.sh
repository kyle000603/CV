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
DIFFUSION_SWEEP_MANIFEST="${DIFFUSION_SWEEP_MANIFEST:-${RESULT_ROOT}/diffusion_sweep_runs.tsv}"

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
: > "${DIFFUSION_SWEEP_MANIFEST}"

DIFFUSION_SWEEPS=(
  "flow_start_mode=source edit_noise_scale=0.00 batch_size=128 dataloader.global_batch_size=128 lr=0.000012 adapter_lr=0.000012 light_transfer_lr=0.000010 backbone_lr=0.000002 train_diffusion_backbone=true train_diffusion_last_n_blocks=4 lambda_transfer=0.03 use_gt_source_light_prob=0.75 grad_clip_norm=0.30 q_clip=1.5 transfer_smooth_l1_beta=0.2"
  "flow_start_mode=source edit_noise_scale=0.02 batch_size=128 dataloader.global_batch_size=128 lr=0.000010 adapter_lr=0.000012 light_transfer_lr=0.000008 backbone_lr=0.000002 train_diffusion_backbone=true train_diffusion_last_n_blocks=8 lambda_transfer=0.03 use_gt_source_light_prob=0.75 grad_clip_norm=0.30 q_clip=1.5 transfer_smooth_l1_beta=0.2"
  "flow_start_mode=source edit_noise_scale=0.02 batch_size=128 dataloader.global_batch_size=128 lr=0.000008 adapter_lr=0.000010 light_transfer_lr=0.000008 backbone_lr=0.000001 train_diffusion_backbone=true train_diffusion_last_n_blocks=12 lambda_transfer=0.02 use_gt_source_light_prob=0.75 grad_clip_norm=0.25 q_clip=1.25 transfer_smooth_l1_beta=0.25"
  "flow_start_mode=source edit_noise_scale=0.05 batch_size=128 dataloader.global_batch_size=128 lr=0.000006 adapter_lr=0.000008 light_transfer_lr=0.000006 backbone_lr=0.000001 train_diffusion_backbone=true train_diffusion_last_n_blocks=16 lambda_transfer=0.02 use_gt_source_light_prob=1.0 grad_clip_norm=0.20 q_clip=1.25 transfer_smooth_l1_beta=0.25"
  "flow_start_mode=source edit_noise_scale=0.00 batch_size=128 dataloader.global_batch_size=128 lr=0.000010 adapter_lr=0.000012 light_transfer_lr=0.000008 backbone_lr=0.000003 train_diffusion_backbone=true train_diffusion_last_n_blocks=8 lambda_transfer=0.05 use_gt_source_light_prob=1.0 grad_clip_norm=0.30 q_clip=1.5 transfer_smooth_l1_beta=0.2"
  "flow_start_mode=source edit_noise_scale=0.02 batch_size=128 dataloader.global_batch_size=128 lr=0.000008 adapter_lr=0.000010 light_transfer_lr=0.000006 backbone_lr=0.000002 train_diffusion_backbone=true train_diffusion_last_n_blocks=12 lambda_transfer=0.05 use_gt_source_light_prob=1.0 grad_clip_norm=0.25 q_clip=1.5 transfer_smooth_l1_beta=0.2"
)

for index in "${!DIFFUSION_SWEEPS[@]}"; do
  run_id="$(printf "%02d" "$((index + 1))")"
  read -r -a SWEEP_ARGS <<< "${DIFFUSION_SWEEPS[$index]}"
  echo "[diffusion ${run_id}/06] CP-LightSiT sweep"
  echo "  ray encoder: ${RAY_CHECKPOINT}"
  echo "  overrides: ${DIFFUSION_SWEEPS[$index]}"
  run_ddp "${SIT_CONFIG}" \
    "result_root=${RESULT_ROOT}" \
    "ray_encoder_checkpoint=${RAY_CHECKPOINT}" \
    "note=cplightsit_sweep_${run_id}" \
    "${COMMON_OVERRIDE_ARGS[@]}" \
    "${DIFFUSION_COMMON_ARGS[@]}" \
    "${SWEEP_ARGS[@]}"

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
  printf "%s\t%s\t%s\t%s\t%s\n" "${run_id}" "${score}" "${run_dir}" "${checkpoint}" "${DIFFUSION_SWEEPS[$index]}" >> "${DIFFUSION_SWEEP_MANIFEST}"
  echo "  run: ${run_dir}"
  echo "  best score: ${score}"
done

echo "Diffusion sweep manifest: ${DIFFUSION_SWEEP_MANIFEST}"
