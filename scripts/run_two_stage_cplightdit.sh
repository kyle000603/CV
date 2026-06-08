#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
DATA_ROOT="${DATA_ROOT:-data/VIDIT_HF}"
RESULT_ROOT="${RESULT_ROOT:-checkpoint}"

GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
USE_TORCHRUN="${USE_TORCHRUN:-1}"

RAY_CONFIG="${RAY_CONFIG:-TrainRayEncoder}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-TrainCPLightSiT}"

RAY_EPOCHS="${RAY_EPOCHS:-3}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"

RAY_BATCH_SIZE="${RAY_BATCH_SIZE:-128}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"

MAX_PAIRS_PER_SCENE="${MAX_PAIRS_PER_SCENE:-8}"

LAMBDA_TRANSFER="${LAMBDA_TRANSFER:-0.1}"
USE_GT_SOURCE_LIGHT_PROB="${USE_GT_SOURCE_LIGHT_PROB:-0.5}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

echo "[1/4] Preparing assets..."
"${PYTHON_BIN}" scripts/prepare_cplightsit_assets.py \
  dataset.train.root="${DATA_ROOT}" \
  dataset.val.root="${DATA_ROOT}" \
  assets.hf_vidit.root="${DATA_ROOT}"

echo "[2/4] Stage 1: RayEncoder pretraining..."
"${PYTHON_BIN}" train.py -cn "${RAY_CONFIG}" \
  dataset.train.root="${DATA_ROOT}" \
  dataset.val.root="${DATA_ROOT}" \
  assets.hf_vidit.root="${DATA_ROOT}" \
  result_root="${RESULT_ROOT}" \
  epochs="${RAY_EPOCHS}" \
  batch_size="${RAY_BATCH_SIZE}" \
  dataloader.global_batch_size="${RAY_BATCH_SIZE}" \
  +dataset.train.max_pairs_per_scene="${MAX_PAIRS_PER_SCENE}" \
  +dataset.val.max_pairs_per_scene="${MAX_PAIRS_PER_SCENE}"

RAY_POINTER="${RESULT_ROOT}/latest_RayEncoder.txt"

if [[ ! -f "${RAY_POINTER}" ]]; then
  echo "ERROR: RayEncoder pointer not found: ${RAY_POINTER}" >&2
  exit 1
fi

RAY_RUN_DIR="$(tr -d '[:space:]' < "${RAY_POINTER}")"
RAY_CKPT="${RAY_RUN_DIR}/ray_encoder_latest.pth"

if [[ ! -f "${RAY_CKPT}" ]]; then
  echo "ERROR: RayEncoder checkpoint not found: ${RAY_CKPT}" >&2
  echo "Available files in ${RAY_RUN_DIR}:" >&2
  ls -lah "${RAY_RUN_DIR}" >&2 || true
  exit 1
fi

echo "[3/4] RayEncoder checkpoint: ${RAY_CKPT}"

echo "[4/4] Stage 2: CP-LightSiT minimal finetuning..."

COMMON_ARGS=(
  train.py -cn "${FINETUNE_CONFIG}"
  dataset.train.root="${DATA_ROOT}"
  dataset.val.root="${DATA_ROOT}"
  assets.hf_vidit.root="${DATA_ROOT}"
  result_root="${RESULT_ROOT}"
  stage=cplightsit_finetune
  loss_mode=minimal
  ray_encoder_checkpoint="${RAY_CKPT}"
  freeze_backbone=true
  freeze_ray_encoder=true
  freeze_tokenizer=true
  train_condition_adapters_only=true
  train_light_transfer_transformer=true
  lambda_flow=1.0
  lambda_transfer="${LAMBDA_TRANSFER}"
  use_gt_source_light_prob="${USE_GT_SOURCE_LIGHT_PROB}"
  enable_image_space_losses=false
  decode_loss_every=0
  epochs="${FINETUNE_EPOCHS}"
  batch_size="${FINETUNE_BATCH_SIZE}"
  dataloader.global_batch_size="${FINETUNE_BATCH_SIZE}"
  +dataset.train.max_pairs_per_scene="${MAX_PAIRS_PER_SCENE}"
  +dataset.val.max_pairs_per_scene="${MAX_PAIRS_PER_SCENE}"
)

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS}" "${TORCHRUN_BIN}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${COMMON_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "${PYTHON_BIN}" "${COMMON_ARGS[@]}"
fi

echo "Two-stage CP-LightSiT training completed."
