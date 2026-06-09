# CP-LightSiT

CP-LightSiT is a minimal standalone training codebase for computational-photography-guided light transfer conditioning in a Diffusion Transformer relighting pipeline. It takes a VIDIT source image under source light, estimates the source light ray, rotates that ray toward the target light direction, builds a physics-guided source-to-target light transfer field, refines that dense field with a small transformer, and conditions a token-space DiT through rectified flow.

## Folder Structure

- `train.py`: Hydra entrypoint that prepares assets before DDP setup, then starts training.
- `inference.py`: checkpoint evaluator that writes metric summaries and good/bad samples into one PDF report.
- `configs/`: dataset, dataloader, model, loss, optimizer, diffusion, trainer, and wandb config groups.
- `dataset/data/`: VIDIT relighting pair dataset.
- `dataset/dataloader/`: seeded dataloader and distributed sampler setup.
- `models/modules/`: light utilities, ray encoder, pretrained VAE tokenizer, physics transfer, and dense transfer transformer.
- `models/diffusion/`: DiT and CP-LightSiT conditioning backbone.
- `rectified_flow/`: rectified-flow and trajectory-flow objectives.
- `losses/`: flow MSE and dense light-transfer loss.
- `trainers/`: training loop, logging, optimizer, and checkpointing.
- `scripts/`: sanity check and sampling scripts.

## VIDIT Format

The dataset expects same-scene images with different light directions and matching color temperature. Images are loaded in `[-1, 1]`; light directions are encoded continuously as `[cos(phi), sin(phi), T_norm]`, where `T_norm = (temperature - 5500) / 2500`.

Each pair also returns:

- `source_ray` and `target_ray`: normalized 3D directional rays.
- `source_angle`, `target_angle`, and `delta_angle`: azimuth metadata used to rotate the source ray into the target ray.
- `depth_valid`: whether the depth map is real or a zero placeholder.

Supported filename examples:

```text
scene001_N_5500.png
scene001_dir_N_temp_5500.png
0001_N_4500_rgb.png
scene-0001_light-NE_temp-6500.png
```

Metadata files are checked first: `metadata.json`, `metadata.csv`, `train.json`, `val.json`, `test.json`, and `index.json`. If no metadata is found, image filenames are scanned recursively. If parsing fails, the dataset raises a clear error with examples and expected patterns.

The default data source is the Hugging Face `Nahrawy/VIDIT-Depth-ControlNet` dataset. It is converted into the filename format above under `data/VIDIT_HF/train` and `data/VIDIT_HF/val`, preserving the real `scene`, `direction`, `temprature`, image, and depth-map fields. This is preferred for the ray-rotation training objective because it exposes explicit direction and color-temperature labels.

The official NTIRE Track1 zip layout is still supported as a fallback. Matching files are paired from folders such as `train/input/Image001.png` and `train/gt/Image001.png`, or `validation/Image301.png` and `validation_gt/Image301.png`. Track1 `.npy` files are loaded as depth maps. Because these challenge archives do not expose explicit light direction labels, Track1 fallback metadata uses pseudo-light metadata: source `N`, target `E`, and temperature `5500`.

## Automatic Assets

By default, `train.py` prepares assets before constructing datasets and models:

- Hugging Face VIDIT is expected under `data/VIDIT_HF`. If it is not prepared yet, `Nahrawy/VIDIT-Depth-ControlNet` is downloaded through `datasets` and converted into local PNG/depth pairs.
- Hugging Face cache files are stored under `data/hf_cache`.
- Converted dataset markers are written under `data/VIDIT_HF/.cplightsit_assets`, so later runs reuse the processed files immediately.
- EPFL/NTIRE zip downloads are disabled by default because `assets.hf_vidit.enabled=true` is the preferred path. You can re-enable `assets.vidit.enabled=true` if you want to use local Track1 zips.
- A pretrained Stable-Diffusion VAE is stored under `pretrained/vae/sd-vae-ft-mse` and used as the frozen image tokenizer.
- Pretrained SiT-L/2 weights are stored under `pretrained/SiT-L-2-256x256.pt`. Existing files are reused. The default source is the Hugging Face `nyu-visionx/SiT-collections` mirror.
- Training checkpoints are saved under numbered run directories like `checkpoint/001_CPLightSiT`, `checkpoint/002_CPLightSiT`, and so on.

The default pretrained backbone is SiT-L/2. The loader imports only tensors whose names and shapes match CP-LightSiT; incompatible image-latent patch weights are skipped safely.

Runs made before the VAE tokenizer change used a temporary conv tokenizer and should be treated as invalid for visual inference. Start a new `CP-LightSiT` run after the VAE assets are prepared.

## Default Config

The default Hydra config is `configs/TrainCPLightSiT.yaml`.

- Model config: `configs/model/CP_LightSiT_L.yaml`
- Model size: CP-LightSiT L, `depth=24`, `hidden_size=1024`, `num_heads=16`
- Image size: `256`
- VAE latent: `4 x 32 x 32`
- Token grid: `16 x 16`
- Feature dim: `16` (`4` latent channels patchified with `patch_size=2`)
- Batch size: `128`
- Learning rates: `1e-4` for condition cross-attention, light-transfer transformer, and full diffusion backbone fine-tuning
- Data loading: `24` workers, `prefetch_factor=8`
- Precision/performance: `bf16` AMP, CUDA batch prefetch, channels-last image tensors
- Conditioning: cross-attention over light and dense-transfer context tokens only; source latents are used as the flow start, not as cross-attention context
- Flow start: source VAE latent tokens with `edit_noise_scale=0.02`
- Epochs: `30` for the default diffusion run; extend the same run to `60` if samples and loss are still improving
- Dataset root: `data/VIDIT_HF`
- Hugging Face dataset: `Nahrawy/VIDIT-Depth-ControlNet`
- Hugging Face cache: `data/hf_cache`
- Pretrained VAE: `pretrained/vae/sd-vae-ft-mse` from `stabilityai/sd-vae-ft-mse`
- Pretrained SiT checkpoint: `pretrained/SiT-L-2-256x256.pt` from `nyu-visionx/SiT-collections/SiT-L-2-256.pt`
- Training checkpoints: numbered directories under `checkpoint/`, for example `checkpoint/001_CPLightSiT`
- Training log file: `log.txt` inside each numbered checkpoint directory
- Latest run pointer: `checkpoint/latest_CPLightSiT.txt`
- W&B mode: `online`, automatically disabled if `WANDB_API_KEY` is not set

Install the Hugging Face and VAE dependencies in the CV environment before the first training run:

```bash
python -m pip install -U datasets huggingface_hub pyarrow diffusers transformers accelerate safetensors
```

`train.py` prepares the dataset and pretrained checkpoint before DDP process-group setup. With `torchrun`, rank 0 performs the Hugging Face conversion and SiT checkpoint download first, while the other ranks wait for a local preflight marker. Once preparation is complete, normal DDP initialization and training begin.

You can also prepare the dataset and pretrained checkpoint manually in a single process:

```bash
HF_HUB_DISABLE_XET=1 python scripts/prepare_cplightsit_assets.py
```

Disable automatic asset downloads for local smoke tests with:

```bash
assets.hf_vidit.enabled=false assets.vidit.enabled=false assets.sit_pretrained.enabled=false
```

## Sanity Check

```bash
python scripts/sanity_check_cplightsit.py --config-name TrainCPLightSiT_Minimal
```

The sanity script creates a temporary VIDIT-like batch, disables automatic downloads, runs one minimal forward/backward step, checks that `total_loss = flow_loss + lambda_transfer * transfer_loss`, and prints trainable parameter counts.

## Training

The recommended workflow has two stages: one RayEncoder pretraining run followed by one CP-LightSiT fine-tuning run.

To run both stages in order with DDP, use:

```bash
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 ./train.sh
```

`train.sh` runs `train_encoder.sh` first, which launches one `TrainRayEncoder` DDP run using the default simple LR setup. It writes the encoder run result to:

```text
checkpoint/encoder_runs.tsv
```

After the encoder run finishes, `train.sh` stores a stable copy of the RayEncoder checkpoint at:

```text
checkpoint/best_RayEncoder/ray_encoder_best.pth
checkpoint/best_RayEncoder/best_ray_encoder.json
```

Then `train.sh` passes that selected RayEncoder into `train_diffusion.sh`, which launches one `TrainCPLightSiT_Minimal` DDP fine-tuning run using the default simple LR setup. The diffusion run manifest is written to:

```text
checkpoint/diffusion_runs.tsv
```

You can also run each stage separately:

```bash
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 ./train_encoder.sh

RAY_CHECKPOINT=checkpoint/best_RayEncoder/ray_encoder_best.pth \
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 ./train_diffusion.sh
```

Stage-specific overrides can be passed through `RAY_ARGS` and `SIT_ARGS`:

```bash
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
RAY_ARGS="epochs=3 batch_size=128 dataloader.global_batch_size=128" \
SIT_ARGS="epochs=30 batch_size=128 dataloader.global_batch_size=128 dataset.train.max_pairs_per_scene=8 dataset.val.max_pairs_per_scene=8" \
./train.sh
```

Both stage scripts are intentionally simple and run one job each. Use `RAY_ARGS` and `SIT_ARGS` for shared settings such as `epochs`, dataset limits, debug flags, or asset toggles.

For fast DDP startup, `train.py` prepares VIDIT metadata manifests before DDP setup. Existing Hugging Face VIDIT conversions get `train/metadata.json` and `val/metadata.json` generated once, so each rank avoids recursive directory scans.

Stage 1 pretrains the RayEncoder on VIDIT illumination labels:

```bash
python train.py -cn TrainRayEncoder \
  epochs=50 \
  batch_size=128 \
  dataloader.global_batch_size=128
```

RayEncoder pretraining uses a ViT-B/16 light encoder and reference pairs from `VIDITRelightingDataset`. Each batch contains a source image and a same-scene reference/target image, so the encoder is trained with three stable losses: 8-way direction classification, source-to-reference rotation consistency, and a physics loss that compares predicted Lambertian log-shading transfer against the observed source/reference log-luminance transfer. The paired dataset preloads unique transformed VIDIT images into RAM before the first epoch, so the training loop consumes memory-resident tensors instead of repeatedly decoding images from disk.

Multi-GPU RayEncoder pretraining uses the same single `train.py` entrypoint:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=2 train.py -cn TrainRayEncoder
```

In this stage, DDP wraps only the trainable RayEncoder module. The rank 0 log should include:

```text
DDP wrapping trainable modules: light_encoder
```

The RayEncoder run writes a clean RayEncoder-only checkpoint:

```text
checkpoint/001_RayEncoder/checkpoint/ray_encoder_best.pth
checkpoint/latest_RayEncoder.txt
```

Stage 2 finetunes CP-LightSiT-L with frozen pretrained RayEncoder, trainable condition cross-attention, trainable `LightTransferTransformer`, and the minimal source-editing objective. The flow still starts from source VAE tokens, but source tokens are not passed as condition context. By default, additive condition adapters are disabled, and all 24 L-backbone DiT blocks, input embedder, and final layer are fine-tuned with LR `1e-4`:

```bash
python train.py -cn TrainCPLightSiT_Minimal \
  ray_encoder_checkpoint=checkpoint/001_RayEncoder/checkpoint/ray_encoder_best.pth \
  epochs=30 \
  batch_size=128 \
  dataloader.global_batch_size=128 \
  dataset.train.max_pairs_per_scene=8
```

This uses the default L model, minimal fine-tuning objective, and writes checkpoints under the next numbered directory such as `checkpoint/001_CPLightSiT`.

Distributed example:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=2 train.py -cn TrainCPLightSiT_Minimal ray_encoder_checkpoint=checkpoint/001_RayEncoder/checkpoint/ray_encoder_best.pth
```

This single command converts or reuses Hugging Face VIDIT under `data/VIDIT_HF`, downloads or reuses SiT-L/2 under `pretrained/`, and only then sets up DDP training.

For local smoke tests without downloading VIDIT, SiT, or a RayEncoder checkpoint:

```bash
python train.py -cn TrainCPLightSiT_Minimal debug_one_batch=true batch_size=2 assets.hf_vidit.enabled=false assets.vidit.enabled=false assets.sit_pretrained.enabled=false allow_freeze_without_pretrain=true
```

## Minimal Fine-Tuning Objective

The default objective is intentionally small for stable fine-tuning:

```text
L_total = L_flow + lambda_transfer * L_transfer
```

- `L_flow`: Rectified Flow / Flow Matching velocity MSE from source VAE latent tokens to target image tokens. With `flow_start_mode=source`, the model learns relighting as image editing instead of unconditional image creation.
- `L_transfer`: SmoothL1 loss between the predicted dense log-luminance transfer map `delta_l` and the VIDIT target-source log-luminance transfer `q_star`.
- `q_star` is clipped by `q_clip=2.0` by default.
- Shadow, reflectance, physics-correlation, linearity, smoothness, ray-rotation, tokenizer-reconstruction, and endpoint image-space losses have been removed from the trainer.
- Endpoint images are not decoded during training when `decode_loss_every=0`.

Fine-tuning defaults train the condition cross-attention blocks, `LightTransferTransformer`, plus the full L diffusion block stack at LR `1e-4`: `x_embedder`, all 24 DiT blocks, and `final_layer`. `RayEncoder` and the pretrained VAE tokenizer are frozen by default. If `freeze_backbone=true`, a pretrained checkpoint must be loaded unless `allow_freeze_without_pretrain=true` is set explicitly.

Default A100 4-GPU training batch size is `128` for both ViT-B reference-pair RayEncoder pretraining and CP-LightSiT-L finetuning. For diffusion, start with `30` epochs. If samples improve and the loss is still trending down, resume the same run to `60` epochs.

RayEncoder quality can be inspected separately:

```bash
python ray_encoder_inference.py \
  --checkpoint checkpoint/001_RayEncoder/checkpoint/ray_encoder_best.pth \
  --split val \
  --eval-batch-size 256 \
  --num-workers 8 \
  --heatmaps \
  --output ray_encoder_report.pdf
```

The RayEncoder report includes 8-way direction accuracy, ray cosine, angular error, confidence, a confusion matrix, and the best/worst examples by angular error. With `--heatmaps`, the report adds a separate heatmap analysis section that groups the same scene under different ground-truth light directions and shows each prediction with its gradient saliency overlay. Use `--heatmap-target gt` to visualize sensitivity to the ground-truth direction class instead of the predicted class.

Equivalent explicit command:

```bash
python train.py --config-name TrainCPLightSiT_Minimal \
  dataset.train.root=/path/to/VIDIT \
  dataset.val.root=/path/to/VIDIT \
  loss_mode=minimal \
  freeze_backbone=true \
  train_diffusion_backbone=true \
  train_diffusion_last_n_blocks=24 \
  lr=0.0001 \
  adapter_lr=0.0001 \
  light_transfer_lr=0.0001 \
  backbone_lr=0.0001 \
  ray_encoder_checkpoint=/path/to/ray_encoder_best.pth \
  lambda_flow=1.0 \
  lambda_transfer=0.1
```

Checkpoints are written under each numbered run's `checkpoint/` subdirectory. The trainer saves `best.pth` plus the most recent 10 `epoch_*.pth` files. RayEncoder pretraining also saves `ray_encoder_best.pth` plus the most recent 10 `ray_encoder_epoch_*.pth` files. Each run directory contains `log.txt`, which mirrors rank 0 stdout/stderr except tqdm carriage-return progress output. In DDP runs, non-zero ranks do not write separate log files by default. The latest run path is also written to `checkpoint/latest_CPLightSiT.txt`.

## Sampling

```bash
python scripts/sample_cplightsit.py --checkpoint checkpoint/001_CPLightSiT/checkpoint/best.pth --source-image examples/source.png --target-direction E --target-temperature 5500 --output outputs/relit_E.png --save-debug-maps
```

The sampler estimates the source ray from the input image, builds an absolute target light from direction and temperature, computes physics and refined dense transfer maps, starts `TrajectoryFlow` from the source VAE tokens, decodes edited target tokens, and optionally saves debug maps. Passing `--target-rotation-deg` instead rotates the estimated source ray by that angle before building the target condition.

## Inference Report

`inference.py` evaluates a CP-LightSiT checkpoint on VIDIT pairs and writes one PDF containing aggregate metrics plus the 5 best and 5 worst samples by `score = MAE + (1 - SSIM)`.

```bash
python inference.py \
  --checkpoint checkpoint/001_CPLightSiT/checkpoint/best.pth \
  --split val \
  --max-samples 64 \
  --eval-batch-size 4 \
  --num-workers 8 \
  --output outputs/cplightsit_inference_report.pdf
```

If `--checkpoint` is omitted, the script reads `checkpoint/latest_CPLightSiT.txt` and uses that run's `checkpoint/best.pth`.

The report includes MAE, MSE, RMSE, PSNR, SSIM, luminance MAE, log-transfer L1, transfer-model L1, source/target ray cosine accuracy, dense-condition magnitude, and remove/create mask statistics. Each sample page shows source, target, prediction, absolute error, ground-truth log transfer, and model transfer maps.

## Components

- `RayEncoder`: ViT-B/16 light encoder that predicts 8-way light-direction logits, converts them into a normalized 3D source light ray, and reports confidence from the direction distribution.
- `PhysicsLightTransfer`: directional-light shading prior using depth-derived normals, ambient light, log-shading transfer, and a pixel-wise light-effect field.
- `LightTransferTransformer`: patch transformer that refines dense light transfer and outputs an 18-channel dense condition.
- `CPLightSiT`: token-space DiT with global light condition, dense condition, and source-token injection.
- `TrajectoryFlow`: rectified-flow objective with a mask-compatible interface.

During training, the estimated or ground-truth source ray is rotated by the VIDIT source-to-target azimuth delta. The rotated ray is converted back to the target light condition and is used by both `PhysicsLightTransfer` and `CPLightSiT`. The flow path starts from the source VAE tokens, optionally with small noise, and ends at the target VAE tokens.

## Losses

- Flow MSE: source-token-to-target-token velocity prediction loss in VAE latent token space.
- Dense transfer SmoothL1: stabilizes the physical light-transfer condition.

## Known Limitations

- VIDIT mainly supervises external directional light transfer.
- Local or in-image point light is only a physics-guided extension unless additional data is provided.
- Real-world performance depends on source-light estimation, depth quality, camera-response gap, and synthetic-to-real robustness.
