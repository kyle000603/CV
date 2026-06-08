# CP-LightDiT

CP-LightDiT is a minimal standalone training codebase for computational-photography-guided light transfer conditioning in a Diffusion Transformer relighting pipeline. It takes a VIDIT source image under source light, estimates the source light ray, rotates that ray toward the target light direction, builds a physics-guided source-to-target light transfer field, refines that dense field with a small transformer, and conditions a token-space DiT through rectified flow.

## Folder Structure

- `train.py`: Hydra entrypoint that prepares assets before DDP setup, then starts training.
- `configs/`: dataset, dataloader, model, loss, optimizer, diffusion, trainer, and wandb config groups.
- `dataset/data/`: VIDIT relighting pair dataset.
- `dataset/dataloader/`: seeded dataloader and distributed sampler setup.
- `models/modules/`: light utilities, ray encoder, physics transfer, dense transfer transformer, and simple tokenizer.
- `models/diffusion/`: DiT and CP-LightDiT conditioning backbone.
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
- Converted dataset markers are written under `data/VIDIT_HF/.cplightdit_assets`, so later runs reuse the processed files immediately.
- EPFL/NTIRE zip downloads are disabled by default because `assets.hf_vidit.enabled=true` is the preferred path. You can re-enable `assets.vidit.enabled=true` if you want to use local Track1 zips.
- Pretrained SiT-XL/2 weights are stored under `pretrained/SiT-XL-2-256x256.pt`. Existing files are reused. The default source is the Hugging Face `nyu-visionx/SiT-collections` mirror, with the original Dropbox URL kept as fallback.
- Training checkpoints are saved under numbered run directories like `checkpoint/001_CPLightDiT`, `checkpoint/002_CPLightDiT`, and so on.

The default pretrained backbone is SiT-XL/2 because the default model config is XL for an A100 40GB setup. The loader imports only tensors whose names and shapes match CP-LightDiT; incompatible image-latent patch weights are skipped safely.

## Default Config

The default Hydra config is `configs/TrainCPLightDiT.yaml`.

- Model config: `configs/model/CP_LightDiT_XL.yaml`
- Model size: CP-LightDiT XL, `depth=28`, `hidden_size=1152`, `num_heads=16`
- Image size: `256`
- Token grid: `16 x 16`
- Feature dim: `392`
- Batch size: `16`
- Epochs: `100`
- Dataset root: `data/VIDIT_HF`
- Hugging Face dataset: `Nahrawy/VIDIT-Depth-ControlNet`
- Hugging Face cache: `data/hf_cache`
- Pretrained SiT checkpoint: `pretrained/SiT-XL-2-256x256.pt` from `nyu-visionx/SiT-collections/SiT-XL-2-256.pt`
- Training checkpoints: numbered directories under `checkpoint/`, for example `checkpoint/001_CPLightDiT`
- Training log file: `log.txt` inside each numbered checkpoint directory
- Latest run pointer: `checkpoint/latest_CPLightDiT.txt`
- W&B mode: `online`, automatically disabled if `WANDB_API_KEY` is not set

Install the Hugging Face dataset dependencies in the CV environment before the first training run:

```bash
/home/jovyan/irrlab/anaconda3/envs/CV/bin/python -m pip install -U datasets huggingface_hub pyarrow
```

`train.py` prepares the dataset and pretrained checkpoint before DDP process-group setup. With `torchrun`, rank 0 performs the Hugging Face conversion and SiT checkpoint download first, while the other ranks wait for a local preflight marker. Once preparation is complete, normal DDP initialization and training begin.

You can also prepare the dataset and pretrained checkpoint manually in a single process:

```bash
HF_HUB_DISABLE_XET=1 /home/jovyan/irrlab/anaconda3/envs/CV/bin/python scripts/prepare_cplightdit_assets.py
```

Disable automatic asset downloads for local smoke tests with:

```bash
assets.hf_vidit.enabled=false assets.vidit.enabled=false assets.sit_pretrained.enabled=false
```

## Sanity Check

```bash
python scripts/sanity_check_cplightdit.py --config-name TrainCPLightDiT_Minimal
```

The sanity script creates a temporary VIDIT-like batch, disables automatic downloads, runs one minimal forward/backward step, checks that `total_loss = flow_loss + lambda_transfer * transfer_loss`, and prints trainable parameter counts.

## Training

The recommended workflow has two stages.

Stage 1 pretrains the RayEncoder on VIDIT illumination labels:

```bash
python train.py -cn TrainRayEncoder
```

The RayEncoder run writes a clean RayEncoder-only checkpoint:

```text
checkpoint/001_RayEncoder/ray_encoder_latest.pth
```

Stage 2 finetunes CP-LightDiT with the frozen pretrained SiT backbone, frozen pretrained RayEncoder, trainable condition adapters, trainable `LightTransferTransformer`, and the minimal objective:

```bash
python train.py -cn TrainCPLightDiT_Minimal \
  ray_encoder_checkpoint=checkpoint/001_RayEncoder/ray_encoder_latest.pth
```

This uses the default XL model, minimal fine-tuning objective, and writes checkpoints under the next numbered directory such as `checkpoint/001_CPLightDiT`.

Distributed example:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node=2 train.py -cn TrainCPLightDiT_Minimal ray_encoder_checkpoint=checkpoint/001_RayEncoder/ray_encoder_latest.pth
```

This single command converts or reuses Hugging Face VIDIT under `data/VIDIT_HF`, downloads or reuses SiT-XL/2 under `pretrained/`, and only then sets up DDP training.

For local smoke tests without downloading VIDIT, SiT, or a RayEncoder checkpoint:

```bash
python train.py -cn TrainCPLightDiT_Minimal debug_one_batch=true batch_size=2 assets.hf_vidit.enabled=false assets.vidit.enabled=false assets.sit_pretrained.enabled=false allow_freeze_without_pretrain=true
```

## Minimal Fine-Tuning Objective

The default objective is intentionally small for stable fine-tuning:

```text
L_total = L_flow + lambda_transfer * L_transfer
```

- `L_flow`: Rectified Flow / Flow Matching velocity MSE on target image tokens.
- `L_transfer`: SmoothL1 loss between the predicted dense log-luminance transfer map `delta_l` and the VIDIT target-source log-luminance transfer `q_star`.
- `q_star` is clipped by `q_clip=2.0` by default.
- Shadow, reflectance, physics-correlation, linearity, smoothness, ray-rotation, tokenizer-reconstruction, and endpoint image-space losses have been removed from the trainer.
- Endpoint images are not decoded during training when `decode_loss_every=0`.

Fine-tuning defaults freeze the pretrained DiT backbone and train only the condition adapters plus `LightTransferTransformer`. `RayEncoder` and `SimpleImageTokenizer` are frozen by default. If `freeze_backbone=true`, a pretrained checkpoint must be loaded unless `allow_freeze_without_pretrain=true` is set explicitly.

Equivalent explicit command:

```bash
python train.py --config-name TrainCPLightDiT_Minimal \
  dataset.train.root=/path/to/VIDIT \
  dataset.val.root=/path/to/VIDIT \
  loss_mode=minimal \
  freeze_backbone=true \
  ray_encoder_checkpoint=/path/to/ray_encoder_latest.pth \
  lambda_flow=1.0 \
  lambda_transfer=0.1
```

Checkpoints are written under numbered directories in `checkpoint/` by default and include the CP-LightDiT backbone, RayEncoder, LightTransferTransformer, SimpleImageTokenizer, optimizer state, epoch, and config. Each run directory also contains `log.txt`, which mirrors rank 0 stdout/stderr, including `print`, Python logger output, and tqdm progress. In DDP runs, non-zero ranks do not write separate log files by default. The latest run path is also written to `checkpoint/latest_CPLightDiT.txt`.

## Sampling

```bash
python scripts/sample_cplightdit.py --checkpoint checkpoint/001_CPLightDiT/latest.pth --source-image examples/source.png --target-direction E --target-temperature 5500 --output outputs/relit_E.png --save-debug-maps
```

The sampler estimates the source ray from the input image, builds an absolute target light from direction and temperature, computes physics and refined dense transfer maps, samples target tokens with `TrajectoryFlow`, decodes tokens, and optionally saves debug maps. Passing `--target-rotation-deg` instead rotates the estimated source ray by that angle before building the target condition.

## Components

- `RayEncoder`: small ConvNet that predicts a normalized 3D source light ray, projected azimuth direction, temperature, and confidence.
- `PhysicsLightTransfer`: directional-light shading prior using depth-derived normals, ambient light, log-shading transfer, and a pixel-wise light-effect field.
- `LightTransferTransformer`: patch transformer that refines dense light transfer and outputs an 18-channel dense condition.
- `CPLightDiT`: token-space DiT with global light condition, dense condition, and source-token injection.
- `TrajectoryFlow`: rectified-flow objective with a mask-compatible interface.

During training, the estimated or ground-truth source ray is rotated by the VIDIT source-to-target azimuth delta. The rotated ray is converted back to the target light condition and is used by both `PhysicsLightTransfer` and `CPLightDiT`.

## Losses

- Flow MSE: velocity prediction loss in token space.
- Dense transfer SmoothL1: stabilizes the physical light-transfer condition.

## Known Limitations

- VIDIT mainly supervises external directional light transfer.
- Local or in-image point light is only a physics-guided extension unless additional data is provided.
- Real-world performance depends on source-light estimation, depth quality, camera-response gap, and synthetic-to-real robustness.
