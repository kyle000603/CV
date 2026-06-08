from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.assets import ensure_hf_vidit_assets, ensure_pretrained_checkpoint, ensure_vidit_assets


os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="../configs", config_name="TrainCPLightDiT")
def main(cfg: DictConfig) -> None:
    ensure_hf_vidit_assets(cfg)
    ensure_vidit_assets(cfg)
    checkpoint_path = ensure_pretrained_checkpoint(cfg)
    if checkpoint_path is not None:
        print(f"Prepared pretrained checkpoint: {checkpoint_path}")
    print("CP-LightDiT asset preparation complete.")


if __name__ == "__main__":
    main()
