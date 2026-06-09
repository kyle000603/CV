#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_ROOT="${RESULT_ROOT:-checkpoint}"
ENCODER_RUN_MANIFEST="${ENCODER_RUN_MANIFEST:-${ENCODER_SWEEP_MANIFEST:-${RESULT_ROOT}/encoder_runs.tsv}}"
DIFFUSION_RUN_MANIFEST="${DIFFUSION_RUN_MANIFEST:-${RESULT_ROOT}/diffusion_runs.tsv}"
BEST_RAY_DIR="${BEST_RAY_DIR:-${RESULT_ROOT}/best_RayEncoder}"

echo "[1/3] Training RayEncoder"
ENCODER_RUN_MANIFEST="${ENCODER_RUN_MANIFEST}" ./train_encoder.sh

echo "[2/3] Preparing selected RayEncoder checkpoint"
read -r BEST_RAY_CHECKPOINT BEST_RAY_SCORE BEST_RAY_RUN_DIR < <(
  "${PYTHON_BIN}" - "${ENCODER_RUN_MANIFEST}" "${BEST_RAY_DIR}" <<'PY'
import csv
import json
import os
import shutil
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
best_dir = Path(sys.argv[2])
if not manifest.exists():
    raise SystemExit(f"encoder run manifest does not exist: {manifest}")

rows = []
with manifest.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle, delimiter="\t")
    for row in reader:
        if len(row) < 5:
            continue
        run_id, score, run_dir, checkpoint, overrides = row[:5]
        path = Path(checkpoint)
        if path.exists():
            rows.append(
                {
                    "run_id": run_id,
                    "score": float(score),
                    "run_dir": run_dir,
                    "checkpoint": str(path),
                    "overrides": overrides,
                }
            )
if not rows:
    raise SystemExit(f"no valid RayEncoder checkpoints found in {manifest}")

best = min(rows, key=lambda item: item["score"])
best_dir.mkdir(parents=True, exist_ok=True)
target = best_dir / "ray_encoder_best.pth"
target.unlink(missing_ok=True)
source = Path(best["checkpoint"])
try:
    os.link(source, target)
except OSError:
    shutil.copy2(source, target)

metadata = {
    "selected": best,
    "all_runs": rows,
    "checkpoint": str(target),
}
(best_dir / "best_ray_encoder.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(str(target), best["score"], best["run_dir"])
PY
)

echo "Selected RayEncoder:"
echo "  checkpoint: ${BEST_RAY_CHECKPOINT}"
echo "  score: ${BEST_RAY_SCORE}"
echo "  source run: ${BEST_RAY_RUN_DIR}"
echo "  metadata: ${BEST_RAY_DIR}/best_ray_encoder.json"

echo "[3/3] Running CP-LightSiT diffusion fine-tuning"
RAY_CHECKPOINT="${BEST_RAY_CHECKPOINT}" DIFFUSION_RUN_MANIFEST="${DIFFUSION_RUN_MANIFEST}" ./train_diffusion.sh

echo "Encoder run manifest: ${ENCODER_RUN_MANIFEST}"
echo "Diffusion run manifest: ${DIFFUSION_RUN_MANIFEST}"
