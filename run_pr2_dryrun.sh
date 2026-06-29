#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/EGNN-Binding-Affinity-Prediction

echo "=== [$(date)] STEP 1: rm -rf .venv ==="
rm -rf .venv

echo "=== [$(date)] STEP 2: uv cache clear ==="
uv cache clear

echo "=== [$(date)] STEP 3: uv sync ==="
uv sync

echo "=== [$(date)] STEP 4: torch / CUDA check (expect CUDA 12.6) ==="
uv run python -c "import torch;print('TORCH',torch.__version__,'CUDA',torch.version.cuda,'avail',torch.cuda.is_available(),'devs',torch.cuda.device_count())"

echo "=== [$(date)] STEP 5: 5-sample dry run on A100 GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 uv run generate_tcr_pmhc_dataset.py --limit 5 --num_workers 0 --output dry_run5_tcr_pmhc_dataset.csv

echo "=== [$(date)] ALL DONE (exit 0) ==="
