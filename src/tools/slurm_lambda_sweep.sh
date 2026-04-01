#!/bin/bash
#SBATCH --job-name=ro47020_lambda_sweep
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err

# Lambda sweep over the Doppler attention weighting parameter.
# Runs three variants of A7 (full model) back-to-back in one job:
#   L05  lambda_doppler=0.5  (weak Doppler bias)
#   L10  lambda_doppler=1.0  (moderate)
#   L40  lambda_doppler=4.0  (strong Doppler bias)
#
# Usage (submit from project root):
#   sbatch src/tools/slurm_lambda_sweep.sh
#
# Optional overrides:
#   sbatch --export=ALL,EPOCHS=12 src/tools/slurm_lambda_sweep.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p outputs/slurm_logs

module load 2024r1 miniconda3/4.12.0 cuda/12.5
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATA_ROOT="${DATA_ROOT:-data/view_of_delft}"
PAINTED_RADAR_DIR="painted_radar_resnet_5frames"
MODEL_CFG="pointPainting_resnet_temporal_5frames_doppler_neck_wide_pfn"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${TMPDIR:-/tmp}/wandb_$$"
mkdir -p "$WANDB_DIR"

echo "=========================================="
echo "Lambda Doppler Sweep — A7 base config"
echo "Epochs: $EPOCHS  |  Model: $MODEL_CFG"
echo "=========================================="

# ── L05: lambda_doppler = 0.5 ─────────────────────────────────────────────────
echo ""
echo "[1/3] Training A7_lambda_0.5 ..."
srun python -u src/tools/train.py \
  model="$MODEL_CFG" \
  exp_id="A7_lambda_0.5" \
  data_root="$DATA_ROOT" \
  epochs="$EPOCHS" \
  batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  model.dataset.use_painted_radar=true \
  model.dataset.painted_radar_dir="$PAINTED_RADAR_DIR" \
  model.backbone.doppler_lambda=0.5

echo "[1/3] Done — A7_lambda_0.5"

# ── L10: lambda_doppler = 1.0 ─────────────────────────────────────────────────
echo ""
echo "[2/3] Training A7_lambda_1.0 ..."
srun python -u src/tools/train.py \
  model="$MODEL_CFG" \
  exp_id="A7_lambda_1.0" \
  data_root="$DATA_ROOT" \
  epochs="$EPOCHS" \
  batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  model.dataset.use_painted_radar=true \
  model.dataset.painted_radar_dir="$PAINTED_RADAR_DIR" \
  model.backbone.doppler_lambda=1.0

echo "[2/3] Done — A7_lambda_1.0"

# ── L40: lambda_doppler = 4.0 ─────────────────────────────────────────────────
echo ""
echo "[3/3] Training A7_lambda_4.0 ..."
srun python -u src/tools/train.py \
  model="$MODEL_CFG" \
  exp_id="A7_lambda_4.0" \
  data_root="$DATA_ROOT" \
  epochs="$EPOCHS" \
  batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  model.dataset.use_painted_radar=true \
  model.dataset.painted_radar_dir="$PAINTED_RADAR_DIR" \
  model.backbone.doppler_lambda=4.0

echo "[3/3] Done — A7_lambda_4.0"

echo ""
echo "=========================================="
echo "Sweep complete. Results in:"
echo "  outputs/A7_lambda_0.5/"
echo "  outputs/A7_lambda_1.0/"
echo "  outputs/A7_lambda_4.0/"
echo "=========================================="
