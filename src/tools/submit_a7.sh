#!/bin/bash
#SBATCH --job-name=ro47020_final_model
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

# Final model: PointPainting preprocessing (ResNet, 5-frame) + training in one job.
#
# Usage:
#   sbatch src/tools/submit_a7.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p outputs/slurm_logs

module load 2024r1 miniconda3/4.12.0 cuda/12.5
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

export TORCH_HOME="${SLURM_SUBMIT_DIR}/torch_cache"
export WANDB_DIR="${TMPDIR:-/tmp}/wandb_$$"
mkdir -p "$WANDB_DIR"

PAINTED_RADAR_DIR="painted_radar_resnet_5frames"

# ── Step 1: Preprocessing ─────────────────────────────────────────────────────
if [ ! -d "$PAINTED_RADAR_DIR" ]; then
  echo "==> Step 1/2: PointPainting preprocessing (resnet, 5_frames)"
  python -u src/dataset/preprocess_pointpainting_resnet.py \
    --radar-mode 5_frames \
    --save-dir "$PAINTED_RADAR_DIR"
else
  echo "==> Step 1/2: $PAINTED_RADAR_DIR already exists, skipping preprocessing"
fi

# ── Step 2: Training ──────────────────────────────────────────────────────────
echo "==> Step 2/2: Training final model (24 epochs)"
srun python -u src/tools/train.py \
  model="pointPainting_resnet_temporal_5frames_doppler_neck_wide_pfn" \
  exp_id="A7_wide_pfn" \
  data_root="data/view_of_delft" \
  epochs=24 \
  val_every=3 \
  save_top_model=1 \
  batch_size=4 \
  num_workers=2 \
  model.dataset.use_painted_radar="true" \
  model.dataset.painted_radar_dir="$PAINTED_RADAR_DIR"
