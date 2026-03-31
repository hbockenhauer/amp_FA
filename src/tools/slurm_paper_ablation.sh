#!/bin/bash
#SBATCH --job-name=ro47020_paper_ablation
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

# Paper ablation study: 3 cumulative improvements over the baseline
#
#  A0 - Baseline        : CenterPoint radar, single frame
#  A1 - +PointPainting  : ResNet PointPainting (no Doppler features)
#  A2 - +Doppler        : A1 + Doppler cluster & backbone attention
#  A3 - +Temporal       : A2 + 5-frame radar accumulation  (= full model)
#
# Usage:
#   sbatch --export=ALL,ABLATION_ID=A1 src/tools/slurm_paper_ablation.sh
#   sbatch --export=ALL,ABLATION_ID=A2,EPOCHS=20 src/tools/slurm_paper_ablation.sh
#   sbatch --export=ALL,ABLATION_ID=A3,MODE=eval,CKPT_PATH=outputs/A3_temporal_5frames/checkpoints/last.ckpt src/tools/slurm_paper_ablation.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p outputs/slurm_logs

module load 2024r1 miniconda3/4.12.0 cuda/12.5
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

ABLATION_ID="${ABLATION_ID:-${1:-A0}}"
MODE="${MODE:-train}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATA_ROOT="${DATA_ROOT:-data/view_of_delft}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_TAG="${RUN_TAG:-}"

# Training schedule — keep within the 4h DelftBlue wall-time limit.
EPOCHS="${EPOCHS:-20}"
VAL_EVERY="${VAL_EVERY:-2}"

export WANDB_MODE

if [ "$EPOCHS" -lt 1 ]; then
  echo "Invalid EPOCHS=$EPOCHS (must be >= 1)"
  exit 2
fi

if [ ! -f "$DATA_ROOT/lidar/ImageSets/train.txt" ]; then
  echo "Missing dataset split file: $DATA_ROOT/lidar/ImageSets/train.txt"
  echo "Set DATA_ROOT to your View-of-Delft root directory."
  exit 2
fi

# ── Ablation definitions ──────────────────────────────────────────────────────
MODEL_CFG=""
EXP_ID=""
USE_PAINTED_RADAR="false"
PAINTED_RADAR_DIR="painted_radar"

case "$ABLATION_ID" in
  A0)
    # Pure radar CenterPoint — no camera, no Doppler features, single frame.
    EXP_ID="A0_baseline"
    MODEL_CFG="centerpoint_baseline"
    USE_PAINTED_RADAR="false"
    PAINTED_RADAR_DIR="painted_radar"
    ;;
  A1)
    # Add ResNet PointPainting: camera segmentation scores painted onto radar.
    # Doppler cluster / attention remain OFF so the PP contribution is isolated.
    EXP_ID="A1_pointpainting_resnet"
    MODEL_CFG="pointPainting_resnet"
    USE_PAINTED_RADAR="true"
    PAINTED_RADAR_DIR="painted_radar_resnet"
    ;;
  A2)
    # Add Doppler: pillar-level Doppler clustering + backbone Doppler attention.
    EXP_ID="A2_doppler"
    MODEL_CFG="pointPainting_resnet_doppler"
    USE_PAINTED_RADAR="true"
    PAINTED_RADAR_DIR="painted_radar_resnet"
    ;;
  A3)
    # Add Temporal: 5-frame radar accumulation (all three improvements active).
    EXP_ID="A3_temporal_5frames"
    MODEL_CFG="pointPainting_resnet_temporal_5frames_doppler"
    USE_PAINTED_RADAR="true"
    PAINTED_RADAR_DIR="painted_radar_resnet_5frames"
    ;;
  *)
    echo "Unknown ABLATION_ID: $ABLATION_ID"
    echo "Supported IDs: A0 A1 A2 A3"
    exit 2
    ;;
esac

if [ -n "$RUN_TAG" ]; then
  EXP_ID="${EXP_ID}_${RUN_TAG}"
fi

# Validate painted radar directory for camera-fusion models.
if [ "$USE_PAINTED_RADAR" = "true" ] && [ ! -d "$PAINTED_RADAR_DIR" ]; then
  echo "Missing painted radar directory: $PAINTED_RADAR_DIR"
  echo "Generate it first with src/dataset/preprocess_pointpainting_resnet.py"
  exit 2
fi

# ── Run ──────────────────────────────────────────────────────────────────────
if [ "$MODE" = "train" ]; then
  echo "Training $ABLATION_ID → $EXP_ID  (model=$MODEL_CFG, painted=$USE_PAINTED_RADAR, epochs=$EPOCHS, val_every=$VAL_EVERY)"
  srun python -u src/tools/train.py \
    model="$MODEL_CFG" \
    exp_id="$EXP_ID" \
    data_root="$DATA_ROOT" \
    epochs="$EPOCHS" \
    val_every="$VAL_EVERY" \
    save_top_model=1 \
    batch_size="$BATCH_SIZE" \
    num_workers="$NUM_WORKERS" \
    model.dataset.use_painted_radar="$USE_PAINTED_RADAR" \
    model.dataset.painted_radar_dir="$PAINTED_RADAR_DIR"

elif [ "$MODE" = "eval" ]; then
  CKPT_PATH="${CKPT_PATH:-outputs/${EXP_ID}/checkpoints/last.ckpt}"
  echo "Evaluating $ABLATION_ID → $EXP_ID  (checkpoint=$CKPT_PATH)"
  srun python -u src/tools/eval.py \
    model="$MODEL_CFG" \
    checkpoint_path="$CKPT_PATH" \
    data_root="$DATA_ROOT" \
    num_workers="$NUM_WORKERS" \
    model.dataset.use_painted_radar="$USE_PAINTED_RADAR" \
    model.dataset.painted_radar_dir="$PAINTED_RADAR_DIR"

elif [ "$MODE" = "test" ]; then
  CKPT_PATH="${CKPT_PATH:-outputs/${EXP_ID}/checkpoints/last.ckpt}"
  echo "Testing $ABLATION_ID → $EXP_ID  (checkpoint=$CKPT_PATH)"
  srun python -u src/tools/test.py \
    model="$MODEL_CFG" \
    checkpoint_path="$CKPT_PATH" \
    data_root="$DATA_ROOT" \
    num_workers="$NUM_WORKERS" \
    model.dataset.use_painted_radar="$USE_PAINTED_RADAR" \
    model.dataset.painted_radar_dir="$PAINTED_RADAR_DIR"

else
  echo "Unsupported MODE: $MODE  (use train | eval | test)"
  exit 2
fi
