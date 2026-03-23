#!/bin/bash
#SBATCH --job-name=ro47020_ablation
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

# Keep ablations under the DelftBlue 4h walltime budget by default.
EPOCHS="${EPOCHS:-6}"
if [ "$EPOCHS" -gt 6 ]; then
  echo "Requested EPOCHS=$EPOCHS exceeds recommended 6 for 4h limit. Capping to 6."
  EPOCHS=6
fi
VAL_EVERY="${VAL_EVERY:-$EPOCHS}"

BASE_OFF_TOGGLES="model.voxel_encoder.with_doppler_cluster=false model.backbone.use_doppler_attention=false model.voxel_encoder.with_doppler_magnitude=false model.voxel_encoder.with_doppler_sign=false model.head.velocity_auxiliary.enabled=false model.head.velocity_smoothness.enabled=false model.dataset.augment_train_points=false model.dataset.drop_static_points=false model.dataset.doppler_normalize=false model.neck_refine.enabled=false model.test_time_augmentation.enabled=false"

EXP_ID=""
OVERRIDES=""

case "$ABLATION_ID" in
  A0)
    EXP_ID="A0_baseline"
    OVERRIDES="$BASE_OFF_TOGGLES"
    ;;
  A1)
    EXP_ID="A1_doppler_cluster"
    OVERRIDES="$BASE_OFF_TOGGLES model.voxel_encoder.with_doppler_cluster=true"
    ;;
  A2)
    EXP_ID="A2_doppler_attention"
    OVERRIDES="$BASE_OFF_TOGGLES model.backbone.use_doppler_attention=true"
    ;;
  A3)
    EXP_ID="A3_doppler_magnitude"
    OVERRIDES="$BASE_OFF_TOGGLES model.voxel_encoder.with_doppler_magnitude=true"
    ;;
  A4)
    EXP_ID="A4_doppler_sign"
    OVERRIDES="$BASE_OFF_TOGGLES model.voxel_encoder.with_doppler_sign=true"
    ;;
  A5)
    EXP_ID="A5_velocity_aux"
    OVERRIDES="$BASE_OFF_TOGGLES model.head.velocity_auxiliary.enabled=true"
    ;;
  A6)
    EXP_ID="A6_velocity_smooth"
    OVERRIDES="$BASE_OFF_TOGGLES model.head.velocity_smoothness.enabled=true"
    ;;
  G1)
    EXP_ID="G1_point_aug"
    OVERRIDES="$BASE_OFF_TOGGLES model.dataset.augment_train_points=true model.dataset.point_dropout_prob=0.1 model.dataset.xy_noise_std=0.05 model.dataset.rcs_noise_std=0.5 model.dataset.doppler_noise_std=0.2"
    ;;
  G2)
    EXP_ID="G2_static_suppress"
    OVERRIDES="$BASE_OFF_TOGGLES model.dataset.drop_static_points=true model.dataset.static_velocity_threshold=0.2 model.dataset.static_keep_prob=1.0"
    ;;
  G3)
    EXP_ID="G3_doppler_norm"
    OVERRIDES="$BASE_OFF_TOGGLES model.dataset.doppler_normalize=true"
    ;;
  G4)
    EXP_ID="G4_neck_refine"
    OVERRIDES="$BASE_OFF_TOGGLES model.neck_refine.enabled=true"
    ;;
  C1)
    EXP_ID="C1_A1_A2"
    OVERRIDES="$BASE_OFF_TOGGLES model.voxel_encoder.with_doppler_cluster=true model.backbone.use_doppler_attention=true"
    ;;
  C2)
    EXP_ID="C2_A1_A2_A5"
    OVERRIDES="$BASE_OFF_TOGGLES model.voxel_encoder.with_doppler_cluster=true model.backbone.use_doppler_attention=true model.head.velocity_auxiliary.enabled=true"
    ;;
  C3)
    EXP_ID="C3_A1_A2_G1"
    OVERRIDES="$BASE_OFF_TOGGLES model.voxel_encoder.with_doppler_cluster=true model.backbone.use_doppler_attention=true model.dataset.augment_train_points=true model.dataset.point_dropout_prob=0.1 model.dataset.xy_noise_std=0.05 model.dataset.rcs_noise_std=0.5 model.dataset.doppler_noise_std=0.2"
    ;;
  C4)
    EXP_ID="C4_full"
    OVERRIDES="$BASE_OFF_TOGGLES model.voxel_encoder.with_doppler_cluster=true model.backbone.use_doppler_attention=true model.head.velocity_auxiliary.enabled=true model.neck_refine.enabled=true"
    ;;
  *)
    echo "Unknown ABLATION_ID: $ABLATION_ID"
    echo "Supported IDs: A0 A1 A2 A3 A4 A5 A6 G1 G2 G3 G4 C1 C2 C3 C4"
    exit 2
    ;;
esac

if [ "$MODE" = "train" ]; then
  echo "Running train ablation $ABLATION_ID as $EXP_ID"
  srun python -u src/tools/train.py \
    exp_id="$EXP_ID" \
    data_root="$DATA_ROOT" \
    epochs="$EPOCHS" \
    val_every="$VAL_EVERY" \
    save_top_model=1 \
    batch_size="$BATCH_SIZE" \
    num_workers="$NUM_WORKERS" \
    $OVERRIDES
elif [ "$MODE" = "eval" ]; then
  CKPT_PATH="${CKPT_PATH:-outputs/${EXP_ID}/checkpoints/last.ckpt}"
  echo "Running eval for $EXP_ID with checkpoint $CKPT_PATH"
  srun python -u src/tools/eval.py \
    model=centerpoint_radar \
    checkpoint_path="$CKPT_PATH" \
    data_root="$DATA_ROOT" \
    num_workers="$NUM_WORKERS" \
    $OVERRIDES
else
  echo "Unsupported MODE: $MODE (use train or eval)"
  exit 2
fi
