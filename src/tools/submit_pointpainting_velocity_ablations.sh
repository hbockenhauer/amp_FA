#!/bin/bash
set -euo pipefail

# Submit PointPainting + velocity ablation jobs to DelftBlue.
# Usage examples:
#   bash src/tools/submit_pointpainting_velocity_ablations.sh
#   EPOCHS=10 BATCH_SIZE=4 RUN_TAG=cv5_e10_v3_r1 bash src/tools/submit_pointpainting_velocity_ablations.sh

IDS=(A0 A1 A2 G4 C3)
MODEL_CFG="${MODEL_CFG:-pointPainting_mobileNet}"
USE_PAINTED_RADAR="${USE_PAINTED_RADAR:-true}"
PAINTED_RADAR_DIR="${PAINTED_RADAR_DIR:-painted_radar}"

for id in "${IDS[@]}"; do
  echo "Submitting ${id} (model=${MODEL_CFG}, painted_radar=${USE_PAINTED_RADAR}, painted_dir=${PAINTED_RADAR_DIR})"
  sbatch --export=ALL,ABLATION_ID="$id",MODEL_CFG="$MODEL_CFG",USE_PAINTED_RADAR="$USE_PAINTED_RADAR",PAINTED_RADAR_DIR="$PAINTED_RADAR_DIR" src/tools/slurm_ablation.sh
  sleep 1
done

echo "Submitted ${#IDS[@]} jobs: ${IDS[*]}"
