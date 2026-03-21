#!/bin/bash
set -euo pipefail

# Submit a default suite of ablation jobs to DelftBlue.
# Usage:
#   bash src/tools/submit_ablations_delftblue.sh
#   EPOCHS=12 BATCH_SIZE=4 NUM_WORKERS=2 bash src/tools/submit_ablations_delftblue.sh

IDS=(A0 A1 A2 A3 A4 A5 A6 G1 G2 G3 G4 C1 C2 C3 C4)

for id in "${IDS[@]}"; do
  echo "Submitting $id"
  sbatch --export=ALL,ABLATION_ID="$id" src/tools/slurm_ablation.sh
  sleep 1
done

echo "Submitted ${#IDS[@]} ablation jobs."
