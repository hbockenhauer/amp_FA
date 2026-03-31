#!/bin/bash
set -euo pipefail

# Submit all four paper ablation jobs to DelftBlue.
#
# Usage:
#   bash src/tools/submit_paper_ablations.sh
#   EPOCHS=20 BATCH_SIZE=4 bash src/tools/submit_paper_ablations.sh
#   RUN_TAG=run2 bash src/tools/submit_paper_ablations.sh

IDS=(A0 A1 A2 A3)

for id in "${IDS[@]}"; do
  echo "Submitting $id"
  sbatch --export=ALL,ABLATION_ID="$id" src/tools/slurm_paper_ablation.sh
  sleep 1
done

echo "Submitted ${#IDS[@]} jobs: ${IDS[*]}"
echo "Monitor with: squeue -u \$USER"
