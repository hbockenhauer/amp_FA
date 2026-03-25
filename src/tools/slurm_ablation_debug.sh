#!/bin/bash
#SBATCH --job-name=ro47020_debug
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --output=outputs/slurm_logs/debug_%j.out
#SBATCH --error=outputs/slurm_logs/debug_%j.err

echo "=== START DEBUG ===" >&2
pwd >&2
echo "ABLATION_ID=$ABLATION_ID" >&2

set -euo pipefail

echo "After set -euo pipefail" >&2

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "After cd" >&2

mkdir -p outputs/slurm_logs
echo "After mkdir" >&2

echo "About to load modules" >&2
module load 2024r1 miniconda3/4.12.0 cuda/12.5
echo "After module load" >&2

unset CONDA_SHLVL
echo "After unset CONDA_SHLVL" >&2

source "$(conda info --base)/etc/profile.d/conda.sh"
echo "After source conda" >&2

conda activate amp
echo "After conda activate" >&2

echo "=== END DEBUG ===" >&2
