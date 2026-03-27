#!/bin/bash
#SBATCH --job-name="centperpoint_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --output=outputs/pointPainting_V2_%j.out
#SBATCH --error=outputs/pointPainting_V2_%j.err

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

#srun python -u src/tools/train.py exp_id=pointPainting_resnet_V2 model=pointPainting_resnet batch_size=4 num_workers=2 epochs=25
srun python -u src/tools/train.py "$@"

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
