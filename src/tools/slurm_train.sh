#!/bin/bash
#SBATCH --job-name="pointPainting_ro47020"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-task=1
#SBATCH --account=education-me-courses-ro47020
#SBATCH --mail-type=END
#SBATCH --output=outputs/slurm_pointpainting_ro47020_%j.out
#SBATCH --error=outputs/slurm_pointpainting_ro47020_%j.err

module load 2024r1 miniconda3/4.12.0 cuda/12.5

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amp

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi


#srun python -u src/tools/train.py exp_id=resnet_voxel_04 model.voxel_size=[0.4,0.4,5] batch_size=4 epochs=25w

srun python -u src/tools/train.py "$@"

#srun python -u src/tools/train.py exp_id=pointpainting_radar_resnet_e25 batch_size=4 num_workers=2 epochs=25
#srun python -u src/tools/train.py exp_id=resnet_voxel_032 model.voxel_size=[0.32,0.32,5] batch_size=4 epochs=25
#srun python -u src/tools/train.py exp_id=resnet_voxel_04 model.voxel_size=[0.4,0.4,5] batch_size=4 epochs=25
#srun python -u src/tools/train.py exp_id=resnet_voxel_048 model.voxel_size=[0.48,0.48,5] batch_size=4 epochs=25

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
