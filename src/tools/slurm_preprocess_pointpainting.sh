#!/bin/bash
#SBATCH --job-name=ro47020_pp_preprocess
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

# Point PyTorch to pre-downloaded model weights (no internet on compute nodes)
export TORCH_HOME="${SLURM_SUBMIT_DIR}/torch_cache"

PREPROCESS_MODEL="${PREPROCESS_MODEL:-mobilenet}"
PREPROCESS_RADAR_MODE="${PREPROCESS_RADAR_MODE:-single}"
PREPROCESS_OUTPUT_DIR="${PREPROCESS_OUTPUT_DIR:-}"

case "${PREPROCESS_MODEL,,}" in
	mobilenet|mobile)
		PREPROCESS_SCRIPT="src/dataset/preprocess_pointpainting_mobileNet.py"
		OUTPUT_DIR="painted_radar"
		;;
	resnet)
		PREPROCESS_SCRIPT="src/dataset/preprocess_pointpainting_resnet.py"
		case "${PREPROCESS_RADAR_MODE}" in
			single) OUTPUT_DIR="painted_radar_resnet" ;;
			3_frames) OUTPUT_DIR="painted_radar_resnet_3frames" ;;
			5_frames) OUTPUT_DIR="painted_radar_resnet_5frames" ;;
			*)
				echo "Unsupported PREPROCESS_RADAR_MODE=${PREPROCESS_RADAR_MODE}."
				exit 2
				;;
		esac
		;;
	*)
		echo "Unsupported PREPROCESS_MODEL=${PREPROCESS_MODEL}. Use mobilenet or resnet."
		exit 2
		;;
esac

if [[ -n "${PREPROCESS_OUTPUT_DIR}" ]]; then
	OUTPUT_DIR="${PREPROCESS_OUTPUT_DIR}"
fi

echo "Running PointPainting preprocessing with ${PREPROCESS_MODEL} (${PREPROCESS_RADAR_MODE}) -> ${OUTPUT_DIR}"

if [[ "${PREPROCESS_MODEL,,}" == "resnet" ]]; then
	python -u "${PREPROCESS_SCRIPT}" \
		--radar-mode "${PREPROCESS_RADAR_MODE}" \
		--save-dir "${OUTPUT_DIR}"
else
	python -u "${PREPROCESS_SCRIPT}"
fi
