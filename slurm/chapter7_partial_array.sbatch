#!/bin/bash
#SBATCH --job-name=c7_partial
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/c7_partial_%A_%a.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/c7_partial_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1

set -euo pipefail

PROJECT_ROOT=/scratch/lustre/home/saan1491/hqcnn_project
VENV_PATH=/scratch/lustre/home/saan1491/venvs/hqcnn

source "${VENV_PATH}/bin/activate"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

CONFIGS=(
  "configs/chapter7/partial/partial_B_32_8_weighted_seed42.yaml"
  "configs/chapter7/partial/partial_C_32_8_weighted_seed42.yaml"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_PATH=${CONFIGS[$TASK_ID]}

echo "[INFO] TASK_ID=${TASK_ID}"
echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] HOSTNAME=$(hostname)"

python scripts/run_experiment.py --config "${CONFIG_PATH}"
