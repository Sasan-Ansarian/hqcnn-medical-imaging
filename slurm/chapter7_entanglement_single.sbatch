#!/bin/bash
#SBATCH --job-name=c7_entangle
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/c7_entangle_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/c7_entangle_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT=/scratch/lustre/home/saan1491/hqcnn_project
VENV_PATH=/scratch/lustre/home/saan1491/venvs/hqcnn
CONFIG_PATH="configs/chapter7/entanglement/quantum_C_32_8_ring_weighted_seed42.yaml"

source "${VENV_PATH}/bin/activate"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] HOSTNAME=$(hostname)"

python scripts/run_experiment.py --config "${CONFIG_PATH}"
