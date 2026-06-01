#!/bin/bash
#SBATCH --job-name=dermamnist_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Workdir: $(pwd)"
echo "==============="

cd ~/hqcnn_project
export PYTHONPATH="$PWD"
source ~/venvs/hqcnn/bin/activate

CONFIG_PATH=${1:-configs/dermamnist_resnet18_small_v1.yaml}
echo "[INFO] Using config: ${CONFIG_PATH}"
python3 scripts/run_experiment.py --config ${CONFIG_PATH}
