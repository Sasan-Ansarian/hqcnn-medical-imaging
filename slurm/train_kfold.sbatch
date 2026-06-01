#!/bin/bash
#SBATCH --job-name=hqcnn_kfold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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
source ~/venvs/hqcnn/bin/activate
# module purge
# module load python
# module load cuda
# source ~/venvs/hqcnn/bin/activate

python3 scripts/run_experiment.py --config configs/example.yaml
