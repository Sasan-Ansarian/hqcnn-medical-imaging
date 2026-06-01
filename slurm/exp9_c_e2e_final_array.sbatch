#!/bin/bash
#SBATCH --job-name=exp9_c_e2e_final
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter8/exp9_c_e2e_final_%A_%a.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter8/exp9_c_e2e_final_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-2

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

CONFIGS=(
  "configs/chapter8/exp9_c_e2e_final_seed21.yaml"
  "configs/chapter8/exp9_c_e2e_final_seed42.yaml"
  "configs/chapter8/exp9_c_e2e_final_seed84.yaml"
)

python scripts/run_experiment.py --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
