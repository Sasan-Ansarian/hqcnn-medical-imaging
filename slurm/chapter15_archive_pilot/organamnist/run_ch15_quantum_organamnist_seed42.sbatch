#!/bin/bash
#SBATCH --job-name=ch15_q_org42
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter15/organamnist/ch15_quantum_organamnist_seed42_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter15/organamnist/ch15_quantum_organamnist_seed42_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

python scripts/run_experiment.py --config configs/chapter15/organamnist/ch15_quantum_organamnist_seed42.yaml
