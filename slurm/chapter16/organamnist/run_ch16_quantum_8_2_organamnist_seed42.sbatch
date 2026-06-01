#!/bin/bash
#SBATCH --job-name=ch16_q82_org
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter16/organamnist/ch16_quantum_8_2_organamnist_seed42_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter16/organamnist/ch16_quantum_8_2_organamnist_seed42_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

python scripts/run_experiment.py --config configs/chapter16/organamnist/ch16_quantum_8_2_organamnist_seed42.yaml
