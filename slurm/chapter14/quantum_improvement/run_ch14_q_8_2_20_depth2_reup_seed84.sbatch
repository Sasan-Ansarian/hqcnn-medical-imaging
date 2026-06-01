#!/bin/bash
#SBATCH --job-name=ch14_q82_d2r_s84
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter14/quantum_improvement/ch14_q_8_2_20_depth2_reup_seed84_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter14/quantum_improvement/ch14_q_8_2_20_depth2_reup_seed84_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

python scripts/run_experiment.py --config configs/chapter14/quantum_improvement/ch14_q_8_2_20_depth2_reup_seed84.yaml
