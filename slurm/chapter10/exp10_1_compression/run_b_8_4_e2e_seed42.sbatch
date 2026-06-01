#!/bin/bash
#SBATCH --job-name=ch10_b84_s42
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter10/exp10_1_compression/b_8_4_e2e_seed42_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter10/exp10_1_compression/b_8_4_e2e_seed42_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

python scripts/run_experiment.py --config configs/chapter10/exp10_1_compression/b_8_4_e2e.yaml
