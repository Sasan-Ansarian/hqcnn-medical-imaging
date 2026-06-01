#!/bin/bash
#SBATCH --job-name=ch11_blin164_d20_s21
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter11/core_bottleneck/ch11_blinear_16_4_e2e_lowdata20_seed21_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter11/core_bottleneck/ch11_blinear_16_4_e2e_lowdata20_seed21_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

python scripts/run_experiment.py --config configs/chapter11/core_bottleneck/ch11_blinear_16_4_e2e_lowdata20_seed21.yaml
