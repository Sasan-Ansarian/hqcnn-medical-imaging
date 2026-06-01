#!/bin/bash
#SBATCH --job-name=ch10_b164_lr1e3
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter10/exp10_4_headlr_sweep/b_16_4_e2e_lowdata20_headlr1e3_seed42_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter10/exp10_4_headlr_sweep/b_16_4_e2e_lowdata20_headlr1e3_seed42_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

python -u scripts/run_experiment.py --config configs/chapter10/exp10_4_headlr_sweep/b_16_4_e2e_lowdata20_headlr1e3_seed42.yaml
