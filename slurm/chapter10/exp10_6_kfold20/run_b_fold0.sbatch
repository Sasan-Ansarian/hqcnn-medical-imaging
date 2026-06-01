#!/bin/bash
#SBATCH --job-name=ch10_b16_kf0
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter10/exp10_6_kfold20/ch10_b16_kfold20_fold0_%j.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/chapter10/exp10_6_kfold20/ch10_b16_kfold20_fold0_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate
cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"

python -u scripts/run_experiment.py --config configs/chapter10/exp10_6_kfold20/b_fold0.yaml
