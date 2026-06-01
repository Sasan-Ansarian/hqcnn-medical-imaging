#!/bin/bash
#SBATCH --job-name=dermamnist_backbone_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

echo "=== JOB INFO ==="
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Workdir: $(pwd)"
echo "==============="

cd /scratch/lustre/home/saan1491/hqcnn_project
export PYTHONPATH="$PWD"
source /scratch/lustre/home/saan1491/venvs/hqcnn/bin/activate

MODELS=("resnet18_small" "densenet121_small")
SEEDS=(21 42 84)

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 3))   # 0,0,0,1,1,1
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))    # 0,1,2 repeated

MODEL_NAME=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

EXP_NAME="dermamnist_${MODEL_NAME}_v2_seed${SEED}"

CONFIG_PATH="/tmp/${EXP_NAME}.yaml"

cat > "${CONFIG_PATH}" <<EOF
experiment:
  name: ${EXP_NAME}
  seed: ${SEED}
  output_dir: outputs

dataset:
  name: dermamnist
  root: /scratch/lustre/home/saan1491/hqcnn_project/data

model:
  name: ${MODEL_NAME}
  num_classes: 7

training:
  epochs: 30
  batch_size: 128
  num_workers: 4
  device: cuda
  lr: 0.00005
  weight_decay: 0.01
EOF

echo "[INFO] Running: ${EXP_NAME}"
echo "[INFO] Using model: ${MODEL_NAME} | seed: ${SEED}"
echo "[INFO] Config path: ${CONFIG_PATH}"
python3 scripts/run_experiment.py --config "${CONFIG_PATH}"
