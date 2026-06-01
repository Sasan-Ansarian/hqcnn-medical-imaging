#!/bin/bash
#SBATCH --job-name=final_abc
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/final_abc_%A_%a.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/final_abc_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-8

set -euo pipefail

PROJECT_ROOT=/scratch/lustre/home/saan1491/hqcnn_project
VENV_PATH=/scratch/lustre/home/saan1491/venvs/hqcnn
OUTPUT_ROOT=${PROJECT_ROOT}/outputs
DATA_ROOT=${PROJECT_ROOT}/data
CONFIG_TMP_DIR=${PROJECT_ROOT}/configs/generated_final_abc
PRETRAINED_BACKBONE=${PROJECT_ROOT}/outputs/dermamnist_resnet18_small_v2_seed42/best_model.pt

mkdir -p "${CONFIG_TMP_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

source "${VENV_PATH}/bin/activate"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

MODELS=(
  "resnet18_small_baseline_frozen"
  "resnet18_small_baseline_frozen"
  "resnet18_small_baseline_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_quantum_8_frozen"
  "resnet18_quantum_8_frozen"
  "resnet18_quantum_8_frozen"
)

SEEDS=(
  "21"
  "42"
  "84"
  "21"
  "42"
  "84"
  "21"
  "42"
  "84"
)

RUN_NAMES=(
  "dermamnist_A_frozen_seed21"
  "dermamnist_A_frozen_seed42"
  "dermamnist_A_frozen_seed84"
  "dermamnist_B_bottleneck8_frozen_seed21"
  "dermamnist_B_bottleneck8_frozen_seed42"
  "dermamnist_B_bottleneck8_frozen_seed84"
  "dermamnist_C_quantum8_frozen_seed21"
  "dermamnist_C_quantum8_frozen_seed42"
  "dermamnist_C_quantum8_frozen_seed84"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_NAME=${MODELS[$TASK_ID]}
SEED=${SEEDS[$TASK_ID]}
RUN_NAME=${RUN_NAMES[$TASK_ID]}

CONFIG_PATH=${CONFIG_TMP_DIR}/${RUN_NAME}.yaml

cat > "${CONFIG_PATH}" <<EOF
experiment:
  name: ${RUN_NAME}
  seed: ${SEED}
  output_dir: ${OUTPUT_ROOT}

dataset:
  root: ${DATA_ROOT}

training:
  batch_size: 128
  num_workers: 4
  device: cuda
  epochs: 20
  lr: 0.0005
  weight_decay: 0.01

model:
  name: ${MODEL_NAME}
  pretrained_backbone: ${PRETRAINED_BACKBONE}
EOF

echo "[INFO] TASK_ID=${TASK_ID}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] SEED=${SEED}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"

PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/scripts/run_experiment.py" --config "${CONFIG_PATH}"
