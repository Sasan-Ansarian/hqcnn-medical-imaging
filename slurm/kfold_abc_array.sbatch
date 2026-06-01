#!/bin/bash
#SBATCH --job-name=kfold_abc
#SBATCH --output=/scratch/lustre/home/saan1491/hqcnn_project/logs/kfold_abc_%A_%a.out
#SBATCH --error=/scratch/lustre/home/saan1491/hqcnn_project/logs/kfold_abc_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-14

set -euo pipefail

PROJECT_ROOT=/scratch/lustre/home/saan1491/hqcnn_project
VENV_PATH=/scratch/lustre/home/saan1491/venvs/hqcnn
OUTPUT_ROOT=${PROJECT_ROOT}/outputs
DATA_ROOT=${PROJECT_ROOT}/data
CONFIG_TMP_DIR=${PROJECT_ROOT}/configs/generated_kfold_abc
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
  "resnet18_small_baseline_frozen"
  "resnet18_small_baseline_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_bottleneck_8_frozen"
  "resnet18_quantum_8_frozen"
  "resnet18_quantum_8_frozen"
  "resnet18_quantum_8_frozen"
  "resnet18_quantum_8_frozen"
  "resnet18_quantum_8_frozen"
)

MODEL_TAGS=(
  "A"
  "A"
  "A"
  "A"
  "A"
  "B"
  "B"
  "B"
  "B"
  "B"
  "C"
  "C"
  "C"
  "C"
  "C"
)

FOLDS=(
  "0"
  "1"
  "2"
  "3"
  "4"
  "0"
  "1"
  "2"
  "3"
  "4"
  "0"
  "1"
  "2"
  "3"
  "4"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_NAME=${MODELS[$TASK_ID]}
MODEL_TAG=${MODEL_TAGS[$TASK_ID]}
FOLD_INDEX=${FOLDS[$TASK_ID]}
SEED=42
RUN_NAME=dermamnist_${MODEL_TAG}_kfold_fold${FOLD_INDEX}

CONFIG_PATH=${CONFIG_TMP_DIR}/${RUN_NAME}.yaml

cat > "${CONFIG_PATH}" <<EOF
experiment:
  name: ${RUN_NAME}
  seed: ${SEED}
  output_dir: ${OUTPUT_ROOT}

dataset:
  root: ${DATA_ROOT}
  use_kfold: true
  fold_index: ${FOLD_INDEX}
  n_splits: 5

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
echo "[INFO] MODEL_TAG=${MODEL_TAG}"
echo "[INFO] FOLD_INDEX=${FOLD_INDEX}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"

PYTHONPATH="${PROJECT_ROOT}" python "${PROJECT_ROOT}/scripts/run_experiment.py" --config "${CONFIG_PATH}"
