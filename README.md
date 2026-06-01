# HQCNN for Medical Image Classification

**Hybrid Quantum–Classical Convolutional Neural Networks under Fair Baselines**

A controlled benchmarking study of hybrid quantum–classical CNN models for medical image classification, developed as part of a PhD Deep Neural Networks course at Vilnius University, Faculty of Mathematics and Informatics.

---

## Overview

This repository implements and evaluates a hybrid quantum–classical CNN framework on the [DermaMNIST](https://medmnist.com/) benchmark. The study investigates **under what conditions hybrid quantum models become competitive with classical alternatives**, using a strictly controlled evaluation protocol designed to address three methodological gaps identified in a systematic review of 153 QML studies.

**Key finding:** Classical baselines consistently outperform the hybrid quantum head under standard conditions. A narrow transition regime emerges under extreme bottleneck compression (`8→2`) and data scarcity (20%), but this effect is metric-dependent and does not generalise across datasets.

---

## Model Variants

| Model | Type | Bottleneck | Description |
|---|---|---|---|
| **A** | Linear baseline | None | Linear classifier on backbone output |
| **B-linear** | Classical bottleneck | d (linear) | Capacity-matched linear comparator |
| **B-MLP** | Nonlinear bottleneck | d (ReLU) | Tests whether quantum edge is artefact of linearity |
| **C** | Hybrid quantum head | d qubits | VQC with angle embedding + variational layers |

All models share a pretrained **ResNet-18** backbone. Only the classifier head varies.

---

## Architecture

```
Input image (28×28 RGB)
        ↓
ResNet-18 backbone (pretrained)
        ↓
512-dimensional feature vector
        ↓
Bottleneck compression (512 → d)
        ↓
┌─────────────────────────────────┐
│  Model A: Linear classifier     │
│  Model B: Classical MLP/linear  │
│  Model C: Variational QC (VQC)  │
│    AngleEmbedding → RY rotations│
│    CNOT entanglement (L layers) │
│    Pauli-Z measurements         │
└─────────────────────────────────┘
        ↓
Class logits (7 classes)
```

**Quantum circuit:** PennyLane + PyTorch integration. Default: 8 qubits, L=2 variational layers, linear entanglement, angle embedding.

---

## Key Results

| Phase | Configuration | Best classical F1 | Quantum F1 | Verdict |
|---|---|---|---|---|
| Frozen baseline | `512→8`, 100% data | A: 0.472 | C: 0.317 | Classical wins |
| Best interface | `512→32→8`, 100% | B: 0.473 | C: 0.464 | Classical wins |
| End-to-end | `512→32→8`, 100% | B: 0.502 | C: 0.471 | Classical wins |
| Low-data fairness | `16→4`, 20% data | B-lin: 0.405 | C: 0.247 | Classical wins |
| Transition regime | `8→2`, 20% data | B-lin: 0.180 (F1) | C: 0.414 (acc) | Metric-dependent |

**Primary metric:** Macro F1 (class-imbalanced dataset).
**Evaluation:** Multi-seed {21, 42, 84} + 5-fold stratified cross-validation.

---

## Repository Structure

```
hqcnn-medical-imaging/
│
├── src/
│   ├── models/
│   │   ├── backbones/          # ResNet-18, DenseNet-121, EfficientNet-B0
│   │   ├── heads/              # Model A, B-linear, B-MLP, C (quantum)
│   │   ├── wrappers/           # Full model assembly
│   │   └── factory.py          # Model creation from config
│   ├── data/
│   │   └── dataset.py          # MedMNIST loading + preprocessing
│   ├── train/
│   │   └── engine.py           # Training loop + evaluation
│   └── utils/
│       ├── metrics.py          # Macro F1, accuracy, balanced accuracy
│       ├── gradient_tracking.py # Gradient norm + variance logging
│       └── seed.py             # Reproducibility utilities
│
├── configs/                    # YAML experiment configurations
│   ├── baseline/               # Chapters 3–6 baseline experiments
│   ├── chapter7/               # E1–E8: controlled hypothesis experiments
│   ├── chapter10/              # E11–E15: low-data + compression
│   ├── chapter11/              # E16–E19: benchmark correction
│   ├── chapter12/              # E20: transition regime search
│   ├── chapter13/              # E21–E22: fairness analysis
│   ├── chapter14/              # E23–E25: quantum architecture
│   ├── chapter15/              # E26: cross-dataset screening
│   └── chapter16/              # E27: end-to-end cross-dataset
│
├── slurm/                      # SLURM batch scripts for VU MIF HPC
│   ├── train_single.sbatch     # Single experiment
│   └── train_array.sbatch      # Array job for multi-seed runs
│
├── notebooks/
│   └── demo_colab.ipynb        # Colab-ready demonstration notebook
│
├── docs/
│   └── report.pdf              # Final course report
│
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/Sasan-Ansarian/hqcnn-medical-imaging.git
cd hqcnn-medical-imaging
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch ≥ 2.0
- PennyLane ≥ 0.35
- MedMNIST
- scikit-learn

---

## Running Experiments

### Single experiment (local or interactive HPC session)

```bash
python scripts/run_experiment.py --config configs/baseline/model_abc_frozen.yaml
```

### HPC batch job (SLURM)

```bash
# Single run
sbatch slurm/train_single.sbatch

# Multi-seed array job
sbatch slurm/train_array.sbatch
```

### Configuration structure

```yaml
# Example: configs/chapter7/exp3_interface.yaml
model:
  backbone: resnet18_small
  head: quantum          # linear | bottleneck | quantum
  bottleneck_dim: 8
  n_qubits: 8
  n_layers: 2

training:
  epochs: 20
  batch_size: 128
  lr_head: 5e-4
  lr_backbone: 5e-5      # only used in e2e regime
  optimizer: adamw
  weight_decay: 0.01
  loss: weighted_ce

data:
  dataset: dermamnist
  data_fraction: 1.0
  seed: 42
```

---

## Experimental Programme

27 controlled experiments organised across 10 chapters:

| Chapter | Experiments | Focus |
|---|---|---|
| 3–6 | B1–B7 | Backbone selection, bottleneck study, baseline |
| 7 | E1–E8 | Controlled hypothesis-driven experiments |
| 8 | E9 | End-to-end training |
| 9 | E10 | Multi-seed integrated analysis |
| 10 | E11–E15 | Compression + low-data regime discovery |
| 11 | E16–E19 | Benchmark correction (B-linear comparator) |
| 12 | E20 | Transition regime search (`8→2`) |
| 13 | E21–E22 | Fairness analysis + robustness |
| 14 | E23–E25 | Quantum architecture ablations |
| 15 | E26 | Cross-dataset screening |
| 16 | E27 | End-to-end cross-dataset evaluation |

---

## HPC Infrastructure

Executed on **Vilnius University MIF HPC** (Tesla V100 GPU partition).

- Scheduler: SLURM
- GPU: Tesla V100
- Framework: PyTorch + PennyLane (classical simulation)
- Environment: Python 3.10, Singularity container

---

## Citation

```bibtex
@misc{ansarian2025hqcnn,
  author    = {Ansarian, Sasan and Paulavi{\v{c}}ius, Remigijus
               and Filatovas, Ernestas},
  title     = {Revisiting Hybrid Quantum--Classical Neural Networks
               for Medical Image Classification under Fair Baselines},
  year      = {2025},
  note      = {PhD DNN Course Project, Vilnius University}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Vilnius University · Faculty of Mathematics and Informatics ·
Institute of Data Science and Digital Technologies*
