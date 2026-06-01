"""
factory.py
----------
Model factory for the HQCNN evaluation framework.

Creates any model variant from a configuration dictionary.
All models share the same ResNet-18 backbone. Only the classifier
head varies, enabling direct comparison under controlled conditions.

Model naming convention
-----------------------
    resnet18_{head}_{interface}_{regime}

    head:
        baseline          → Model A  (linear, no bottleneck)
        bottleneck        → Model B  (nonlinear MLP bottleneck)
        bottleneck_linear → B-linear (capacity-matched linear bottleneck)
        quantum           → Model C  (variational quantum classifier)

    interface (optional):
        8                 → direct 512 → 8 compression
        32_8              → two-stage 512 → 32 → 8 compression
        16_4              → two-stage 512 → 16 → 4 compression
        8_2               → two-stage 512 → 8 → 2 compression

    regime:
        (none)            → end-to-end training
        frozen            → backbone frozen
        partial           → only last ResNet block trainable
        e2e               → explicit end-to-end (alias for no suffix)

Usage
-----
    from src.models.factory import build_model

    model = build_model(
        model_cfg={"name": "resnet18_quantum_32_8_e2e"},
        num_classes=7
    )
"""

import torch.nn as nn

from src.models.backbones.resnet18_small import build_resnet18_small_backbone
from src.models.heads.linear_head import LinearHead
from src.models.heads.classical_bottleneck_head import ClassicalBottleneckHead
from src.models.heads.classical_bottleneck_linear_head import ClassicalBottleneckLinearHead
from src.models.heads.quantum_head import QuantumHead
from src.models.wrappers.cnn_baseline import CNNBaselineModel


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    """
    Build a model from a configuration dictionary.

    Parameters
    ----------
    model_cfg : dict
        Must contain a 'name' key with the model identifier string.
        Quantum models may optionally include:
            - 'quantum_n_q_layers' (int): number of variational layers
            - 'quantum_data_reuploading' (bool): enable data re-uploading
            - 'quantum_entanglement' (str): 'linear' or 'ring'
    num_classes : int
        Number of output classes (7 for DermaMNIST).

    Returns
    -------
    nn.Module
        Assembled model ready for training.

    Raises
    ------
    ValueError
        If model_cfg['name'] does not match any known model.

    Examples
    --------
    # Model A — linear baseline (frozen backbone)
    model = build_model({"name": "resnet18_small_baseline_frozen"}, 7)

    # Model B — nonlinear bottleneck 512→32→8 (end-to-end)
    model = build_model({"name": "resnet18_bottleneck_32_8_e2e"}, 7)

    # Model C — 8-qubit quantum head 512→32→8 (end-to-end)
    model = build_model({"name": "resnet18_quantum_32_8_e2e"}, 7)

    # Model C — 2-qubit quantum head 512→8→2 with ring entanglement
    model = build_model({
        "name": "resnet18_quantum_8_2_e2e",
        "quantum_n_q_layers": 2,
        "quantum_data_reuploading": True,
        "quantum_entanglement": "ring"
    }, 7)
    """
    model_name = model_cfg["name"]

    # ------------------------------------------------------------------
    # Model A: Linear baseline (no bottleneck)
    # ------------------------------------------------------------------

    if model_name == "resnet18_small_baseline":
        backbone = build_resnet18_small_backbone()
        head = LinearHead(in_features=backbone.out_features, num_classes=num_classes)
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_small_baseline_frozen":
        backbone = build_resnet18_small_backbone()
        head = LinearHead(in_features=backbone.out_features, num_classes=num_classes)
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    # ------------------------------------------------------------------
    # Model B: Classical bottleneck (nonlinear MLP)
    # ------------------------------------------------------------------

    if model_name == "resnet18_bottleneck_4":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features, bottleneck_dim=4, num_classes=num_classes
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_4_frozen":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features, bottleneck_dim=4, num_classes=num_classes
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_bottleneck_8":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features, bottleneck_dim=8, num_classes=num_classes
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_8_frozen":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features, bottleneck_dim=8, num_classes=num_classes
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_bottleneck_32_8_frozen":
        # Improved interface: 512 → 32 → 8 (Chapter 7, Experiment 3)
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8, num_classes=num_classes, hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_bottleneck_32_8_partial":
        # Partial fine-tuning: only last ResNet block trainable (Chapter 7, Experiment 5)
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8, num_classes=num_classes, hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head, unfreeze_last_block=True)

    if model_name == "resnet18_bottleneck_32_8_e2e":
        # End-to-end training: 512 → 32 → 8 (Chapter 8, Experiment 9)
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8, num_classes=num_classes, hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_16_4_e2e":
        # Capacity-constrained: 512 → 16 → 4 (Chapter 10, Experiment 11)
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=4, num_classes=num_classes, hidden_dim=16,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_8_2_e2e":
        # Severe compression: 512 → 8 → 2 (Chapter 12, transition regime)
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=2, num_classes=num_classes, hidden_dim=8,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    # ------------------------------------------------------------------
    # B-linear: Capacity-matched linear bottleneck (no activations)
    # Key comparator for Gap 1 (baseline mismatch)
    # ------------------------------------------------------------------

    if model_name == "resnet18_bottleneck_linear_8_2_e2e":
        # B-linear 8→2: introduced in Chapter 11 as corrected comparator
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckLinearHead(
            in_features=backbone.out_features,
            bottleneck_dim=2, num_classes=num_classes, hidden_dim=8,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_linear_16_4_e2e":
        # B-linear 16→4: main comparator for Chapter 11 low-data analysis
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckLinearHead(
            in_features=backbone.out_features,
            bottleneck_dim=4, num_classes=num_classes, hidden_dim=16,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    # ------------------------------------------------------------------
    # Model C: Hybrid quantum head (variational quantum classifier)
    # ------------------------------------------------------------------

    if model_name == "resnet18_quantum_8_frozen":
        # Direct 512→8 compression, frozen backbone (Chapter 6 baseline)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8, num_classes=num_classes,
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_quantum_32_8_frozen":
        # Improved interface 512→32→8, frozen (Chapter 7, Experiment 3)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8, num_classes=num_classes, hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_quantum_32_8_reupload_frozen":
        # Data re-uploading ablation (Chapter 7, Experiment 7)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8, num_classes=num_classes, hidden_dim=32,
            data_reuploading=True,
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_quantum_32_8_ring_frozen":
        # Ring entanglement ablation (Chapter 7, Experiment 8)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8, num_classes=num_classes, hidden_dim=32,
            entanglement="ring",
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_quantum_32_8_partial":
        # Partial fine-tuning ablation (Chapter 7, Experiment 5)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8, num_classes=num_classes, hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head, unfreeze_last_block=True)

    if model_name == "resnet18_quantum_32_8_e2e":
        # End-to-end training (Chapter 8, Experiment 9)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8, num_classes=num_classes, hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_quantum_16_4_e2e":
        # Capacity-constrained 4-qubit (Chapter 10, Experiment 11)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=4, num_classes=num_classes, hidden_dim=16,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_quantum_8_2_e2e":
        # Transition regime: 2-qubit, configurable architecture
        # (Chapters 12–14, Experiments 20–25)
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=2,
            num_classes=num_classes,
            hidden_dim=8,
            n_q_layers=model_cfg.get("quantum_n_q_layers", 2),
            data_reuploading=model_cfg.get("quantum_data_reuploading", False),
            entanglement=model_cfg.get("quantum_entanglement", "linear"),
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    raise ValueError(
        f"Unknown model name: '{model_name}'.\n"
        f"See src/models/factory.py for all available model identifiers."
    )
