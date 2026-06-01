import torch.nn as nn

from src.models.backbones.resnet18_small import build_resnet18_small_backbone
from src.models.heads.linear_head import LinearHead
from src.models.heads.classical_bottleneck_head import ClassicalBottleneckHead
from src.models.heads.classical_bottleneck_linear_head import ClassicalBottleneckLinearHead
from src.models.wrappers.cnn_baseline import CNNBaselineModel
from src.models.heads.quantum_head import QuantumHead


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    model_name = model_cfg["name"]

    if model_name == "resnet18_small_baseline":
        backbone = build_resnet18_small_backbone()
        head = LinearHead(in_features=backbone.out_features, num_classes=num_classes)
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_small_baseline_frozen":
        backbone = build_resnet18_small_backbone()
        head = LinearHead(in_features=backbone.out_features, num_classes=num_classes)
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_bottleneck_4":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=4,
            num_classes=num_classes,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_4_frozen":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=4,
            num_classes=num_classes,
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_bottleneck_8":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8,
            num_classes=num_classes,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_8_frozen":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8,
            num_classes=num_classes,
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_bottleneck_32_8_frozen":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8,
            num_classes=num_classes,
            hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head, freeze_backbone=True)

    if model_name == "resnet18_bottleneck_32_8_partial":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8,
            num_classes=num_classes,
            hidden_dim=32,
        )
        return CNNBaselineModel(
            backbone=backbone,
            head=head,
            unfreeze_last_block=True,
        )

    if model_name == "resnet18_bottleneck_32_8_e2e":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=8,
            num_classes=num_classes,
            hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_16_4_e2e":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=4,
            num_classes=num_classes,
            hidden_dim=16,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_8_4_e2e":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=4,
            num_classes=num_classes,
            hidden_dim=8,
        )
        return CNNBaselineModel(backbone=backbone, head=head)


    if model_name == "resnet18_bottleneck_8_2_e2e":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckHead(
            in_features=backbone.out_features,
            bottleneck_dim=2,
            num_classes=num_classes,
            hidden_dim=8,
        )
        return CNNBaselineModel(backbone=backbone, head=head)






    if model_name == "resnet18_bottleneck_linear_8_2_e2e":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckLinearHead(
            in_features=backbone.out_features,
            bottleneck_dim=2,
            num_classes=num_classes,
            hidden_dim=8,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_bottleneck_linear_16_4_e2e":
        backbone = build_resnet18_small_backbone()
        head = ClassicalBottleneckLinearHead(
            in_features=backbone.out_features,
            bottleneck_dim=4,
            num_classes=num_classes,
            hidden_dim=16,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_quantum_8_frozen":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8,
            num_classes=num_classes,
        )
        return CNNBaselineModel(
            backbone=backbone,
            head=head,
            freeze_backbone=True,
        )

    if model_name == "resnet18_quantum_32_8_frozen":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8,
            num_classes=num_classes,
            hidden_dim=32,
        )
        return CNNBaselineModel(
            backbone=backbone,
            head=head,
            freeze_backbone=True,
        )

    if model_name == "resnet18_quantum_32_8_reupload_frozen":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8,
            num_classes=num_classes,
            hidden_dim=32,
            data_reuploading=True,
        )
        return CNNBaselineModel(
            backbone=backbone,
            head=head,
            freeze_backbone=True,
        )

    if model_name == "resnet18_quantum_32_8_ring_frozen":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8,
            num_classes=num_classes,
            hidden_dim=32,
            entanglement="ring",
        )
        return CNNBaselineModel(
            backbone=backbone,
            head=head,
            freeze_backbone=True,
        )

    if model_name == "resnet18_quantum_32_8_partial":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8,
            num_classes=num_classes,
            hidden_dim=32,
        )
        return CNNBaselineModel(
            backbone=backbone,
            head=head,
            unfreeze_last_block=True,
        )

    if model_name == "resnet18_quantum_32_8_e2e":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=8,
            num_classes=num_classes,
            hidden_dim=32,
        )
        return CNNBaselineModel(backbone=backbone, head=head)


    if model_name == "resnet18_quantum_8_2_e2e":
        backbone = build_resnet18_small_backbone()

        n_q_layers = model_cfg.get("quantum_n_q_layers", 2)
        data_reuploading = model_cfg.get("quantum_data_reuploading", False)
        entanglement = model_cfg.get("quantum_entanglement", "linear")

        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=2,
            num_classes=num_classes,
            hidden_dim=8,
            n_q_layers=n_q_layers,
            data_reuploading=data_reuploading,
            entanglement=entanglement,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_quantum_16_4_e2e":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=4,
            num_classes=num_classes,
            hidden_dim=16,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    if model_name == "resnet18_quantum_8_4_e2e":
        backbone = build_resnet18_small_backbone()
        head = QuantumHead(
            in_features=backbone.out_features,
            n_qubits=4,
            num_classes=num_classes,
            hidden_dim=8,
        )
        return CNNBaselineModel(backbone=backbone, head=head)

    raise ValueError(f"Unknown model name: {model_name}")
