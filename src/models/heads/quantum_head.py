import torch
import torch.nn as nn
import pennylane as qml


class QuantumHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_qubits: int,
        num_classes: int,
        n_q_layers: int = 2,
        hidden_dim: int | None = None,
        data_reuploading: bool = False,
        entanglement: str = "linear",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.data_reuploading = data_reuploading
        self.entanglement = entanglement

        if hidden_dim is None:
            self.input_proj = nn.Linear(in_features, n_qubits)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_qubits),
            )

        dev = qml.device("default.qubit", wires=n_qubits)

        weight_shapes = {"weights": (n_q_layers, n_qubits)}

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            if not data_reuploading:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            for layer in range(n_q_layers):
                if data_reuploading:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)

                # Entanglement pattern
                if entanglement == "linear":
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])

                elif entanglement == "ring":
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])

                else:
                    raise ValueError(f"Unknown entanglement mode: {entanglement}")

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = torch.tanh(x) * torch.pi
        x = self.quantum_layer(x)
        x = self.classifier(x)
        return x
