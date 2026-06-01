"""
quantum_head.py
---------------
Hybrid quantum classifier head for HQCNN models.

Implements Model C from the controlled A/B/C evaluation framework.
The quantum head compresses backbone features into a low-dimensional
latent space, processes them through a variational quantum circuit (VQC),
and maps the measurement outputs to class logits.

Architecture:
    backbone output (512-d)
        → input projection (512 → hidden_dim → n_qubits)
        → tanh scaling to [-π, π]
        → AngleEmbedding (RY rotations)
        → L variational layers (trainable RY + CNOT entanglement)
        → Pauli-Z expectation values
        → linear classifier (n_qubits → num_classes)

Configurable parameters:
    - n_qubits:        number of qubits (= bottleneck dimension)
    - n_q_layers:      number of variational layers (depth)
    - hidden_dim:      optional intermediate projection dimension
    - data_reuploading: re-embed inputs at each variational layer
    - entanglement:    'linear' (CNOT chain) or 'ring' (+ wrap-around)

Implemented with PennyLane + PyTorch integration.
Trained entirely on a classical simulator (default.qubit).
"""

import torch
import torch.nn as nn
import pennylane as qml


class QuantumHead(nn.Module):
    """
    Variational quantum classifier head.

    This is Model C in the HQCNN evaluation framework. It replaces the
    classical bottleneck classifier with a parameterised quantum circuit
    (PQC) operating on a compressed low-dimensional feature vector.

    Parameters
    ----------
    in_features : int
        Dimensionality of the backbone output (typically 512 for ResNet-18).
    n_qubits : int
        Number of qubits in the quantum circuit. Also defines the bottleneck
        dimension (one qubit per input feature after projection).
    num_classes : int
        Number of output classes.
    n_q_layers : int, optional
        Number of variational layers (default: 2).
    hidden_dim : int or None, optional
        If provided, adds an intermediate linear projection layer before the
        final n_qubits projection, creating a two-stage compression pathway
        (e.g. 512 → hidden_dim → n_qubits). Default: None (single projection).
    data_reuploading : bool, optional
        If True, re-embeds the classical input at each variational layer.
        Increases effective circuit expressivity without adding qubits.
        Default: False.
    entanglement : str, optional
        Entanglement pattern for CNOT gates. Options:
        - 'linear': CNOT chain (0→1, 1→2, ..., n-2→n-1)
        - 'ring':   linear chain + additional CNOT (n-1→0)
        Default: 'linear'.

    Notes
    -----
    Input features are scaled via tanh to the range [-π, π] before
    angle embedding to ensure compatibility with the RY gate domain.
    Gradient optimisation uses the parameter-shift rule via PennyLane.
    """

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

        # --- Classical input projection ---
        if hidden_dim is None:
            # Single-stage: 512 → n_qubits
            self.input_proj = nn.Linear(in_features, n_qubits)
        else:
            # Two-stage: 512 → hidden_dim → n_qubits
            self.input_proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_qubits),
            )

        # --- Quantum circuit definition ---
        dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (n_q_layers, n_qubits)}

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            # Optional single embedding at circuit entry (non-reuploading mode)
            if not data_reuploading:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            for layer in range(n_q_layers):
                # Re-uploading: embed inputs at each variational layer
                if data_reuploading:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

                # Trainable single-qubit rotations
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)

                # Entanglement layer
                if entanglement == "linear":
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                elif entanglement == "ring":
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])
                else:
                    raise ValueError(
                        f"Unknown entanglement mode: '{entanglement}'. "
                        f"Choose 'linear' or 'ring'."
                    )

            # Measurement: Pauli-Z expectation values on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # --- Classical output projection ---
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum head.

        Parameters
        ----------
        x : torch.Tensor
            Backbone feature vector, shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Class logits, shape (batch_size, num_classes).
        """
        # Compress features to qubit-compatible dimensionality
        x = self.input_proj(x)

        # Scale to [-π, π] for angle encoding
        x = torch.tanh(x) * torch.pi

        # Quantum circuit forward pass
        x = self.quantum_layer(x)

        # Map measurements to class logits
        x = self.classifier(x)
        return x
