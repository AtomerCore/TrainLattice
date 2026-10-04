"""Built-in configurable neural network architectures."""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """Configurable multilayer perceptron for classification."""

    def __init__(
        self,
        *,
        input_dim: int = 784,
        hidden_dims: Sequence[int] = (512, 256, 128),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.GELU(), nn.Dropout(0.1)])
            previous = int(width)
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
