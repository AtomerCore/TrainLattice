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


class CNN(nn.Module):
    """Configurable convolutional network for classification."""

    def __init__(
        self,
        *,
        input_shape: Sequence[int] = (1, 28, 28),
        channels: Sequence[int] = (32, 64),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        height, width = int(input_shape[1]), int(input_shape[2])
        blocks: list[nn.Module] = []
        previous = int(input_shape[0])
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(previous, int(out_channels), kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            previous = int(out_channels)
            height //= 2
            width //= 2
        blocks.append(nn.Flatten())
        blocks.append(nn.Linear(previous * height * width, num_classes))
        self.network = nn.Sequential(*blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
