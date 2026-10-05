"""Verify metric weighting and evaluation mode boundaries."""

import pytest

pytest.importorskip("torch")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainlattice.trainer import Trainer


@pytest.mark.parametrize("training", [True, False])
def test_validation_weights_uneven_batches_and_restores_mode(training) -> None:
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
        model.bias.zero_()
    inputs = torch.tensor([[3., 0.], [0., 3.], [3., 0.]])
    targets = torch.tensor([0, 1, 1])
    trainer = Trainer(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                      loss_fn=nn.CrossEntropyLoss(), device=torch.device("cpu"), epochs=1)
    model.train(training)
    metrics = trainer.validate(DataLoader(TensorDataset(inputs, targets), batch_size=2))
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["loss"] == pytest.approx(nn.functional.cross_entropy(inputs, targets).item())
    assert model.training is training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_empty_validation_restores_training_mode() -> None:
    model = nn.Linear(2, 2)
    trainer = Trainer(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                      loss_fn=nn.CrossEntropyLoss(), device=torch.device("cpu"), epochs=1)
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="no samples"):
        trainer.validate(empty)
    assert model.training
