"""Task-aware metrics and sample-weighted aggregation."""

from dataclasses import dataclass

from torch import Tensor


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Return the number of correct predictions as a float."""
    return float((logits.argmax(dim=1) == targets).sum().item())


@dataclass
class LossTotals:
    """Accumulate loss over uneven batches for regression objectives."""

    loss: float = 0.0
    samples: int = 0

    def update(self, outputs: Tensor, targets: Tensor, loss: Tensor) -> dict[str, float]:
        """Record a batch and return its scalar metrics."""
        samples = int(targets.shape[0])
        if samples == 0:
            raise ValueError("regression batch contains no samples")
        batch_loss = float(loss.item())
        self.loss += batch_loss * samples
        self.samples += samples
        return {"loss": batch_loss}

    def compute(self, phase: str) -> dict[str, float]:
        """Return sample means, rejecting an empty data loader."""
        if self.samples == 0:
            raise ValueError(f"{phase} loader produced no samples")
        return {"loss": self.loss / self.samples}


@dataclass
class ClassificationTotals:
    """Accumulate loss and correct predictions over uneven batches."""

    loss: float = 0.0
    correct: float = 0.0
    samples: int = 0

    def update(self, logits: Tensor, targets: Tensor, loss: Tensor) -> dict[str, float]:
        """Record a batch and return its scalar metrics."""
        samples = int(targets.shape[0])
        if samples == 0:
            raise ValueError("classification batch contains no samples")
        batch_loss = float(loss.item())
        correct = accuracy(logits, targets)
        self.loss += batch_loss * samples
        self.correct += correct
        self.samples += samples
        return {"loss": batch_loss, "accuracy": correct / samples}

    def compute(self, phase: str) -> dict[str, float]:
        """Return sample means, rejecting an empty data loader."""
        if self.samples == 0:
            raise ValueError(f"{phase} loader produced no samples")
        return {"loss": self.loss / self.samples, "accuracy": self.correct / self.samples}
