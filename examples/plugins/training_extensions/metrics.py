"""Extend sample-weighted metrics with the mean model prediction."""


class PredictionTotals:
    """Decorate a task's totals without duplicating its loss semantics."""

    def __init__(self, base, scale):
        self.base = base
        self.scale = scale
        self.prediction_sum = 0.0

    @property
    def samples(self):
        """Expose the sample count used by throughput calculations."""
        return self.base.samples

    def update(self, outputs, targets, loss):
        """Weight each sample's prediction mean equally."""
        self.prediction_sum += float(outputs.float().reshape(outputs.shape[0], -1).mean(1).sum().item())
        return self.base.update(outputs, targets, loss)

    def compute(self, phase):
        """Return task metrics and the scaled sample mean prediction."""
        values = self.base.compute(phase)
        return {**values, "mean_prediction": self.scale * self.prediction_sum / self.samples}
