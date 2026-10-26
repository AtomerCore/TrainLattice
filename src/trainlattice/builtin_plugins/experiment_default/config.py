"""Strict domain validation after the native service graph is bound."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from trainlattice.contracts.components import compare_contracts
from trainlattice.contracts.config import ExperimentConfig
from trainlattice.errors import ConfigError


def validate(raw: Mapping[str, Any], services: dict[str, Any]) -> ExperimentConfig:
    """Validate component options and contracts before constructing components."""
    data = deepcopy({key: value for key, value in raw.items() if key not in {"profile", "plugins", "bindings"}})
    for category in ("model", "dataset", "optimizer", "loss", "scheduler"):
        options = data.get(category)
        factory = services[category]
        if factory is None:
            if options is not None:
                raise ConfigError(f"{category} requires a service in bindings.org.trainlattice.experiment.default.{category}")
            continue
        if options is None:
            options = {}
        if not isinstance(options, Mapping):
            raise ConfigError(f"{category} must be a mapping")
        if "name" in options:
            raise ConfigError(f"{category}.name is selected through bindings.org.trainlattice.experiment.default.{category}")
        data[category] = {"name": factory.name, **options}
    config = ExperimentConfig.from_dict(data)
    for category in ("model", "dataset", "optimizer", "loss", "scheduler"):
        section = getattr(config, category)
        if section is not None:
            services[category].validate(category, section.options)
    if services["loss"].task.kind not in {"classification", "regression"}:
        raise ConfigError("loss.task must be classification or regression")
    if config.trainer.monitor == "val_accuracy" and services["loss"].task.kind != "classification":
        raise ConfigError("trainer.monitor 'val_accuracy' requires a classification loss")
    model, dataset = services["model"], services["dataset"]
    if model.contract is not None and dataset.contract is not None:
        errors = compare_contracts(model.contract(config.model.options), dataset.contract(config.dataset.options))
        if errors:
            raise ConfigError("; ".join(errors))
    if config.scheduler is not None and config.scheduler.options.get("min_lr", 0) > config.optimizer.lr:
        raise ConfigError("scheduler.min_lr must not exceed optimizer.lr")
    return config


def native_config(config: ExperimentConfig, snapshot: Any) -> dict[str, Any]:
    """Produce a reloadable, resolved native configuration for run artifacts."""
    data = config.to_dict()
    for category in ("model", "dataset", "optimizer", "loss", "scheduler"):
        if isinstance(data.get(category), dict):
            data[category].pop("name", None)
    plugins = snapshot.to_dict()["plugins"]
    return {"plugins": [{"source": item["source"] if item["source"].startswith("builtin://") else item["path"],
                          "config": item["config"]} for item in plugins],
            "bindings": {item["id"]: item["bindings"] for item in plugins if item["bindings"]}, **data}
