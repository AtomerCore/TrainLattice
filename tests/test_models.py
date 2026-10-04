import pytest

torch = pytest.importorskip("torch")

from trainlattice.config.schema import ModelConfig
from trainlattice.errors import ConfigError
from trainlattice.modeling import ARCHITECTURES, build_model, validate_model_config
from trainlattice.modeling.architectures import MLP


def test_builds_configurable_mlp() -> None:
    config = ModelConfig("mlp", {"input_dim": 12, "hidden_dims": [8, 4], "num_classes": 3})
    model = build_model(config)

    assert isinstance(model, MLP)
    assert model(torch.randn(2, 3, 2, 2)).shape == (2, 3)
    dropouts = [module for module in model.network if isinstance(module, torch.nn.Dropout)]
    assert len(dropouts) == 2
    assert all(module.p == 0.1 for module in dropouts)


def test_architecture_registry_exposes_only_mlp() -> None:
    assert ARCHITECTURES.names() == ("mlp",)


def test_rejects_unsupported_architecture() -> None:
    with pytest.raises(ConfigError, match="unsupported model.name 'unknown'"):
        build_model(ModelConfig("unknown", {}))


def test_rejects_unsupported_model_options() -> None:
    config = ModelConfig("mlp", {"input_dim": 4, "extra": True})

    with pytest.raises(ConfigError, match="model.extra"):
        validate_model_config(config)


def test_rejects_invalid_mlp_dimensions() -> None:
    with pytest.raises(ConfigError, match="model.input_dim must be a positive integer"):
        build_model(ModelConfig("mlp", {"input_dim": 0}))
    with pytest.raises(ConfigError, match=r"model.hidden_dims\[1\] must be a positive integer"):
        build_model(ModelConfig("mlp", {"input_dim": 4, "hidden_dims": [8, -1]}))
    with pytest.raises(ConfigError, match="model.num_classes must be a positive integer"):
        build_model(ModelConfig("mlp", {"input_dim": 4, "num_classes": True}))
