import pytest

torch = pytest.importorskip("torch")

from trainlattice.contracts.config import ModelConfig
from trainlattice.errors import ConfigError
from helpers import get_component, build_model, validate_model_config
from trainlattice.kernel.errors import PluginError


def test_builds_configurable_mlp() -> None:
    config = ModelConfig("mlp", {"input_dim": 12, "hidden_dims": [8, 4], "num_classes": 3})
    model = build_model(config)

    assert type(model).__name__ == "MLP"
    assert model(torch.randn(2, 3, 2, 2)).shape == (2, 3)
    dropouts = [module for module in model.network if isinstance(module, torch.nn.Dropout)]
    assert len(dropouts) == 2
    assert all(module.p == 0.1 for module in dropouts)


def test_model_services_expose_cnn_and_mlp() -> None:
    assert get_component("model", "cnn").name == "cnn"
    assert get_component("model", "mlp").name == "mlp"


def test_rejects_unsupported_architecture() -> None:
    with pytest.raises(PluginError, match="service unavailable: trainlattice.model/unknown"):
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


def test_builds_configurable_cnn() -> None:
    config = ModelConfig("cnn", {"input_shape": [1, 8, 8], "channels": [4, 8], "num_classes": 3})
    model = build_model(config)

    assert type(model).__name__ == "CNN"
    assert model(torch.randn(2, 1, 8, 8)).shape == (2, 3)
    convs = [module for module in model.network if isinstance(module, torch.nn.Conv2d)]
    assert [(conv.in_channels, conv.out_channels) for conv in convs] == [(1, 4), (4, 8)]


def test_rejects_invalid_cnn_options() -> None:
    with pytest.raises(ConfigError, match="model.input_shape must contain three positive integers"):
        build_model(ModelConfig("cnn", {"input_shape": [1, 28]}))
    with pytest.raises(ConfigError, match=r"model.channels\[0\] must be a positive integer"):
        build_model(ModelConfig("cnn", {"channels": [0, 8]}))
    with pytest.raises(ConfigError, match="model.num_classes must be a positive integer"):
        build_model(ModelConfig("cnn", {"num_classes": 0}))
    with pytest.raises(ConfigError, match="model.kernel"):
        validate_model_config(ModelConfig("cnn", {"kernel_size": 5}))


def test_rejects_cnn_shape_below_pooling_factor() -> None:
    with pytest.raises(ConfigError, match="divisible by the pooling factor 4"):
        build_model(ModelConfig("cnn", {"input_shape": [1, 8, 15], "channels": [4, 8]}))


def test_cnn_contract_states_structured_shape() -> None:
    contract = get_component("model", "cnn").contract

    assert contract is not None
    declared = contract({"input_shape": [3, 32, 32], "num_classes": 5})
    assert declared.input_shape == (3, 32, 32)
    assert declared.input_dim is None
    assert declared.num_classes == 5
