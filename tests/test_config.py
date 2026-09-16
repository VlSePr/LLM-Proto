import pytest

from src.config import (
    MODEL_CONFIGS,
    ModelConfig,
    TrainConfig,
    config_from_dict,
    get_model_config,
    load_model_config,
    load_train_config,
)


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError, match="head_dim"):
        ModelConfig(dim=72, n_heads=8, n_kv_heads=4)  # head_dim = 9


def test_vocab_must_fit_uint16():
    with pytest.raises(ValueError, match="vocab_size"):
        ModelConfig(vocab_size=70_000)


def test_presets_are_copies():
    a = get_model_config("tiny")
    b = get_model_config("tiny")
    assert a is not b
    a.vocab_size = 1000
    assert get_model_config("tiny").vocab_size == 32_000


@pytest.mark.parametrize("name,millions", [
    ("tiny", 35), ("small", 100), ("medium", 303), ("base", 466), ("large", 1509),
])
def test_preset_param_counts_match_readme(name, millions):
    """MODEL_CONFIGS is the single source of truth; the README table must agree with it."""
    assert round(get_model_config(name).param_count_estimate() / 1e6) == millions


def test_yaml_scientific_notation_string_is_coerced(tmp_path):
    p = tmp_path / "train.yaml"
    p.write_text("peak_lr: 3e-4\nmin_lr: '6e-5'\nmax_steps: '100'\nuse_wandb: 'false'\n")
    cfg = load_train_config(str(p))
    assert isinstance(cfg.peak_lr, float) and cfg.peak_lr == pytest.approx(3e-4)
    assert cfg.min_lr == pytest.approx(6e-5)
    assert cfg.max_steps == 100
    assert cfg.use_wandb is False


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "model.yaml"
    p.write_text("dim: 64\nn_heads: 4\nn_kv_heads: 2\nn_layer: 2\n")  # typo: n_layer
    with pytest.raises(ValueError, match="n_layer"):
        load_model_config(str(p))


def test_empty_yaml_gives_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_train_config(str(p)) == TrainConfig()


def test_bad_bool_rejected():
    with pytest.raises(ValueError):
        config_from_dict(TrainConfig, {"use_wandb": "maybe"})


def test_param_count_estimate_matches_model(tiny_cfg):
    from src.model import TransformerLM
    model = TransformerLM(tiny_cfg)
    assert tiny_cfg.param_count_estimate() == model.count_parameters(trainable_only=False)


@pytest.mark.parametrize("name", list(MODEL_CONFIGS))
def test_param_count_breakdown_sums_to_estimate(name):
    cfg = get_model_config(name)
    assert sum(cfg.param_count_breakdown().values()) == cfg.param_count_estimate()
