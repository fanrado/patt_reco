"""Dotted-path config overrides, and the training config they act on.

`--set` is what the difficulty sweep drives the generator with, so a silently
ignored override would corrupt a whole sweep while every run still looked
healthy. Every rejection path is asserted.
"""
import pytest

from patt_reco.cliutil import apply_override, apply_overrides
from patt_reco.config import SourceConfig
from patt_reco.train.config import (DataConfig, ModelConfig, OptimConfig, RunConfig,
                                    TrainConfig, load_train_yaml)

# --------------------------------------------------------------------------- #
# overrides
# --------------------------------------------------------------------------- #


def test_a_top_level_field_is_replaced():
    cfg = apply_override(SourceConfig(), "seed=42")

    assert cfg.seed == 42


def test_a_nested_field_is_replaced_without_disturbing_its_siblings():
    base = SourceConfig()
    cfg = apply_override(base, "render.height=64")

    assert cfg.render.height == 64
    assert cfg.render.width == base.render.width
    assert cfg.track == base.track


def test_the_original_config_is_left_alone():
    """The dataclasses are frozen and hashable; an override rebuilds rather
    than mutates, so a caller holding the old config still sees the old one."""
    base = SourceConfig()
    apply_override(base, "render.height=64")

    assert base.render.height == 128


def test_lists_become_tuples_so_the_result_stays_hashable():
    cfg = apply_override(SourceConfig(), "track.curvature=[0.5, 3.0]")

    assert cfg.track.curvature == (0.5, 3.0)
    hash(cfg.track)


@pytest.mark.parametrize("spec,field,expected", [
    ("seed=7", "seed", 7),
    ("kind=shower", "kind", "shower"),
    ("name=hard tracks", "name", "hard tracks"),
    ("n_train=1000", "n_train", 1000),
])
def test_values_are_parsed_as_yaml(spec, field, expected):
    assert getattr(apply_override(SourceConfig(), spec), field) == expected


def test_a_float_stays_a_float():
    cfg = apply_override(SourceConfig(), "track.step=0.002")

    assert cfg.track.step == pytest.approx(0.002)


def test_a_value_containing_an_equals_sign_survives():
    """The spec splits on the FIRST '=', so the value may contain more."""
    cfg = apply_override(SourceConfig(), "name=a=b")

    assert cfg.name == "a=b"


def test_overrides_are_applied_in_order():
    cfg = apply_overrides(SourceConfig(), ["seed=1", "seed=2", "render.height=8"])

    assert cfg.seed == 2
    assert cfg.render.height == 8


def test_no_overrides_leaves_the_config_untouched():
    base = SourceConfig()

    assert apply_overrides(base, []) == base


# --------------------------------------------------------------------------- #
# rejection
# --------------------------------------------------------------------------- #


def test_an_unknown_top_level_field_is_rejected_by_name():
    with pytest.raises(SystemExit, match="wobble"):
        apply_override(SourceConfig(), "wobble=3")


def test_an_unknown_nested_field_is_rejected_by_name():
    with pytest.raises(SystemExit, match="bragg"):
        apply_override(SourceConfig(), "track.bragg=3")


def test_the_rejection_lists_what_was_available():
    with pytest.raises(SystemExit, match="available:"):
        apply_override(SourceConfig(), "wobble=3")


def test_a_spec_without_an_equals_sign_is_rejected():
    with pytest.raises(SystemExit, match="no '=' found"):
        apply_override(SourceConfig(), "render.height")


def test_an_empty_field_path_is_rejected():
    with pytest.raises(SystemExit, match="empty"):
        apply_override(SourceConfig(), "=3")


def test_descending_into_a_leaf_is_rejected():
    """`render.height` is an int, so `render.height.nope` has nowhere to go."""
    with pytest.raises(SystemExit, match="not a config block"):
        apply_override(SourceConfig(), "render.height.nope=3")


def test_a_bad_override_in_a_sequence_stops_the_run():
    with pytest.raises(SystemExit):
        apply_overrides(SourceConfig(), ["seed=1", "nope=2"])


# --------------------------------------------------------------------------- #
# the training config
# --------------------------------------------------------------------------- #


def test_the_training_config_nests_the_four_blocks():
    cfg = TrainConfig()

    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.optim, OptimConfig)
    assert isinstance(cfg.run, RunConfig)


def test_min_steps_is_off_by_default():
    """0 disables the floor, so an existing config keeps its old behaviour."""
    assert OptimConfig().min_steps == 0


def test_a_training_yaml_loads_into_nested_dataclasses(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("optim:\n  epochs: 3\n  min_steps: 500\nrun:\n  seed: 9\n")
    cfg = load_train_yaml(path)

    assert isinstance(cfg.optim, OptimConfig)
    assert cfg.optim.epochs == 3
    assert cfg.optim.min_steps == 500
    assert cfg.run.seed == 9
    assert cfg.model == ModelConfig()          # untouched blocks keep defaults


def test_an_unknown_training_key_is_rejected(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("optim:\n  momentum: 0.9\n")

    with pytest.raises(KeyError, match="momentum"):
        load_train_yaml(path)


def test_an_empty_training_yaml_gives_the_defaults(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("")

    assert load_train_yaml(path) == TrainConfig()


def test_training_configs_can_be_overridden_too():
    """`scripts/train.py --set optim.epochs=8` goes through the same path."""
    cfg = apply_overrides(TrainConfig(), ["optim.epochs=8", "optim.min_steps=600",
                                          "data.root=data/hard"])

    assert cfg.optim.epochs == 8
    assert cfg.optim.min_steps == 600
    assert cfg.data.root == "data/hard"


@pytest.mark.parametrize("name", ["base", "smoke"])
def test_the_shipped_training_configs_load(name):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_train_yaml(root / "configs" / "train" / f"{name}.yaml")

    assert cfg.optim.epochs > 0
    assert cfg.data.batch_size > 0
