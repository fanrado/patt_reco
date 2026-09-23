"""`build_model`, and the depth knob it exists to deliver.

PLAN.md section 8 names a recurring defect class: a configured value that
never reaches what it configures, invisible in the output. `model.n_blocks`
was the third instance -- `deep.yaml` was committed and inert, and a "deep"
run would have built the baseline and reported that depth changed nothing.

`build_model` is the structural fix, so these tests target the guarantee
rather than the one instance: any ModelConfig field either reaches the model
or stops the run. The test that matters most invents a knob that does not
exist yet.
"""
from dataclasses import dataclass, fields

import pytest

torch = pytest.importorskip("torch")

import inspect                                          # noqa: E402

from torch import nn                                    # noqa: E402

from patt_reco.config import N_CLASSES                  # noqa: E402
from patt_reco.models import CNN, build_model           # noqa: E402
from patt_reco.train.config import ModelConfig          # noqa: E402


# --------------------------------------------------------------------------- #
# the guarantee
# --------------------------------------------------------------------------- #


def test_every_model_config_field_reaches_the_model():
    """The defect class, closed at the source: a field the model cannot accept
    is a silent no-op, so the two signatures must agree."""
    accepted = set(inspect.signature(CNN.__init__).parameters)
    for f in fields(ModelConfig()):
        assert f.name in accepted, (
            f"ModelConfig.{f.name} has nowhere to go in CNN.__init__")


def test_build_model_passes_every_field_through():
    cfg = ModelConfig(n_filters=8, kernel_size=5, pool=2, hidden=32,
                      dropout=0.25, n_blocks=2)
    model = build_model(cfg, 32, 32)

    assert model.conv.out_channels == 8
    assert model.conv.kernel_size == (5, 5)
    assert model.pool == 2
    assert model.dense.out_features == 32
    assert model.dropout.p == 0.25
    assert model.n_blocks == 2


def test_n_blocks_actually_reaches_the_model():
    """The exact defect: `deep.yaml` set n_blocks=3 and the script built a
    1-block baseline anyway."""
    shallow = build_model(ModelConfig(n_blocks=1), 64, 64)
    deep = build_model(ModelConfig(n_blocks=3), 64, 64)

    assert shallow.n_blocks == 1
    assert deep.n_blocks == 3
    assert len(deep.extra_convs) == 2
    assert shallow.n_parameters() != deep.n_parameters()


def test_a_future_knob_that_cannot_reach_the_model_stops_the_run():
    """The guarantee generalised. This is what makes the fix structural: a
    knob nobody has written yet still cannot be silently ignored."""
    @dataclass(frozen=True)
    class FutureModelConfig:
        n_filters: int = 16
        residual: bool = True        # CNN.__init__ has no such parameter

    with pytest.raises(TypeError, match="residual"):
        build_model(FutureModelConfig(), 32, 32)


def test_the_rejection_explains_the_two_ways_out():
    @dataclass(frozen=True)
    class FutureModelConfig:
        attention_heads: int = 4

    with pytest.raises(TypeError, match="silently ignored"):
        build_model(FutureModelConfig(), 32, 32)


def test_height_and_width_are_the_callers_business_not_the_configs():
    """They come from the data, not from ModelConfig, so they must not be
    mistaken for an unmappable field."""
    model = build_model(ModelConfig(), 48, 80)

    assert (model.height, model.width) == (48, 80)


def test_neither_script_hand_enumerates_model_fields():
    """PLAN.md: 'Neither script now names a config field.' If one starts
    again, the next knob added can go missing exactly as n_blocks did."""
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    names = [f.name for f in fields(ModelConfig())]

    for script in ("train.py", "evaluate.py"):
        source = (scripts / script).read_text()
        body = source[source.index("build_model"):]
        for name in names:
            assert f"{name}=" not in body, f"{script} names {name} by hand"


# --------------------------------------------------------------------------- #
# the baseline is unchanged
# --------------------------------------------------------------------------- #


def test_the_default_baseline_still_has_its_published_parameter_count():
    """patt_reco-1dj pins this: every baseline number already recorded in
    PLAN.md is only valid if the default architecture did not move."""
    assert CNN(128, 128).n_parameters() == 1_048_930


def test_one_block_is_still_exactly_one_conv_one_pool_and_two_linears():
    model = CNN(32, 32, n_blocks=1)
    kinds = [type(m) for m in model.modules() if not isinstance(m, CNN)]

    assert kinds.count(nn.Conv2d) == 1
    assert kinds.count(nn.MaxPool2d) == 1
    assert kinds.count(nn.Linear) == 2
    assert len(model.extra_convs) == 0


def test_a_one_block_checkpoint_written_before_the_depth_knob_still_loads():
    """The first conv keeps its old attribute name on purpose, so the
    state_dict is unchanged at n_blocks=1."""
    model = CNN(32, 32)
    keys = set(model.state_dict())

    assert keys == {"conv.weight", "conv.bias", "dense.weight", "dense.bias",
                    "head.weight", "head.bias"}


def test_depth_adds_no_normalisation_or_residual_machinery():
    """One variable at a time: the measured gain must be attributable to
    depth alone."""
    model = CNN(64, 64, n_blocks=3)

    for forbidden in (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm,
                      nn.AdaptiveAvgPool2d, nn.AvgPool2d):
        assert not any(isinstance(m, forbidden) for m in model.modules())
    assert sum(isinstance(m, nn.Dropout) for m in model.modules()) == 1


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_blocks", [1, 2, 3])
def test_each_block_adds_a_convolution_and_doubles_the_channels(n_blocks):
    model = CNN(64, 64, n_filters=4, pool=2, n_blocks=n_blocks)
    convs = [model.conv, *model.extra_convs]

    assert len(convs) == n_blocks
    for i, conv in enumerate(convs):
        assert conv.out_channels == 4 * 2 ** i
        assert conv.in_channels == (1 if i == 0 else 4 * 2 ** (i - 1))


@pytest.mark.parametrize("n_blocks,pool,height", [
    (1, 4, 128), (2, 4, 128), (3, 4, 128), (2, 2, 64), (3, 2, 32),
])
def test_the_flattened_width_follows_the_pooling_of_every_block(n_blocks, pool, height):
    model = CNN(height, height, n_filters=4, pool=pool, n_blocks=n_blocks)

    pooled = height
    for _ in range(n_blocks):
        pooled //= pool
    channels = 4 * 2 ** (n_blocks - 1)
    assert model.dense.in_features == channels * pooled * pooled


@pytest.mark.parametrize("n_blocks", [1, 2, 3])
def test_forward_works_at_every_depth(n_blocks):
    model = CNN(64, 64, n_filters=4, pool=2, n_blocks=n_blocks)
    out = model(torch.randn(3, 1, 64, 64))

    assert out.shape == (3, N_CLASSES)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n_blocks", [2, 3])
def test_every_parameter_still_receives_a_gradient_at_depth(n_blocks):
    model = CNN(64, 64, n_filters=4, pool=2, n_blocks=n_blocks)
    model(torch.randn(2, 1, 64, 64)).sum().backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_depth_shrinks_the_model_because_pooling_beats_doubling():
    """PLAN.md reports the deep arm at 26x FEWER parameters, and reads that as
    evidence the gain is receptive field rather than capacity. The direction
    is part of the claim."""
    shallow = CNN(128, 128, n_blocks=1).n_parameters()
    deep = CNN(128, 128, n_blocks=3).n_parameters()

    assert deep < shallow / 10


@pytest.mark.parametrize("n_blocks", [0, -1])
def test_fewer_than_one_block_fails_loudly(n_blocks):
    with pytest.raises(ValueError, match="at least 1"):
        CNN(32, 32, n_blocks=n_blocks)


def test_pooling_away_the_whole_image_fails_loudly():
    """4 blocks of pool=4 on 32x32 leaves nothing; the error names both knobs
    so the caller knows which to turn."""
    with pytest.raises(ValueError, match="n_blocks or pool"):
        CNN(32, 32, pool=4, n_blocks=4)


def test_the_repr_reports_the_depth_and_the_parameter_count():
    assert "3 blocks" in repr(CNN(64, 64, n_blocks=3))
    assert "1 block," in repr(CNN(64, 64, n_blocks=1))
    assert f"{CNN(64, 64).n_parameters():,}" in repr(CNN(64, 64))
