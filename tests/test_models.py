"""The baseline CNN.

The architecture is pinned deliberately. `patt_reco-arf` specifies exactly
four layers and forbids batch normalisation, a second convolutional block,
residual connections and global average pooling -- the whole point of the
model is to be the floor that later architectures are measured against, so a
quiet improvement to it would invalidate every recorded comparison.
"""
import pytest

torch = pytest.importorskip("torch")
from torch import nn                                    # noqa: E402

from patt_reco.config import N_CLASSES                  # noqa: E402
from patt_reco.models import CNN                        # noqa: E402


@pytest.fixture
def model():
    return CNN(32, 32)


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #


def test_the_baseline_is_exactly_one_conv_one_pool_and_two_linears(model):
    kinds = [type(m) for m in model.modules() if not isinstance(m, CNN)]

    assert kinds.count(nn.Conv2d) == 1
    assert kinds.count(nn.MaxPool2d) == 1
    assert kinds.count(nn.Flatten) == 1
    assert kinds.count(nn.Linear) == 2
    assert kinds.count(nn.Dropout) == 1


@pytest.mark.parametrize("forbidden", [
    nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm,
    nn.AdaptiveAvgPool2d, nn.AvgPool2d,
])
def test_the_baseline_carries_no_extra_machinery(model, forbidden):
    """Anything added here silently raises the floor the benchmark is measured
    against, so the exclusions are asserted rather than assumed."""
    assert not any(isinstance(m, forbidden) for m in model.modules())


def test_the_defaults_match_the_specified_signature(model):
    assert model.conv.in_channels == 1
    assert model.conv.out_channels == 16
    assert model.conv.kernel_size == (3, 3)
    assert model.pool == 4
    assert model.dense.out_features == 64
    assert model.dropout.p == 0.0          # no hidden regularisation in the baseline


def test_the_head_emits_one_logit_per_class(model):
    """Logits for cross-entropy, not a single sigmoid output: a third shape
    class has to be a config change, not a rewrite."""
    assert model.head.out_features == N_CLASSES


def test_the_convolution_preserves_the_frame(model):
    """`padding="same"`, so the pool divisor is the only thing setting the
    flattened width."""
    x = torch.zeros(1, 1, 32, 32)
    assert model.relu(model.conv(x)).shape == (1, 16, 32, 32)


@pytest.mark.parametrize("height,width,pool,n_filters", [
    (32, 32, 4, 16),
    (64, 64, 8, 4),
    (128, 128, 4, 16),
    (30, 20, 4, 8),      # not divisible: floor division
])
def test_the_flattened_width_follows_the_pool_divisor(height, width, pool, n_filters):
    m = CNN(height, width, n_filters=n_filters, pool=pool)

    assert m.dense.in_features == n_filters * (height // pool) * (width // pool)


def test_a_pool_larger_than_the_image_fails_loudly():
    with pytest.raises(ValueError, match="pool"):
        CNN(8, 8, pool=16)


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("batch", [1, 5])
def test_forward_maps_a_batch_of_images_to_class_logits(model, batch):
    out = model(torch.randn(batch, 1, 32, 32))

    assert out.shape == (batch, N_CLASSES)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_a_mismatched_input_size_fails_loudly_rather_than_reshaping(model):
    with pytest.raises(ValueError, match=r"expected input of size 32x32.*\(64, 64\)"):
        model(torch.randn(2, 1, 64, 64))


def test_every_parameter_receives_a_gradient(model):
    """A parameter with no gradient is a disconnected graph -- the exact
    failure `overfit_one_batch` exists to catch, caught earlier and cheaper."""
    loss = model(torch.randn(4, 1, 32, 32)).sum()
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_images_in_a_batch_do_not_influence_each_other(model):
    """No batch normalisation, so one image's logits must not depend on what
    it was batched with."""
    model.eval()
    a = torch.randn(1, 1, 32, 32)
    alone = model(a)
    batched = model(torch.cat([a, torch.randn(3, 1, 32, 32)]))[:1]

    assert torch.allclose(alone, batched, atol=1e-6)


def test_dropout_is_inert_at_zero_and_active_when_asked():
    plain, dropped = CNN(32, 32, dropout=0.0), CNN(32, 32, dropout=0.9)
    x = torch.randn(8, 1, 32, 32)

    plain.train()
    assert torch.equal(plain(x), plain(x))

    dropped.train()
    assert not torch.equal(dropped(x), dropped(x))

    dropped.eval()                       # inference must be deterministic
    assert torch.equal(dropped(x), dropped(x))


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #


def test_the_model_is_not_lazy_so_a_checkpoint_loads_cold(model, tmp_path):
    """`nn.LazyLinear` would leave the dense weight unmaterialised until a
    warm-up forward pass, making a checkpoint unloadable. The shapes have to
    be known at construction."""
    assert not any(isinstance(m, nn.modules.lazy.LazyModuleMixin)
                   for m in model.modules())

    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), path)

    fresh = CNN(32, 32)                  # never run forward
    fresh.load_state_dict(torch.load(path, weights_only=True))

    fresh.eval(), model.eval()
    x = torch.randn(2, 1, 32, 32)
    assert torch.allclose(fresh(x), model(x))


def test_the_parameter_count_is_reported_and_dominated_by_the_dense_layer(model):
    total = model.n_parameters()

    assert total == sum(p.numel() for p in model.parameters())
    assert model.dense.weight.numel() > 0.5 * total
    assert f"{total:,}" in repr(model)


def test_the_pool_is_the_knob_that_controls_the_parameter_count():
    """Documented in the module docstring: the flattened width scales as
    1/pool**2, so doubling the pool quarters the dense layer."""
    coarse = CNN(64, 64, pool=8).dense.weight.numel()
    fine = CNN(64, 64, pool=4).dense.weight.numel()

    assert fine == 4 * coarse
