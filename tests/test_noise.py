"""Noise-layer invariants."""
import numpy as np
import pytest

from patt_reco.config import NoiseConfig
from patt_reco.noise import realise_noise
from patt_reco.noise.coherent import coherent
from patt_reco.noise.incoherent import blips, pink, white

SHAPE = (2, 64, 128)


def test_noise_is_reproducible_from_its_seed():
    a = realise_noise(np.random.default_rng(7), SHAPE, NoiseConfig())
    b = realise_noise(np.random.default_rng(7), SHAPE, NoiseConfig())
    assert np.array_equal(a, b)


def test_different_seeds_give_different_realisations():
    a = realise_noise(np.random.default_rng(1), SHAPE, NoiseConfig())
    b = realise_noise(np.random.default_rng(2), SHAPE, NoiseConfig())
    assert not np.allclose(a, b)


@pytest.mark.parametrize("kwargs", [{"scale": 0.0}, {}])
def test_noise_can_be_switched_off(kwargs):
    cfg = NoiseConfig(enabled=False) if not kwargs else NoiseConfig()
    out = realise_noise(np.random.default_rng(0), SHAPE, cfg, **kwargs)
    assert np.all(out == 0.0)


def test_scale_is_linear_in_amplitude():
    """The SNR sweep depends on this: scale must rescale, not reshape."""
    cfg = NoiseConfig(p_blip=0.0)
    a = realise_noise(np.random.default_rng(4), SHAPE, cfg, scale=1.0)
    b = realise_noise(np.random.default_rng(4), SHAPE, cfg, scale=3.0)
    # the components are summed in float32, so pixels where they nearly cancel
    # carry a relative error of order 1e-4; that is accumulation, not nonlinearity
    assert np.allclose(b, 3.0 * a, rtol=1e-3, atol=1e-3 * float(np.abs(a).max()))


def test_dead_channels_stay_silent():
    dead = np.zeros((SHAPE[0], SHAPE[1]), dtype=bool)
    dead[0, ::4] = True
    out = realise_noise(np.random.default_rng(0), SHAPE, NoiseConfig(), dead_mask=dead)
    assert np.all(out[dead] == 0.0)


def test_white_noise_has_the_requested_sigma():
    out = white(np.random.default_rng(0), (256, 256), 2.5)
    assert out.std() == pytest.approx(2.5, rel=0.03)


def test_coherent_noise_is_shared_within_a_group_and_not_across():
    out = coherent(np.random.default_rng(0), (128, 256), sigma=3.0, group=32)
    within = np.corrcoef(out[0], out[5])[0, 1]
    across = np.corrcoef(out[0], out[64])[0, 1]
    assert within > 0.95
    assert abs(across) < 0.3


def test_pink_noise_has_more_power_at_low_frequency_than_white():
    n_ch, n_tk = 64, 512
    def low_fraction(series):
        power = np.abs(np.fft.rfft(series, axis=1)) ** 2
        return power[:, 1:16].sum() / power[:, 1:].sum()

    assert low_fraction(pink(np.random.default_rng(0), (n_ch, n_tk), 2.0)) > \
        3 * low_fraction(white(np.random.default_rng(0), (n_ch, n_tk), 2.0))


def test_pink_noise_has_no_dc_offset():
    out = pink(np.random.default_rng(0), (32, 256), 2.0)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-4)


def test_blips_are_local_and_positive():
    out = blips(np.random.default_rng(0), (64, 128), n_blips=5, amplitude=(10.0, 20.0))
    assert out.max() >= 10.0
    assert out.min() >= 0.0
    assert (out > 0).mean() < 0.05           # localised, not a wash
    assert np.count_nonzero(out.any(axis=1)) <= 5
