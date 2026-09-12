"""Classical baselines and input normalisation."""
import numpy as np
import pytest

from patt_reco.config import EMPTY, TRACK, load_yaml
from patt_reco.dataset.generate import generate_event
from patt_reco.dataset.preprocess import noise_sigma, normalise
from patt_reco.models.baselines import BASELINES, hough_lines


def test_normalise_puts_noise_at_unit_sigma():
    rng = np.random.default_rng(0)
    adc = rng.normal(100.0, 7.0, (3, 128, 128)).astype(np.float32)
    out = normalise(adc)
    assert abs(float(out.std()) - 1.0) < 0.05
    assert abs(float(out.mean())) < 0.05          # median subtracted


def test_normalise_is_gain_invariant():
    """Gain is randomised per event, so the model must not see it."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 5.0, (2, 64, 64)).astype(np.float32)
    base[0, 32, 20:40] += 200.0
    assert np.allclose(normalise(base), normalise(base * 3.0), atol=0.02)


def test_noise_sigma_has_a_floor_for_noise_free_images():
    clean = np.zeros((2, 32, 32), dtype=np.float32)
    assert np.all(noise_sigma(clean, floor=1.0) == 1.0)
    assert np.isfinite(normalise(clean)).all()


def test_normalise_ignores_the_signal_pixels():
    """A MAD estimate must not be dragged up by the few percent that are signal."""
    rng = np.random.default_rng(2)
    adc = rng.normal(0.0, 4.0, (1, 128, 128)).astype(np.float32)
    adc[0, :4, :] += 500.0                        # ~3% of pixels, very bright
    assert abs(float(noise_sigma(adc).ravel()[0]) - 4.0) < 0.4


def test_hough_finds_a_planted_line():
    mask = np.zeros((64, 64), dtype=bool)
    rows = np.arange(5, 60)
    mask[rows, (0.5 * rows + 10).astype(int)] = True
    found = hough_lines(mask, min_votes=10)
    # most of the planted line is recovered, and nothing outside it is invented
    assert found[mask].mean() > 0.8
    assert not found[~mask].any()


def test_hough_on_an_empty_image_returns_nothing():
    assert not hough_lines(np.zeros((32, 32), dtype=bool)).any()


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_return_a_wellformed_prediction(name):
    cfg = load_yaml("configs/data/l2_noise.yaml")
    record = generate_event(cfg, 0)
    prediction = BASELINES[name]().predict(record.adc(cfg.noise))

    assert prediction.semantic.shape == record.shape
    assert prediction.instance.shape == record.shape
    assert prediction.semantic.dtype == np.uint8
    # the two outputs must agree about which pixels are occupied
    assert np.array_equal(prediction.semantic != EMPTY, prediction.instance >= 0)


def test_threshold_baseline_responds_to_its_threshold():
    cfg = load_yaml("configs/data/l2_noise.yaml")
    adc = generate_event(cfg, 0).adc(cfg.noise)
    loose = BASELINES["threshold"](n_sigma=2.0).predict(adc).foreground.sum()
    tight = BASELINES["threshold"](n_sigma=6.0).predict(adc).foreground.sum()
    assert loose > tight > 0


def test_connected_components_separates_disjoint_blobs():
    from patt_reco.models.baselines import ConnectedComponents
    adc = np.zeros((1, 64, 64), dtype=np.float32)
    adc[0, 10:14, 10:14] = 100.0
    adc[0, 40:44, 40:44] = 100.0
    prediction = ConnectedComponents(n_sigma=3.0).predict(adc)
    assert len(np.unique(prediction.instance[prediction.instance >= 0])) == 2


def test_baselines_beat_nothing_but_are_not_perfect_on_l2():
    """The floor must be a real floor: better than chance, far from solved.

    If this ever starts passing with a near-perfect IoU, the generator has become
    too easy and needs harder crossings or lower SNR before any network work.
    """
    from patt_reco.eval.metrics_sem import SemanticMetrics
    cfg = load_yaml("configs/data/l2_noise.yaml")
    metrics = SemanticMetrics()
    for i in range(10):
        record = generate_event(cfg, i)
        prediction = BASELINES["threshold"]().predict(record.adc(cfg.noise))
        metrics.update(prediction.semantic, record.dense_semantic(),
                       record.dense_n_contrib())
    iou = metrics.foreground_summary("exclusive")["iou"]
    assert 0.15 < iou < 0.85, f"foreground IoU {iou:.3f} outside the plausible band"
