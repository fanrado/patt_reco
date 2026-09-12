"""Detector-layer invariants: projection, response, rendering, labels."""
import numpy as np
import pytest

from patt_reco.config import (DetectorConfig, EMPTY, EventConfig, GeometryConfig,
                              NoiseConfig)
from patt_reco.detector.digitize import digitise
from patt_reco.detector.projection import render_event
from patt_reco.detector.readout import PlaneView, sample_readout
from patt_reco.detector.response import apply_response, response_kernel
from patt_reco.geometry.compose import compose_event

GCFG = GeometryConfig()


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #

def test_tick_is_identical_in_every_view():
    """All views share the drift coordinate -- that is what makes stereo work."""
    det = DetectorConfig()
    readout = sample_readout(np.random.default_rng(0), det)
    points = np.random.default_rng(1).uniform(-10, 10, (50, 3))
    ticks = [view.project(points)[1] for view in readout.views]
    for other in ticks[1:]:
        assert np.allclose(ticks[0], other)


def test_projection_matches_the_analytic_formula():
    view = PlaneView(angle_deg=60.0, pitch_cm=0.4, tick_cm=0.5, n_channels=128, n_ticks=128)
    points = np.array([[5.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    channel, tick = view.project(points)
    expected_w = 2.0 * np.cos(np.deg2rad(60)) + 3.0 * np.sin(np.deg2rad(60))
    assert channel[0] == pytest.approx(expected_w / 0.4 + 64.0)
    assert tick[0] == pytest.approx(10.0)
    assert channel[1] == pytest.approx(64.0)   # the origin sits mid-plane


def test_a_line_along_y_is_seen_at_the_expected_relative_slope():
    """A view at 60 degrees sees cos(60) = 1/2 of a displacement along y."""
    kwargs = dict(pitch_cm=0.4, tick_cm=0.4, n_channels=128, n_ticks=128)
    straight = PlaneView(angle_deg=0.0, **kwargs)
    stereo = PlaneView(angle_deg=60.0, **kwargs)
    points = np.stack([np.full(10, 5.0), np.linspace(0, 10, 10), np.zeros(10)], axis=1)
    slope_0 = np.polyfit(points[:, 1], straight.project(points)[0], 1)[0]
    slope_60 = np.polyfit(points[:, 1], stereo.project(points)[0], 1)[0]
    assert slope_60 / slope_0 == pytest.approx(np.cos(np.deg2rad(60)))


def test_volume_is_visible_in_every_view():
    readout = sample_readout(np.random.default_rng(2), DetectorConfig())
    volume = readout.volume
    corners = np.array(np.meshgrid(*zip(volume.lo, volume.hi))).reshape(3, -1).T
    for view in readout.views:
        channel, tick = view.project(corners)
        assert channel.min() >= -1e-9
        assert channel.max() <= view.n_channels + 1e-9
        assert tick.min() >= -1e-9 and tick.max() <= view.n_ticks + 1e-9


# --------------------------------------------------------------------------- #
# response
# --------------------------------------------------------------------------- #

def test_unipolar_response_conserves_charge():
    kernel = response_kernel(2.0, bipolar=False)
    assert kernel.sum() == pytest.approx(1.0)
    image = np.zeros((2, 64))
    image[0, 30] = 100.0
    assert apply_response(image, kernel).sum() == pytest.approx(100.0, rel=1e-6)


def test_bipolar_response_has_a_negative_lobe_and_a_unit_positive_lobe():
    kernel = response_kernel(2.0, bipolar=True, asym=0.7)
    assert kernel.min() < 0 < kernel.max()
    assert kernel[kernel > 0].sum() == pytest.approx(1.0)
    # asymmetric lobes: the negative side is scaled down by `asym`
    assert abs(kernel.min()) == pytest.approx(0.7 * kernel.max(), rel=1e-6)


def test_response_preserves_image_shape_and_only_mixes_along_ticks():
    kernel = response_kernel(3.0, bipolar=True)
    image = np.zeros((4, 32))
    image[2, 16] = 50.0
    out = apply_response(image, kernel)
    assert out.shape == image.shape
    assert np.allclose(out[[0, 1, 3]], 0.0)    # no leakage between channels
    assert not np.allclose(out[2], 0.0)


def test_response_is_centred():
    """A delta in must give a response centred on the same tick."""
    image = np.zeros((1, 41))
    image[0, 20] = 1.0
    out = apply_response(image, response_kernel(2.0, bipolar=False))
    assert int(np.argmax(out[0])) == 20


# --------------------------------------------------------------------------- #
# rendering and labels
# --------------------------------------------------------------------------- #

@pytest.fixture
def rendered():
    det = DetectorConfig(n_channels=64, n_ticks=64)
    ecfg = EventConfig(n_objects=(3, 6), n_cosmics=(0, 1))
    rng = np.random.default_rng(42)
    readout = sample_readout(rng, det)
    event = compose_event(rng, GCFG, ecfg, readout.volume)
    return render_event(rng, event, readout, det), readout


def test_labels_cover_exactly_the_pixels_with_charge(rendered):
    ev, _ = rendered
    filled = ev.charge > 0
    assert np.array_equal(ev.instance >= 0, filled)
    assert np.array_equal(ev.semantic != EMPTY, filled)
    assert np.array_equal(ev.n_contrib > 0, filled)


def test_semantic_label_agrees_with_the_object_table(rendered):
    ev, _ = rendered
    cls_of = {o.obj_id: o.cls for o in ev.objects}
    filled = ev.instance >= 0
    expected = np.vectorize(cls_of.get)(ev.instance[filled])
    assert np.array_equal(ev.semantic[filled], expected)


def test_instance_ids_exist_in_the_object_table(rendered):
    ev, _ = rendered
    known = {o.obj_id for o in ev.objects}
    assert set(np.unique(ev.instance[ev.instance >= 0]).tolist()) <= known


def test_dominant_pixel_counts_match_the_instance_image(rendered):
    ev, _ = rendered
    for v in range(ev.n_views):
        for obj in ev.objects:
            assert int((ev.instance[v] == obj.obj_id).sum()) == int(obj.n_pixels_dominant[v])


def test_contributions_are_stored_exactly_for_ambiguous_pixels(rendered):
    ev, _ = rendered
    n_ch, n_tk = ev.charge.shape[1:]
    ambiguous = ev.n_contrib > 1
    if not ambiguous.any():
        pytest.skip("no overlapping pixels in this event")

    # one contribution row per (ambiguous pixel, contributing object)
    assert len(ev.contrib["pixel"]) == int(ev.n_contrib[ambiguous].sum())

    # and those contributions sum back to the total charge in the pixel
    flat_index = (ev.contrib["view"].astype(np.int64) * n_ch * n_tk
                  + ev.contrib["pixel"].astype(np.int64))
    summed = np.bincount(flat_index, weights=ev.contrib["charge"],
                         minlength=ev.charge.size)
    assert np.allclose(summed[ambiguous.reshape(-1)],
                       ev.charge.reshape(-1)[ambiguous.reshape(-1)], rtol=1e-5)


def test_the_dominant_contributor_really_is_the_largest(rendered):
    ev, _ = rendered
    n_ch, n_tk = ev.charge.shape[1:]
    if len(ev.contrib["pixel"]) == 0:
        pytest.skip("no overlapping pixels in this event")
    for v, pixel, obj_id, q in zip(ev.contrib["view"], ev.contrib["pixel"],
                                   ev.contrib["obj_id"], ev.contrib["charge"]):
        winner = ev.instance[v].reshape(-1)[pixel]
        if winner != obj_id:
            same = ((ev.contrib["view"] == v) & (ev.contrib["pixel"] == pixel)
                    & (ev.contrib["obj_id"] == winner))
            assert ev.contrib["charge"][same][0] >= q


def test_charge_is_conserved_for_a_fully_contained_object():
    """Rendering must not create or destroy charge, only attenuate it."""
    from patt_reco.geometry.compose import Event3D, DepositedObject
    from patt_reco.geometry.primitives import LineTrack

    det = DetectorConfig(n_channels=128, n_ticks=128, frac_dead_channels=(0.0, 0.0))
    rng = np.random.default_rng(3)
    readout = sample_readout(rng, det)
    volume = readout.volume
    centre = 0.5 * (volume.lo + volume.hi)

    track = LineTrack(origin=centre - np.array([0.0, 3.0, 0.0]),
                      direction=np.array([0.0, 1.0, 0.0]), length=6.0, stopping=False)
    points, dq = track.deposit(np.random.default_rng(9), GCFG)
    event = Event3D(objects=[DepositedObject(0, track.CLASS, points, dq, track)],
                    volume=volume)

    ev = render_event(np.random.default_rng(11), event, readout, det)
    expected = float((dq * np.exp(-points[:, 0] / readout.attenuation_cm)).sum())
    for v in range(ev.n_views):
        assert ev.charge[v].sum() == pytest.approx(expected, rel=1e-5)


def test_dead_channels_are_silent_in_the_adc(rendered):
    ev, readout = rendered
    if not readout.dead_mask.any():
        pytest.skip("no dead channels sampled")
    for v in range(ev.n_views):
        assert np.all(ev.adc[v][readout.dead_mask[v]] == 0.0)


def test_digitisation_saturates_and_quantises():
    det = DetectorConfig(n_channels=16, n_ticks=16)
    readout = sample_readout(np.random.default_rng(0), det)
    charge = np.zeros((len(readout.views), 16, 16), dtype=np.float32)
    charge[0, 8, 8] = 1e9
    adc = digitise(charge, readout)
    assert abs(adc).max() <= readout.saturation
    assert np.array_equal(adc, np.round(adc))
