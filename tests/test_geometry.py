"""Geometry-layer invariants."""
import numpy as np
import pytest

from patt_reco.config import GeometryConfig, EventConfig, TRACK, COSMIC
from patt_reco.geometry.compose import compose_event
from patt_reco.geometry.primitives import (PRIMITIVES, Helix, LineTrack, Ring,
                                           orthonormal_basis, rotate_towards, unit)
from patt_reco.geometry.volume import Volume

GCFG = GeometryConfig()
VOL = Volume(0, 40, -15, 15, -15, 15)


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_every_primitive_deposits_positive_charge(name):
    prim = PRIMITIVES[name].sample(np.random.default_rng(0), GCFG, VOL)
    points, dq = prim.deposit(np.random.default_rng(1), GCFG)
    assert points.ndim == 2 and points.shape[1] == 3
    assert points.shape[0] == dq.shape[0] > 0
    assert np.all(dq >= 0) and dq.sum() > 0
    assert np.isfinite(points).all()


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_deposit_is_reproducible(name):
    prim = PRIMITIVES[name].sample(np.random.default_rng(3), GCFG, VOL)
    a = prim.deposit(np.random.default_rng(7), GCFG)
    b = prim.deposit(np.random.default_rng(7), GCFG)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_line_track_is_straight_and_has_the_requested_length():
    track = LineTrack(origin=np.zeros(3), direction=unit(np.array([1.0, 2.0, 3.0])),
                      length=20.0, stopping=False)
    points, _ = track.deposit(np.random.default_rng(0), GCFG)
    # every point lies on the line: the cross product with the direction vanishes
    offsets = points - track.anchor
    residual = np.linalg.norm(np.cross(offsets, track.direction), axis=1)
    assert residual.max() < 1e-9
    assert np.linalg.norm(points[-1] - points[0]) == pytest.approx(20.0, rel=1e-2)


def test_bragg_peak_puts_the_charge_at_the_end():
    common = dict(origin=np.zeros(3), direction=np.array([1.0, 0.0, 0.0]), length=20.0)
    stopper = LineTrack(**common, stopping=True).deposit(np.random.default_rng(0), GCFG)
    flat = LineTrack(**common, stopping=False).deposit(np.random.default_rng(0), GCFG)

    n = len(stopper[1])
    last_tenth = slice(int(0.9 * n), None)
    assert stopper[1][last_tenth].mean() > 2.0 * flat[1][last_tenth].mean()
    assert stopper[1].sum() > flat[1].sum()


def test_scattered_track_bends_more_at_low_momentum():
    def deflection(momentum):
        prim = PRIMITIVES["scattered"].sample(
            np.random.default_rng(0), GCFG, VOL,
            origin=np.zeros(3), direction=np.array([1.0, 0.0, 0.0]),
            length=50.0, momentum=momentum)
        points, _ = prim.deposit(np.random.default_rng(5), GCFG)
        end = unit(points[-1] - points[0])
        return float(np.arccos(np.clip(end @ prim.direction, -1, 1)))

    assert deflection(100.0) > deflection(2000.0)


def test_helix_points_stay_on_a_circle_about_its_axis():
    helix = Helix(origin=np.array([10.0, 0.0, 0.0]), direction=np.array([0.0, 1.0, 0.0]),
                  axis=np.array([0.0, 0.0, 1.0]), radius=12.0, pitch=0.0, arc=3.0, sense=1.0)
    points, _ = helix.deposit(np.random.default_rng(0), GCFG)
    centre, _, _, axis = helix._frame()
    offsets = points - centre
    radial = np.linalg.norm(offsets - (offsets @ axis)[:, None] * axis, axis=1)
    assert np.allclose(radial, 12.0, atol=1e-6)
    # pitch = 0 means the trajectory stays in one plane
    assert np.allclose(offsets @ axis, 0.0, atol=1e-6)


def test_helix_starts_at_its_origin():
    helix = PRIMITIVES["helix"].sample(np.random.default_rng(2), GCFG, VOL)
    points, _ = helix.deposit(np.random.default_rng(0), GCFG)
    assert np.allclose(points[0], helix.anchor, atol=1e-6)
    assert np.allclose(helix.point_at(0.0), helix.anchor, atol=1e-6)


def test_ring_radius_matches_and_is_planar_to_within_its_thickness():
    ring = Ring(centre=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]), radius=10.0,
                thickness=0.3, arc_frac=1.0, phi0=0.0)
    points, _ = ring.deposit(np.random.default_rng(0), GCFG)
    radial = np.linalg.norm(points[:, :2], axis=1)
    assert abs(radial.mean() - 10.0) < 0.1
    assert abs(points[:, 2]).max() < 6 * 0.3


def test_shower_conserves_energy_to_within_the_truncation():
    prim = PRIMITIVES["shower"].sample(np.random.default_rng(0), GCFG, VOL, energy=500.0)
    _, dq = prim.deposit(np.random.default_rng(1), GCFG)
    ceiling = 500.0 * GCFG.shower_ke_per_mev
    # Moyal's mean sits above its mode, so the sampled total can exceed the
    # nominal yield by ~30%; it must not exceed it by more than that.
    assert 0.3 * ceiling < dq.sum() < 1.4 * ceiling


def test_orthonormal_basis_is_orthonormal():
    for seed in range(20):
        d = unit(np.random.default_rng(seed).normal(size=3))
        u, v = orthonormal_basis(d)
        assert abs(u @ v) < 1e-12 and abs(u @ d) < 1e-12 and abs(v @ d) < 1e-12
        assert np.linalg.norm(u) == pytest.approx(1.0)
        assert np.linalg.norm(v) == pytest.approx(1.0)


@pytest.mark.parametrize("theta", [0.05, 0.5, 1.2])
def test_rotate_towards_produces_the_requested_angle(theta):
    d = unit(np.array([1.0, 1.0, 0.0]))
    out = rotate_towards(d, theta, 0.7)
    assert float(np.arccos(np.clip(out @ d, -1, 1))) == pytest.approx(theta, abs=1e-9)


def test_volume_shrink_is_concentric():
    vol = Volume(0, 40, -15, 15, -15, 15)
    small = vol.shrink(0.5)
    assert np.allclose(0.5 * (small.lo + small.hi), 0.5 * (vol.lo + vol.hi))
    assert np.allclose(small.span, 0.5 * vol.span)


def test_compose_assigns_contiguous_ids_and_tracks_parents():
    ecfg = EventConfig(n_objects=(3, 6), n_cosmics=(1, 2), p_delta=1.0, p_vertex=1.0)
    for seed in range(10):
        event = compose_event(np.random.default_rng(seed), GCFG, ecfg, VOL)
        ids = [o.obj_id for o in event.objects]
        assert ids == list(range(len(ids)))
        for obj in event.objects:
            assert -1 <= obj.parent_id < obj.obj_id
        assert sum(o.cls == COSMIC for o in event.objects) == event.meta["n_cosmics"]


def test_vertex_group_members_share_an_origin():
    ecfg = EventConfig(n_objects=(4, 4), p_vertex=1.0, vertex_size=(3, 3),
                       p_delta=0.0, p_crossing=0.0, n_cosmics=(0, 0))
    event = compose_event(np.random.default_rng(0), GCFG, ecfg, VOL)
    anchors = [o.primitive.anchor for o in event.objects if o.parent_id == 0 or o.obj_id == 0]
    assert len(anchors) >= 2
    for anchor in anchors[1:]:
        assert np.allclose(anchor, anchors[0])


def test_forced_crossing_brings_two_objects_close_together():
    ecfg = EventConfig(n_objects=(2, 2), p_crossing=1.0, p_vertex=0.0, p_delta=0.0,
                       n_cosmics=(0, 0), crossing_angle_deg=(20.0, 20.0),
                       class_weights={"track": 1.0})
    hits = 0
    for seed in range(20):
        event = compose_event(np.random.default_rng(seed), GCFG, ecfg, VOL)
        a, b = event.objects[0].points, event.objects[1].points
        # closest approach between the two trajectories
        gap = np.min(np.linalg.norm(a[::20, None, :] - b[None, ::20, :], axis=-1))
        hits += gap < 1.0
    assert hits == 20


def test_cosmics_traverse_the_whole_volume():
    ecfg = EventConfig(n_objects=(1, 1), n_cosmics=(1, 1), p_delta=0.0, p_vertex=0.0,
                       p_crossing=0.0)
    event = compose_event(np.random.default_rng(0), GCFG, ecfg, VOL)
    cosmic = next(o for o in event.objects if o.cls == COSMIC)
    inside = VOL.contains(cosmic.points)
    assert inside.sum() > 0
    # it enters and leaves: deposits exist on both sides of the fiducial volume
    assert (~inside).sum() > 0
