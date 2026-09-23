"""Geometry layer: the vector helpers, the box, the two primitives and the
rasteriser.

The tests are written against what each piece is *specified* to produce -- an
arc of a stated radius, a constant-valued track, a cascade that reaches a
stated distance -- rather than against whatever the current code happens to
emit, so a silent change of model shows up as a failure.
"""
import numpy as np
import pytest

from patt_reco.config import RenderConfig, SHOWER, TRACK
from patt_reco.geometry import PRIMITIVES, Shower, Track, Volume
from patt_reco.geometry.base import (orthonormal_basis, random_direction,
                                     rotate_towards, unit)
from patt_reco.geometry.project import project

# --------------------------------------------------------------------------- #
# vector helpers
# --------------------------------------------------------------------------- #


def test_unit_returns_length_one_vectors(rng):
    for _ in range(20):
        v = rng.normal(size=3) * rng.uniform(0.01, 100.0)
        assert np.linalg.norm(unit(v)) == pytest.approx(1.0)


def test_unit_does_not_divide_by_zero():
    """A zero vector has no direction; the helper must still return a number."""
    assert np.all(np.isfinite(unit(np.zeros(3))))


@pytest.mark.parametrize("d", [
    [0.0, 0.0, 1.0],    # the helper-vector branch boundary
    [1.0, 0.0, 0.0],    # the other branch: |d[0]| >= 0.9
    [0.3, -0.5, 0.8],
])
def test_orthonormal_basis_is_orthonormal_and_perpendicular(d):
    d = unit(np.asarray(d, dtype=float))
    u, v = orthonormal_basis(d)

    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert u @ v == pytest.approx(0.0, abs=1e-12)
    assert u @ d == pytest.approx(0.0, abs=1e-12)
    assert v @ d == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.1, 0.7, 1.5])
def test_rotate_towards_tilts_by_exactly_the_requested_angle(theta, rng):
    d = unit(rng.normal(size=3))
    tilted = rotate_towards(d, theta, azimuth=rng.uniform(0, 2 * np.pi))

    assert np.linalg.norm(tilted) == pytest.approx(1.0)
    assert np.arccos(np.clip(tilted @ d, -1.0, 1.0)) == pytest.approx(theta, abs=1e-9)


def test_random_direction_is_a_unit_vector_and_covers_the_sphere(rng):
    dirs = np.stack([random_direction(rng) for _ in range(2000)])

    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    # isotropic: the mean of a uniform sample on the sphere is near the origin
    assert np.linalg.norm(dirs.mean(axis=0)) < 0.1
    # and no hemisphere is starved
    assert (dirs[:, 2] > 0).sum() == pytest.approx(1000, abs=150)


# --------------------------------------------------------------------------- #
# volume
# --------------------------------------------------------------------------- #


def test_cube_is_centred_on_the_origin_with_the_requested_side():
    v = Volume.cube(4.0)

    assert np.allclose(v.lo, [-2.0, -2.0, -2.0])
    assert np.allclose(v.hi, [2.0, 2.0, 2.0])
    assert np.allclose(v.span, 4.0)
    assert v.diagonal == pytest.approx(4.0 * np.sqrt(3.0))


def test_shrink_is_concentric_and_scales_the_span():
    v = Volume(-1.0, 3.0, 0.0, 10.0, -5.0, -1.0)
    small = v.shrink(0.5)

    mid, mid_small = 0.5 * (v.lo + v.hi), 0.5 * (small.lo + small.hi)
    assert np.allclose(mid_small, mid)
    assert np.allclose(small.span, 0.5 * v.span)


def test_contains_agrees_with_the_bounds():
    v = Volume.cube(2.0)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.01, 0.0, 0.0],
                       [0.0, -5.0, 0.0]])

    assert np.array_equal(v.contains(points), [True, True, False, False])


def test_sampled_points_land_inside_the_box(rng):
    v = Volume(-1.0, 3.0, 0.0, 10.0, -5.0, -1.0)
    assert np.all(v.contains(v.sample_point(rng, 500)))


# --------------------------------------------------------------------------- #
# track
# --------------------------------------------------------------------------- #


def test_sampled_track_respects_the_config_ranges(rng, track_cfg, vol):
    for _ in range(50):
        t = Track.sample(rng, track_cfg, vol)

        assert t.cls == TRACK
        assert vol.contains(t.anchor)
        assert np.linalg.norm(t.direction) == pytest.approx(1.0)
        assert track_cfg.length[0] <= t.params["length"] <= track_cfg.length[1]
        assert track_cfg.curvature[0] <= t.params["curvature"] <= track_cfg.curvature[1]

        # the bend axis defines the plane the arc turns in, so it has to be a
        # unit vector perpendicular to the direction of travel
        bend = t.params["bend_axis"]
        assert np.linalg.norm(bend) == pytest.approx(1.0)
        assert bend @ t.direction == pytest.approx(0.0, abs=1e-9)


def test_sample_honours_explicit_overrides(rng, track_cfg, vol):
    t = Track.sample(rng, track_cfg, vol, origin=[1.0, 2.0, 3.0],
                     direction=[0.0, 0.0, 2.0], length=0.75, curvature=0.0,
                     bend_axis=[1.0, 0.0, 0.0])

    assert np.allclose(t.anchor, [1.0, 2.0, 3.0])
    assert np.allclose(t.direction, [0.0, 0.0, 1.0])     # normalised on the way in
    assert t.params["length"] == 0.75


def test_a_zero_curvature_track_is_a_straight_segment(rng, track_cfg, vol):
    t = Track.sample(rng, track_cfg, vol, curvature=0.0, length=1.0)
    points, _ = t.deposit(rng, track_cfg)

    offsets = points - points[0]
    # every offset is parallel to the direction: no transverse component at all
    transverse = offsets - (offsets @ t.direction)[:, None] * t.direction
    assert np.max(np.abs(transverse)) == pytest.approx(0.0, abs=1e-12)


def test_track_starts_at_its_origin_and_leaves_along_its_direction(rng, track_cfg, vol):
    t = Track.sample(rng, track_cfg, vol, length=1.0, curvature=1.5)
    points, _ = t.deposit(rng, track_cfg)

    assert np.allclose(points[0], t.anchor)
    # the first step is tangent to `direction`
    tangent = unit(points[1] - points[0])
    assert tangent @ unit(t.direction) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("curvature", [0.5, 1.5, 3.0])
def test_a_curved_track_lies_on_a_circle_of_radius_one_over_curvature(
        curvature, rng, track_cfg, vol):
    t = Track.sample(rng, track_cfg, vol, length=1.0, curvature=curvature)
    points, _ = t.deposit(rng, track_cfg)

    centre = t.anchor + (1.0 / curvature) * t.params["bend_axis"]
    radii = np.linalg.norm(points - centre, axis=1)
    assert np.allclose(radii, 1.0 / curvature, atol=1e-9)


def test_a_curved_track_stays_in_its_bend_plane(rng, track_cfg, vol):
    t = Track.sample(rng, track_cfg, vol, length=1.0, curvature=2.0)
    points, _ = t.deposit(rng, track_cfg)

    normal = np.cross(unit(t.direction), t.params["bend_axis"])
    assert np.max(np.abs((points - t.anchor) @ normal)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("curvature", [0.0, 2.0])
def test_points_are_one_step_apart_in_arc_length(curvature, rng, track_cfg, vol):
    t = Track.sample(rng, track_cfg, vol, length=1.0, curvature=curvature)
    points, _ = t.deposit(rng, track_cfg)

    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # chord ~ step for a gentle arc; the chord is shorter than the arc, hence
    # the one-sided tolerance
    assert np.all(gaps <= track_cfg.step + 1e-12)
    assert np.all(gaps > 0.99 * track_cfg.step)


def test_deposited_arc_length_matches_the_requested_length(rng, track_cfg, vol):
    t = Track.sample(rng, track_cfg, vol, length=0.83, curvature=1.0)
    points, _ = t.deposit(rng, track_cfg)

    arc = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    # the walk stops on the last whole step, so it can fall short by one
    assert 0.83 - track_cfg.step <= arc <= 0.83


def test_track_values_are_exactly_constant(rng, track_cfg, vol):
    """A track must be recognisable by shape alone.

    Any variation along the line -- a Bragg peak, a Landau draw, a stopping
    flag -- would let a classifier key on intensity instead of geometry, so
    the constancy is exact, not approximate.
    """
    for _ in range(20):
        t = Track.sample(rng, track_cfg, vol)
        _, values = t.deposit(rng, track_cfg)

        assert np.all(values == track_cfg.value)


def test_deposit_is_reproducible_for_the_same_seed(track_cfg, vol):
    def once():
        rng = np.random.default_rng(7)
        t = Track.sample(rng, track_cfg, vol)
        return t.deposit(rng, track_cfg)

    a_points, a_values = once()
    b_points, b_values = once()
    assert np.array_equal(a_points, b_points)
    assert np.array_equal(a_values, b_values)


# --------------------------------------------------------------------------- #
# shower
# --------------------------------------------------------------------------- #


def test_sampled_shower_starts_inside_the_volume(rng, shower_cfg, vol):
    for _ in range(50):
        s = Shower.sample(rng, shower_cfg, vol)

        assert s.cls == SHOWER
        assert vol.contains(s.anchor)
        assert np.linalg.norm(s.direction) == pytest.approx(1.0)


def test_shower_values_vary_per_point_and_stay_in_range(rng, shower_cfg, vol):
    s = Shower.sample(rng, shower_cfg, vol)
    _, values = s.deposit(rng, shower_cfg)

    assert np.all(values >= shower_cfg.value[0])
    assert np.all(values <= shower_cfg.value[1])
    assert values.std() > 0.0          # drawn per point, unlike a track


def test_shower_growth_is_bounded_by_max_nodes(rng, vol, straight_shower_cfg):
    from dataclasses import replace

    one = replace(straight_shower_cfg, max_nodes=1)
    s = Shower.sample(rng, one, vol)
    points, values = s.deposit(rng, one)

    # exactly one segment of max(2, round(1.0 / 0.5)) = 2 points
    assert len(points) == 2
    assert len(values) == 2


def test_more_nodes_deposit_more_points(rng, vol, straight_shower_cfg):
    from dataclasses import replace

    counts = []
    for max_nodes in (1, 3, 7):
        cfg = replace(straight_shower_cfg, max_nodes=max_nodes)
        s = Shower.sample(np.random.default_rng(3), cfg, vol)
        points, _ = s.deposit(np.random.default_rng(4), cfg)
        counts.append(len(points))

    assert counts[0] < counts[1] < counts[2]


def test_branch_lengths_do_not_compound_with_depth(rng, vol, straight_shower_cfg):
    """Segment length is a one-shot scale, not a budget divided down the tree.

    With even splits and no opening angle the cascade is a single ray, so the
    distance it reaches is a direct readout of the length rule. Breadth-first
    growth with `max_nodes=7` completes three generations, and scales that
    average 1.0 put the far end at 3 x seg_len. Were the scale multiplied by
    the parent's instead, the ray would converge to roughly 2 x seg_len no
    matter how many nodes were allowed.
    """
    s = Shower.sample(rng, straight_shower_cfg, vol)
    points, _ = s.deposit(rng, straight_shower_cfg)

    reach = np.max((points - s.anchor) @ unit(s.direction))
    assert reach == pytest.approx(3.0, rel=0.02)


def test_deeper_cascades_reach_steadily_further(rng, vol, straight_shower_cfg):
    """The corollary: each extra generation adds a full segment, not a
    geometrically shrinking one."""
    from dataclasses import replace

    reaches = []
    for max_nodes in (1, 3, 7, 15):
        cfg = replace(straight_shower_cfg, max_nodes=max_nodes)
        s = Shower.sample(np.random.default_rng(5), cfg, vol)
        points, _ = s.deposit(np.random.default_rng(6), cfg)
        reaches.append(np.max((points - s.anchor) @ unit(s.direction)))

    assert reaches == pytest.approx([1.0, 2.0, 3.0, 4.0], rel=0.02)


def test_a_shower_spreads_transversely_but_a_track_does_not(rng, vol, shower_cfg,
                                                            track_cfg):
    """The one property the whole dataset rests on: the two classes differ by
    shape, so a shower must not be collapsible onto a line."""
    def transverse_fraction(points):
        centred = points - points.mean(axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        return sv[1] / sv[0]

    t = Track.sample(rng, track_cfg, vol, curvature=0.0)
    track_points, _ = t.deposit(rng, track_cfg)

    s = Shower.sample(rng, shower_cfg, vol)
    shower_points, _ = s.deposit(rng, shower_cfg)

    assert transverse_fraction(track_points) == pytest.approx(0.0, abs=1e-9)
    assert transverse_fraction(shower_points) > 0.05


def test_the_primitive_registry_maps_kinds_to_classes():
    assert PRIMITIVES == {"track": Track, "shower": Shower}


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #


def _ring(n=200, radius=1.0):
    """A flat circle in the x-y plane -- a shape with a known aspect ratio."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)], axis=1)


def test_projection_returns_a_normalised_float32_frame(render_cfg):
    points = _ring()
    image = project(points, np.ones(len(points)), render_cfg)

    assert image.shape == (render_cfg.height, render_cfg.width)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() == pytest.approx(1.0)


def test_an_empty_cloud_gives_an_empty_frame(render_cfg):
    image = project(np.zeros((0, 3)), np.zeros(0), render_cfg)

    assert image.shape == (render_cfg.height, render_cfg.width)
    assert image.dtype == np.float32
    assert not image.any()


def test_all_zero_values_do_not_divide_by_zero(render_cfg):
    points = _ring()
    image = project(points, np.zeros(len(points)), render_cfg)

    assert np.all(np.isfinite(image))
    assert not image.any()


def test_the_margin_is_kept_clear(render_cfg):
    cfg = RenderConfig(height=64, width=64, margin=0.25, view=(0.0, 0.0, 1.0))
    points = _ring()
    image = project(points, np.ones(len(points)), cfg)

    border = int(cfg.margin * min(cfg.height, cfg.width)) - 1
    assert not image[:border].any()
    assert not image[-border:].any()
    assert not image[:, :border].any()
    assert not image[:, -border:].any()


def test_pixel_values_accumulate_and_are_weighted(render_cfg):
    """Two points in one pixel sum; the weight is `values`, not the count."""
    # four coincident pairs at the corners of a square, so the fit is stable
    corners = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                        [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    points = np.repeat(corners, 2, axis=0)

    flat = project(points, np.ones(len(points)), render_cfg)
    # double the weight on one corner only: it becomes the unique peak
    weights = np.ones(len(points))
    weights[:2] = 3.0
    skewed = project(points, weights, render_cfg)

    assert (flat == 1.0).sum() == 4           # four equal corners, all at peak
    assert (skewed == 1.0).sum() == 1         # one corner now dominates


def test_the_object_is_fitted_to_the_frame_so_scale_does_not_matter(render_cfg):
    """The renderer normalises size, so a big and a small copy of the same
    shape produce the same image -- absolute scale is not a class cue."""
    small = project(_ring(radius=0.01), np.ones(200), render_cfg)
    large = project(_ring(radius=100.0), np.ones(200), render_cfg)

    assert np.array_equal(small, large)


def test_translating_the_cloud_does_not_move_the_image(render_cfg):
    points = _ring()
    here = project(points, np.ones(len(points)), render_cfg)
    there = project(points + np.array([7.0, -3.0, 2.0]), np.ones(len(points)),
                    render_cfg)

    assert np.array_equal(here, there)


def test_aspect_ratio_is_preserved(render_cfg):
    """A circle must not come out as an ellipse on a non-square frame."""
    cfg = RenderConfig(height=32, width=64, margin=0.0, view=(0.0, 0.0, 1.0))
    image = project(_ring(n=2000), np.ones(2000), cfg)

    rows, cols = np.nonzero(image)
    assert (rows.max() - rows.min()) == pytest.approx(cols.max() - cols.min(), abs=1)


def test_the_view_direction_selects_the_projection_plane(render_cfg):
    """A flat disc seen face-on fills the frame; seen edge-on it is a line."""
    points = _ring(n=2000)
    face_on = project(points, np.ones(2000),
                      RenderConfig(height=32, width=32, margin=0.1, view=(0.0, 0.0, 1.0)))
    edge_on = project(points, np.ones(2000),
                      RenderConfig(height=32, width=32, margin=0.1, view=(0.0, 1.0, 0.0)))

    assert np.count_nonzero(face_on) > 3 * np.count_nonzero(edge_on)
