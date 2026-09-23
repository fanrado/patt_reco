"""Config loading, event generation, dataset building and the torch wrapper.

The generator's central promise is that `(seed, index)` alone determines an
image. Most of what follows is that promise, checked from several directions:
the same pair reproduces, different pairs diverge, and the split layout never
lets two splits draw the same pair.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from patt_reco import build_dataset, generate_event, load_yaml
from patt_reco.config import (CLASS_NAMES, N_CLASSES, RenderConfig, SHOWER,
                              ShowerConfig, SourceConfig, TRACK, TrackConfig,
                              config_hash, to_dict)
from patt_reco.dataset.build import SPLITS
from patt_reco.dataset.generate import event_rng

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "data"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_the_two_classes_are_distinct_and_named():
    assert TRACK != SHOWER
    assert N_CLASSES == 2
    assert set(CLASS_NAMES) == {TRACK, SHOWER}


@pytest.mark.parametrize("name,kind,label", [
    ("tracks.yaml", "track", TRACK),
    ("showers.yaml", "shower", SHOWER),
])
def test_the_shipped_configs_load_into_nested_dataclasses(name, kind, label):
    cfg = load_yaml(CONFIG_DIR / name)

    assert isinstance(cfg, SourceConfig)
    assert cfg.kind == kind
    assert cfg.label == label
    # the nested blocks must be real config objects, not leftover dicts
    assert isinstance(cfg.track, TrackConfig)
    assert isinstance(cfg.shower, ShowerConfig)
    assert isinstance(cfg.render, RenderConfig)


def test_the_shipped_configs_can_be_built_together():
    """tracks.yaml and showers.yaml are separate files with no shared parent,
    so nothing but this agreement keeps them combinable."""
    tracks = load_yaml(CONFIG_DIR / "tracks.yaml")
    showers = load_yaml(CONFIG_DIR / "showers.yaml")

    assert tracks.label != showers.label
    assert (tracks.render.height, tracks.render.width) == \
           (showers.render.height, showers.render.width)


def test_yaml_lists_become_tuples_so_configs_stay_hashable(tmp_path):
    path = tmp_path / "src.yaml"
    path.write_text("name: t\nkind: track\ntrack:\n  length: [1.0, 2.0]\n")
    cfg = load_yaml(path)

    assert cfg.track.length == (1.0, 2.0)
    hash(cfg.track)          # frozen dataclasses of tuples hash cleanly


def test_an_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    path = tmp_path / "src.yaml"
    path.write_text("name: t\nkind: track\nwobble: 3\n")

    with pytest.raises(KeyError, match="wobble"):
        load_yaml(path)


def test_an_unknown_nested_key_is_rejected(tmp_path):
    path = tmp_path / "src.yaml"
    path.write_text("name: t\nkind: track\ntrack:\n  bragg: true\n")

    with pytest.raises(KeyError, match="bragg"):
        load_yaml(path)


def test_an_empty_config_file_gives_the_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")

    assert load_yaml(path) == SourceConfig()


def test_config_hash_is_stable_and_sensitive(track_source):
    assert config_hash(track_source) == config_hash(replace(track_source))
    assert config_hash(track_source) != config_hash(replace(track_source, seed=999))
    assert len(config_hash(track_source)) == 12


def test_to_dict_is_plain_json_serialisable_data(track_source):
    blob = to_dict(track_source)

    assert isinstance(blob["render"], dict)
    json.dumps(blob, default=str)


# --------------------------------------------------------------------------- #
# event generation
# --------------------------------------------------------------------------- #


def test_the_same_seed_and_index_give_a_bit_identical_image(track_source):
    a, la = generate_event(track_source, 3)
    b, lb = generate_event(track_source, 3)

    assert np.array_equal(a, b)
    assert la == lb


def test_different_indices_give_different_images(track_source):
    a, _ = generate_event(track_source, 0)
    b, _ = generate_event(track_source, 1)

    assert not np.array_equal(a, b)


def test_changing_the_seed_changes_the_image(track_source):
    a, _ = generate_event(track_source, 0)
    b, _ = generate_event(replace(track_source, seed=track_source.seed + 1), 0)

    assert not np.array_equal(a, b)


def test_the_seed_and_index_are_not_interchangeable():
    """(seed, index) must be ordered; if they were pooled, these would match."""
    assert not np.array_equal(event_rng(1, 2).random(8), event_rng(2, 1).random(8))


@pytest.mark.parametrize("source", ["track_source", "shower_source"])
def test_a_generated_image_matches_the_render_config(source, request):
    cfg = request.getfixturevalue(source)
    image, label = generate_event(cfg, 0)

    assert image.shape == (cfg.render.height, cfg.render.width)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() == pytest.approx(1.0)
    assert label == cfg.label


def test_an_unknown_kind_fails_loudly(track_source):
    with pytest.raises(ValueError, match="unknown kind"):
        generate_event(replace(track_source, kind="helix"), 0)


def test_the_label_follows_the_config_not_the_kind(track_source):
    """`label` and `kind` are independent fields, so the label is whatever the
    config says -- that is what lets a source be relabelled without touching
    the generator."""
    _, label = generate_event(replace(track_source, label=7), 0)

    assert label == 7


def test_tracks_and_showers_look_different(track_source, shower_source):
    """Averaged over events, a shower lights a visibly larger share of the
    frame than a single line does."""
    def lit_fraction(cfg):
        return np.mean([np.count_nonzero(generate_event(cfg, i)[0])
                        for i in range(12)])

    assert lit_fraction(shower_source) > 1.5 * lit_fraction(track_source)


# --------------------------------------------------------------------------- #
# dataset building
# --------------------------------------------------------------------------- #


@pytest.fixture
def built(tmp_path, track_source, shower_source):
    out = build_dataset([track_source, shower_source], tmp_path / "ds",
                        shuffle_seed=5)
    return out, [track_source, shower_source]


def test_build_writes_every_split_and_a_manifest(built):
    out, _ = built

    for split in SPLITS:
        assert (out / f"{split}.npz").is_file()
    assert (out / "meta.json").is_file()


def test_every_split_holds_both_classes_in_the_expected_numbers(built):
    out, sources = built

    for split in SPLITS:
        with np.load(out / f"{split}.npz") as data:
            images, labels = data["images"], data["labels"]

        expected = sum(getattr(cfg, f"n_{split}") for cfg in sources)
        assert len(images) == len(labels) == expected
        assert images.dtype == np.float32
        assert labels.dtype == np.uint8
        assert np.bincount(labels, minlength=2).tolist() == \
               [getattr(sources[0], f"n_{split}"), getattr(sources[1], f"n_{split}")]


def test_stored_images_have_the_configured_frame_size(built):
    out, sources = built
    render = sources[0].render

    with np.load(out / "train.npz") as data:
        assert data["images"].shape[1:] == (render.height, render.width)


def test_no_two_splits_share_an_image(built):
    """Each split draws from its own index range, so the same object never
    appears in both training and evaluation."""
    out, _ = built

    seen = {}
    for split in SPLITS:
        with np.load(out / f"{split}.npz") as data:
            seen[split] = {img.tobytes() for img in data["images"]}

    assert not seen["train"] & seen["val"]
    assert not seen["train"] & seen["test"]
    assert not seen["val"] & seen["test"]


def test_the_train_split_is_exactly_the_first_index_range(built):
    """Pins the offset arithmetic: train must be indices 0..n_train-1."""
    out, sources = built
    with np.load(out / "train.npz") as data:
        stored = {img.tobytes() for img in data["images"]}

    expected = {generate_event(cfg, i)[0].tobytes()
                for cfg in sources for i in range(cfg.n_train)}
    assert stored == expected


def test_labels_still_match_their_images_after_shuffling(built):
    out, sources = built
    by_label = {cfg.label: cfg for cfg in sources}

    with np.load(out / "train.npz") as data:
        images, labels = data["images"], data["labels"]

    # rebuild the truth: which images each source produces
    truth = {cfg.label: {generate_event(cfg, i)[0].tobytes()
                         for i in range(cfg.n_train)} for cfg in sources}
    for image, label in zip(images, labels):
        assert image.tobytes() in truth[int(label)]


def test_building_twice_gives_identical_files(tmp_path, track_source, shower_source):
    a = build_dataset([track_source, shower_source], tmp_path / "a", shuffle_seed=5)
    b = build_dataset([track_source, shower_source], tmp_path / "b", shuffle_seed=5)

    with np.load(a / "train.npz") as x, np.load(b / "train.npz") as y:
        assert np.array_equal(x["images"], y["images"])
        assert np.array_equal(x["labels"], y["labels"])


def test_the_shuffle_seed_changes_the_order_but_not_the_contents(
        tmp_path, track_source, shower_source):
    a = build_dataset([track_source, shower_source], tmp_path / "a", shuffle_seed=1)
    b = build_dataset([track_source, shower_source], tmp_path / "b", shuffle_seed=2)

    with np.load(a / "train.npz") as x, np.load(b / "train.npz") as y:
        assert not np.array_equal(x["images"], y["images"])
        assert {i.tobytes() for i in x["images"]} == {i.tobytes() for i in y["images"]}


def test_the_manifest_records_what_was_generated(built):
    out, sources = built
    meta = json.loads((out / "meta.json").read_text())

    assert [s["name"] for s in meta["sources"]] == [c.name for c in sources]
    assert meta["shuffle_seed"] == 5
    assert meta["counts"]["train"] == sum(c.n_train for c in sources)
    assert meta["class_names"] == {str(k): v for k, v in CLASS_NAMES.items()}


def test_sources_that_disagree_on_image_size_are_rejected(
        tmp_path, track_source, shower_source):
    odd = replace(shower_source, render=replace(shower_source.render, height=8))

    with pytest.raises(ValueError, match="image size"):
        build_dataset([track_source, odd], tmp_path / "ds")


def test_two_sources_claiming_the_same_label_are_rejected(
        tmp_path, track_source, shower_source):
    clash = replace(shower_source, label=track_source.label)

    with pytest.raises(ValueError, match="label"):
        build_dataset([track_source, clash], tmp_path / "ds")


def test_building_from_no_sources_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no sources"):
        build_dataset([], tmp_path / "ds")


def test_a_source_with_no_events_still_writes_wellformed_splits(
        tmp_path, track_source):
    empty = replace(track_source, n_train=0, n_val=0, n_test=0)
    out = build_dataset([empty], tmp_path / "ds")

    with np.load(out / "train.npz") as data:
        assert data["images"].shape == (0, track_source.render.height,
                                        track_source.render.width)
        assert data["images"].dtype == np.float32
        assert len(data["labels"]) == 0


# --------------------------------------------------------------------------- #
# torch wrapper
# --------------------------------------------------------------------------- #


def test_the_dataset_exposes_every_stored_image(built):
    torch = pytest.importorskip("torch")
    from patt_reco.dataset.torch_dataset import ImageDataset

    out, _ = built
    ds = ImageDataset(out / "train.npz")

    with np.load(out / "train.npz") as data:
        assert len(ds) == len(data["images"])

        image, label = ds[0]
        assert image.shape == (1, *data["images"].shape[1:])   # channel first
        assert image.dtype == torch.float32
        assert isinstance(label, int)
        assert np.array_equal(image.numpy()[0], data["images"][0])
        assert label == int(data["labels"][0])


def test_the_dataset_batches_cleanly(built):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from patt_reco.dataset.torch_dataset import ImageDataset

    out, sources = built
    ds = ImageDataset(out / "train.npz")
    images, labels = next(iter(DataLoader(ds, batch_size=4)))

    assert images.shape == (4, 1, sources[0].render.height, sources[0].render.width)
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64
    assert set(labels.tolist()) <= {TRACK, SHOWER}


def test_the_dataset_reads_the_file_once_and_releases_it(built, tmp_path):
    """The split is loaded eagerly, so the npz must not stay open behind it."""
    pytest.importorskip("torch")
    from patt_reco.dataset.torch_dataset import ImageDataset

    out, _ = built
    copied = tmp_path / "copy.npz"
    copied.write_bytes((out / "val.npz").read_bytes())

    ds = ImageDataset(copied)
    copied.unlink()               # would fail on Windows if a handle survived

    assert len(ds) > 0
    assert ds[0][0] is not None
