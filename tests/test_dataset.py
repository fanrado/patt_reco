"""Dataset-level invariants: determinism, storage roundtrip, label derivation."""
import numpy as np
import pytest

from patt_reco.config import DatasetConfig, DetectorConfig, EMPTY, EventConfig, NoiseConfig
from patt_reco.dataset.generate import generate_dataset, generate_event
from patt_reco.dataset.io_hdf5 import DatasetReader
from patt_reco.detector.response import response_kernel


def test_the_same_seed_and_index_give_a_bit_identical_event(small_cfg):
    a = generate_event(small_cfg, 3)
    b = generate_event(small_cfg, 3)
    assert np.array_equal(a.pixel, b.pixel)
    assert np.array_equal(a.charge, b.charge)
    assert np.array_equal(a.instance, b.instance)
    assert np.allclose(a.obj_params, b.obj_params, equal_nan=True)


def test_different_indices_give_different_events(small_cfg):
    a, b = generate_event(small_cfg, 3), generate_event(small_cfg, 4)
    assert not (len(a.pixel) == len(b.pixel) and np.array_equal(a.pixel, b.pixel))


def test_changing_the_dataset_seed_changes_the_event(small_cfg):
    from dataclasses import replace
    other = replace(small_cfg, seed=small_cfg.seed + 1)
    a, b = generate_event(small_cfg, 0), generate_event(other, 0)
    assert not (len(a.pixel) == len(b.pixel) and np.array_equal(a.pixel, b.pixel))


def test_stored_pixels_are_exactly_the_pixels_with_charge(small_cfg):
    rec = generate_event(small_cfg, 1)
    dense = rec.dense_charge()
    assert np.array_equal(np.flatnonzero(dense.reshape(-1)), rec.pixel)
    assert np.all(rec.charge > 0)


def test_semantic_is_derived_consistently_from_instance(small_cfg):
    rec = generate_event(small_cfg, 2)
    semantic, instance = rec.dense_semantic(), rec.dense_instance()
    assert np.array_equal(semantic != EMPTY, instance >= 0)
    lookup = dict(zip(rec.obj_id.tolist(), rec.obj_cls.tolist()))
    filled = instance >= 0
    assert np.array_equal(semantic[filled],
                          np.array([lookup[i] for i in instance[filled]], dtype=np.uint8))


def test_object_table_pixel_counts_match_the_instance_image(small_cfg):
    rec = generate_event(small_cfg, 5)
    instance = rec.dense_instance()
    for k, obj_id in enumerate(rec.obj_id):
        for v in range(rec.shape[0]):
            assert int((instance[v] == obj_id).sum()) == int(rec.obj_ndom[k, v])


def test_adc_is_deterministic_and_noise_is_additive(small_cfg):
    rec = generate_event(small_cfg, 0)
    clean_a, clean_b = rec.adc(), rec.adc()
    assert np.array_equal(clean_a, clean_b)

    noisy_a = rec.adc(noise_cfg=small_cfg.noise)
    noisy_b = rec.adc(noise_cfg=small_cfg.noise)
    assert np.array_equal(noisy_a, noisy_b)          # derived seed -> frozen
    assert not np.array_equal(noisy_a, clean_a)

    fresh = rec.adc(noise_cfg=small_cfg.noise, noise_seed=999)
    assert not np.array_equal(fresh, noisy_a)        # explicit seed -> augmentation


def test_noise_scale_zero_matches_the_clean_image(small_cfg):
    rec = generate_event(small_cfg, 0)
    assert np.array_equal(rec.adc(noise_cfg=small_cfg.noise, noise_scale=0.0), rec.adc())


def test_a_different_response_kernel_changes_only_the_adc(small_cfg):
    """The domain-shift test relies on re-rendering without touching truth."""
    rec = generate_event(small_cfg, 0)
    other = response_kernel(4.0, bipolar=not bool(rec.scalars["bipolar"]))
    shifted = rec.adc(kernel=other)
    assert shifted.shape == rec.adc().shape
    assert not np.allclose(shifted, rec.adc())
    assert np.array_equal(rec.dense_semantic(), rec.dense_semantic())


def test_shard_roundtrip_is_lossless(tmp_path, small_cfg):
    from dataclasses import replace
    cfg = replace(small_cfg, n_events=6, shard_size=4)
    generate_dataset(cfg, tmp_path / "ds", verbose=False)
    reader = DatasetReader(tmp_path / "ds")
    assert len(reader) == 6

    for i in (0, 3, 5):
        stored, fresh = reader[i], generate_event(cfg, i)
        assert stored.index == fresh.index and stored.seed == fresh.seed
        assert stored.shape == fresh.shape
        for field in ("pixel", "charge", "instance", "n_contrib", "obj_id", "obj_cls",
                      "obj_parent", "obj_charge", "obj_ndom", "obj_ntouch",
                      "contrib_pixel", "contrib_obj", "contrib_charge"):
            assert np.array_equal(getattr(stored, field), getattr(fresh, field)), field
        assert np.allclose(stored.obj_params, fresh.obj_params, equal_nan=True)
        assert np.allclose(stored.kernel, fresh.kernel)
        assert np.array_equal(stored.dead_mask, fresh.dead_mask)
        assert np.allclose(stored.adc(cfg.noise), fresh.adc(cfg.noise))
    reader.close()


def test_manifest_records_what_was_generated(tmp_path, small_cfg):
    import json
    from dataclasses import replace
    cfg = replace(small_cfg, n_events=4, shard_size=4)
    generate_dataset(cfg, tmp_path / "ds", verbose=False)
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert manifest["n_events"] == 4
    assert manifest["config"]["seed"] == cfg.seed
    assert manifest["shards"] == ["shard_0000.h5"]
    assert "config_hash" in manifest and "git_sha" in manifest


def test_parallel_generation_matches_serial(tmp_path, small_cfg):
    from dataclasses import replace
    cfg = replace(small_cfg, n_events=8, shard_size=2)
    generate_dataset(cfg, tmp_path / "serial", workers=1, verbose=False)
    generate_dataset(cfg, tmp_path / "parallel", workers=4, verbose=False)
    a, b = DatasetReader(tmp_path / "serial"), DatasetReader(tmp_path / "parallel")
    assert len(a) == len(b) == 8
    for i in range(8):
        assert np.array_equal(a[i].pixel, b[i].pixel)
        assert np.array_equal(a[i].charge, b[i].charge)
    a.close(); b.close()


def test_config_yaml_roundtrip(tmp_path):
    import yaml
    from patt_reco.config import config_hash, load_yaml, to_dict
    cfg = DatasetConfig(name="rt", n_events=3, event=EventConfig(n_objects=(2, 4)))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(to_dict(cfg)))
    assert config_hash(load_yaml(path)) == config_hash(cfg)


def test_buffered_writes_survive_flush_boundaries(tmp_path, small_cfg):
    """Events must round-trip identically across a buffer flush, not just within one."""
    from dataclasses import replace
    from patt_reco.dataset.io_hdf5 import ShardReader, ShardWriter

    cfg = replace(small_cfg, n_events=7)
    records = [generate_event(cfg, i) for i in range(7)]
    shape = (len(cfg.detector.view_angles_deg), cfg.detector.n_channels, cfg.detector.n_ticks)

    path = tmp_path / "tiny_batch.h5"
    with ShardWriter(path, shape, np.asarray(cfg.detector.view_angles_deg), batch=2) as w:
        for rec in records:
            w.append(rec)

    reader = ShardReader(path)
    assert len(reader) == 7
    for i, fresh in enumerate(records):
        stored = reader[i]
        assert np.array_equal(stored.pixel, fresh.pixel), i
        assert np.array_equal(stored.charge, fresh.charge), i
        assert np.array_equal(stored.instance, fresh.instance), i
        assert np.array_equal(stored.obj_id, fresh.obj_id), i
        assert np.allclose(stored.kernel, fresh.kernel), i
    reader.close()


def test_an_event_with_no_overlapping_pixels_still_round_trips(tmp_path, small_cfg):
    """The contrib table is empty for isolated events; the reader must not care."""
    from dataclasses import replace
    from patt_reco.config import EventConfig
    from patt_reco.dataset.io_hdf5 import DatasetReader

    cfg = replace(small_cfg, n_events=3, shard_size=3,
                  event=EventConfig(n_objects=(1, 1), p_vertex=0.0, p_delta=0.0,
                                    p_crossing=0.0, n_cosmics=(0, 0)))
    generate_dataset(cfg, tmp_path / "solo", verbose=False)
    reader = DatasetReader(tmp_path / "solo")
    for i in range(3):
        rec = reader[i]
        assert len(rec.contrib_pixel) == 0
        assert np.array_equal(rec.pixel, generate_event(cfg, i).pixel)
    reader.close()
