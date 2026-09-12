"""HDF5 shard I/O.

Events are ragged (different numbers of hit pixels and objects), so each field
is stored as one concatenated array plus a per-event offset table -- the usual
layout for variable-length scientific data, and one that reads a single event
without touching the rest of the file.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .schema import EventRecord

_RAGGED_EVENT = ("pixel", "charge", "instance", "n_contrib")
_RAGGED_CONTRIB = ("contrib_pixel", "contrib_obj", "contrib_charge")
_RAGGED_OBJECT = ("obj_id", "obj_cls", "obj_parent", "obj_params", "obj_charge",
                  "obj_ndom", "obj_ntouch")
_SCALAR_KEYS = ("pitch_cm", "tick_cm", "diffusion_t", "diffusion_l", "attenuation_cm",
                "gain", "pedestal", "saturation", "bipolar", "quantise")
_META_KEYS = ("occupancy", "n_primary", "n_cosmics", "has_vertex")


def _append(group: h5py.Group, name: str, array: np.ndarray) -> None:
    array = np.asarray(array)
    if name not in group:
        group.create_dataset(
            name, shape=(0,) + array.shape[1:], maxshape=(None,) + array.shape[1:],
            dtype=array.dtype, chunks=True, compression="gzip", compression_opts=4)
    if array.shape[0] == 0:
        return          # dataset now exists but has nothing to add
    dset = group[name]
    start = dset.shape[0]
    dset.resize(start + array.shape[0], axis=0)
    dset[start:] = array


class ShardWriter:
    """Buffered writer.

    Appending one event at a time costs ~10 ms because every one of the ~25
    datasets pays HDF5 chunk overhead per call -- twice the cost of generating
    the event. Buffering `batch` events and writing one concatenated block per
    dataset removes that; the buffer is a few MB.
    """

    def __init__(self, path, shape: tuple[int, int, int], view_angles: np.ndarray,
                 batch: int = 256):
        self.path = Path(path)
        self.file = h5py.File(self.path, "w")
        self.file.attrs["shape"] = np.asarray(shape, dtype=np.int32)
        self.file.attrs["view_angles"] = np.asarray(view_angles, dtype=np.float32)
        self.groups = {name: self.file.create_group(name)
                       for name in ("events", "contrib", "objects", "readout", "meta")}
        self.offsets = {"events": [0], "contrib": [0], "objects": [0], "kernel": [0]}
        self.n_events = 0
        self.batch = max(1, int(batch))
        self._buffer: dict[tuple[str, str], list[np.ndarray]] = {}
        self._buffered = 0

    def _stage(self, group: str, name: str, array: np.ndarray) -> None:
        self._buffer.setdefault((group, name), []).append(np.asarray(array))

    def flush(self) -> None:
        for (group, name), chunks in self._buffer.items():
            _append(self.groups[group], name, np.concatenate(chunks))
        self._buffer.clear()
        self._buffered = 0

    def append(self, rec: EventRecord) -> None:
        for name in _RAGGED_EVENT:
            self._stage("events", name, getattr(rec, name))
        for name in _RAGGED_CONTRIB:
            self._stage("contrib", name.removeprefix("contrib_"), getattr(rec, name))
        for name in _RAGGED_OBJECT:
            self._stage("objects", name.removeprefix("obj_"), getattr(rec, name))

        self._stage("readout", "kernel", rec.kernel)
        self._stage("readout", "channel_gain", rec.channel_gain[None])
        self._stage("readout", "dead_mask", rec.dead_mask[None])
        for key in _SCALAR_KEYS:
            self._stage("readout", key, np.array([rec.scalars[key]], dtype=np.float64))

        self._stage("meta", "index", np.array([rec.index], dtype=np.int64))
        self._stage("meta", "seed", np.array([rec.seed], dtype=np.int64))
        for key in _META_KEYS:
            self._stage("meta", key, np.array([float(rec.meta.get(key, 0))], dtype=np.float64))

        self.offsets["events"].append(self.offsets["events"][-1] + len(rec.pixel))
        self.offsets["contrib"].append(self.offsets["contrib"][-1] + len(rec.contrib_pixel))
        self.offsets["objects"].append(self.offsets["objects"][-1] + rec.n_objects)
        self.offsets["kernel"].append(self.offsets["kernel"][-1] + len(rec.kernel))
        self.n_events += 1

        self._buffered += 1
        if self._buffered >= self.batch:
            self.flush()

    def close(self) -> None:
        self.flush()
        for name, values in self.offsets.items():
            self.file.create_dataset(f"offsets/{name}", data=np.array(values, dtype=np.int64))
        self.file.attrs["n_events"] = self.n_events
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ShardReader:
    """Random access to a shard. Only the requested event is read from disk."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        self.shape = tuple(int(x) for x in self.file.attrs["shape"])
        self.view_angles = np.asarray(self.file.attrs["view_angles"])
        self.n_events = int(self.file.attrs["n_events"])
        self._off = {k: self.file[f"offsets/{k}"][:] for k in
                     ("events", "contrib", "objects", "kernel")}

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, i: int) -> EventRecord:
        if i < 0:
            i += self.n_events
        if not 0 <= i < self.n_events:
            raise IndexError(i)
        f = self.file
        ev0, ev1 = self._off["events"][i], self._off["events"][i + 1]
        ct0, ct1 = self._off["contrib"][i], self._off["contrib"][i + 1]
        ob0, ob1 = self._off["objects"][i], self._off["objects"][i + 1]
        kn0, kn1 = self._off["kernel"][i], self._off["kernel"][i + 1]

        return EventRecord(
            index=int(f["meta/index"][i]),
            seed=int(f["meta/seed"][i]),
            shape=self.shape,
            pixel=f["events/pixel"][ev0:ev1],
            charge=f["events/charge"][ev0:ev1],
            instance=f["events/instance"][ev0:ev1],
            n_contrib=f["events/n_contrib"][ev0:ev1],
            contrib_pixel=f["contrib/pixel"][ct0:ct1],
            contrib_obj=f["contrib/obj"][ct0:ct1],
            contrib_charge=f["contrib/charge"][ct0:ct1],
            obj_id=f["objects/id"][ob0:ob1],
            obj_cls=f["objects/cls"][ob0:ob1],
            obj_parent=f["objects/parent"][ob0:ob1],
            obj_params=f["objects/params"][ob0:ob1],
            obj_charge=f["objects/charge"][ob0:ob1],
            obj_ndom=f["objects/ndom"][ob0:ob1],
            obj_ntouch=f["objects/ntouch"][ob0:ob1],
            kernel=f["readout/kernel"][kn0:kn1],
            channel_gain=f["readout/channel_gain"][i],
            dead_mask=f["readout/dead_mask"][i],
            view_angles=self.view_angles,
            scalars={k: float(f[f"readout/{k}"][i]) for k in _SCALAR_KEYS},
            meta={k: float(f[f"meta/{k}"][i]) for k in _META_KEYS},
        )

    def close(self) -> None:
        self.file.close()


class DatasetReader:
    """A whole dataset directory: every shard, indexed end to end."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.shards = [ShardReader(self.root / name) for name in self.manifest["shards"]]
        self.bounds = np.cumsum([0] + [len(s) for s in self.shards])

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def __getitem__(self, i: int) -> EventRecord:
        if i < 0:
            i += len(self)
        shard = int(np.searchsorted(self.bounds, i, side="right") - 1)
        return self.shards[shard][i - self.bounds[shard]]

    def close(self) -> None:
        for shard in self.shards:
            shard.close()
