"""torch `Dataset` over a generated dataset directory.

Two things that are easy to get wrong and are handled here:

* **h5py handles do not survive a fork.** The reader is opened lazily and
  re-opened whenever the pid changes, so `num_workers > 0` works.
* **Noise is realised here, not at generation time.** With `freeze_noise` the
  seed is derived from the event, so a test set is reproducible to the bit;
  without it a fresh seed is drawn each epoch, which is the cheapest and most
  physically honest augmentation available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DatasetConfig, NoiseConfig, _build
from .augment import apply_geometric, mix_events
from .io_hdf5 import DatasetReader
from .preprocess import normalise


class PattRecoDataset(Dataset):
    """Yields `views`, `semantic`, `instance`, `n_contrib` for one event."""

    def __init__(self, root, noise_cfg: NoiseConfig | None = None,
                 freeze_noise: bool | None = None, noise_scale: float = 1.0,
                 augment: bool = False, mix_prob: float = 0.0,
                 max_tick_shift: int = 8, seed: int = 0, normalise_input: bool = True):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.config: DatasetConfig = _build(DatasetConfig, manifest["config"])
        self.n_events = int(manifest["n_events"])

        self.noise_cfg = self.config.noise if noise_cfg is None else noise_cfg
        self.freeze_noise = self.config.freeze_noise if freeze_noise is None else freeze_noise
        self.noise_scale = noise_scale
        self.augment = augment
        self.mix_prob = mix_prob
        self.max_tick_shift = max_tick_shift
        self.normalise_input = normalise_input
        self.seed = seed
        self.epoch = 0

        self._reader: DatasetReader | None = None
        self._pid: int | None = None

    # -- worker-safe reader -------------------------------------------------
    @property
    def reader(self) -> DatasetReader:
        if self._reader is None or self._pid != os.getpid():
            self._reader = DatasetReader(self.root)
            self._pid = os.getpid()
        return self._reader

    def set_epoch(self, epoch: int) -> None:
        """Makes the fresh-noise and mixing draws differ between epochs."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_events

    # -- item ---------------------------------------------------------------
    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.epoch, index])

    def __getitem__(self, index: int) -> dict:
        rng = self._rng(index)
        record = self.reader[index]

        if self.mix_prob > 0 and rng.random() < self.mix_prob:
            other = self.reader[int(rng.integers(0, self.n_events))]
            event = mix_events(record, other, self.noise_cfg, rng, self.noise_scale)
        else:
            noise_seed = None if self.freeze_noise else int(rng.integers(0, 2**31 - 1))
            event = _SingleEvent(record, self.noise_cfg, noise_seed, self.noise_scale)

        arrays = {
            "views": event.adc(),
            "semantic": event.dense_semantic(),
            "instance": event.dense_instance(),
            "n_contrib": event.dense_n_contrib(),
        }
        if self.augment:
            arrays = apply_geometric(arrays, rng, max_shift=self.max_tick_shift)

        views = arrays["views"]
        if self.normalise_input:
            views = normalise(views)

        return {
            "views": torch.from_numpy(np.ascontiguousarray(views))[:, None].float(),
            "semantic": torch.from_numpy(np.ascontiguousarray(arrays["semantic"])).long(),
            "instance": torch.from_numpy(np.ascontiguousarray(arrays["instance"])).long(),
            "n_contrib": torch.from_numpy(np.ascontiguousarray(arrays["n_contrib"])).long(),
            "index": index,
        }


class _SingleEvent:
    """Adapter so a plain record and a mixed event look the same to the Dataset."""

    def __init__(self, record, noise_cfg, noise_seed, noise_scale):
        self._adc = record.adc(noise_cfg=noise_cfg, noise_seed=noise_seed,
                               noise_scale=noise_scale)
        self._record = record

    def adc(self):
        return self._adc

    def dense_semantic(self):
        return self._record.dense_semantic()

    def dense_instance(self):
        return self._record.dense_instance()

    def dense_n_contrib(self):
        return self._record.dense_n_contrib()


def make_loader(dataset: PattRecoDataset, batch_size: int, shuffle: bool,
                num_workers: int = 4, drop_last: bool = False):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=drop_last, persistent_workers=num_workers > 0)
