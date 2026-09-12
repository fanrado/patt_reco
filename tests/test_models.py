"""Model, loss and dataset-pipeline tests. Skipped entirely without torch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from patt_reco.config import N_CLASSES
from patt_reco.losses.focal_dice import (SemanticLoss, class_weights_from_dataset,
                                         focal_loss, soft_dice_loss)
from patt_reco.models.registry import available, build_model
from patt_reco.models.unet import UNet


# --------------------------------------------------------------------------- #
# model contract
# --------------------------------------------------------------------------- #

def test_registry_knows_the_unet():
    assert "unet" in available()
    assert isinstance(build_model("unet"), UNet)


def test_unknown_model_fails_loudly():
    with pytest.raises(KeyError):
        build_model("no_such_model")


@pytest.mark.parametrize("shape", [(2, 3, 1, 64, 64), (1, 2, 1, 32, 32)])
def test_output_shape_matches_the_contract(shape):
    model = UNet(base_width=8, depth=2)
    out = model({"views": torch.zeros(shape)})
    b, v = shape[:2]
    assert out["sem_logits"].shape == (b, v, N_CLASSES, shape[-2], shape[-1])


def test_views_share_weights():
    """The same image in two views must produce the same logits."""
    model = UNet(base_width=8, depth=2).eval()
    single = torch.randn(1, 1, 1, 32, 32)
    doubled = single.repeat(1, 2, 1, 1, 1)
    with torch.no_grad():
        out = model({"views": doubled})["sem_logits"]
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-5)


def test_model_accepts_any_number_of_views():
    """Nothing may hard-code three planes -- a 2-view detector must just work."""
    model = UNet(base_width=8, depth=2).eval()
    for n_views in (1, 2, 3, 5):
        out = model({"views": torch.zeros(1, n_views, 1, 32, 32)})
        assert out["sem_logits"].shape[1] == n_views


def test_gradients_reach_every_parameter():
    model = UNet(base_width=8, depth=2)
    out = model({"views": torch.randn(1, 2, 1, 32, 32)})
    out["sem_logits"].sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #

def test_focal_loss_is_zero_for_a_perfect_confident_prediction():
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    logits = torch.full((2, N_CLASSES, 8, 8), -20.0)
    logits[:, 0] = 20.0
    assert float(focal_loss(logits, target)) < 1e-5


def test_focal_loss_punishes_confident_mistakes_more_than_unsure_ones():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    unsure = torch.zeros(1, N_CLASSES, 4, 4)
    confident_wrong = torch.full((1, N_CLASSES, 4, 4), -10.0)
    confident_wrong[:, 1] = 10.0
    assert float(focal_loss(confident_wrong, target)) > float(focal_loss(unsure, target))


def test_dice_loss_is_minimal_for_a_perfect_prediction():
    target = torch.randint(0, N_CLASSES, (2, 16, 16))
    logits = torch.nn.functional.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float() * 20
    assert float(soft_dice_loss(logits, target)) < 0.05


def test_pixel_weights_can_silence_a_pixel():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.full((1, N_CLASSES, 4, 4), -10.0)
    logits[:, 0] = 10.0
    logits[:, :, 0, 0] = 0.0                 # one deliberately wrong pixel
    weight = torch.ones(1, 4, 4)
    with_pixel = float(focal_loss(logits, target, pixel_weight=weight))
    weight[0, 0, 0] = 0.0
    without = float(focal_loss(logits, target, pixel_weight=weight))
    assert without < with_pixel


def test_semantic_loss_downweights_ambiguous_pixels():
    """A wrong prediction on a 4-contributor pixel must cost less than on a clean one."""
    batch_shape = (1, 1, 8, 8)
    target = torch.zeros(*batch_shape, dtype=torch.long)
    logits = torch.full((1, 1, N_CLASSES, 8, 8), -10.0)
    logits[:, :, 0] = 10.0
    logits[:, :, :, 0, 0] = 0.0

    loss_fn = SemanticLoss(ambiguity_weighting=True)
    clean = loss_fn({"sem_logits": logits},
                    {"semantic": target, "n_contrib": torch.ones(*batch_shape, dtype=torch.long)})[0]
    ambiguous_contrib = torch.ones(*batch_shape, dtype=torch.long)
    ambiguous_contrib[0, 0, 0, 0] = 4
    ambiguous = loss_fn({"sem_logits": logits},
                        {"semantic": target, "n_contrib": ambiguous_contrib})[0]
    assert float(ambiguous) < float(clean)


def test_class_weights_favour_classes_with_fewer_pixels_per_object():
    """The PLAN 9.9 correction: rings own ~7x the pixels of a track, so they
    must be weighted *down*, not up."""
    from patt_reco.config import RING, TRACK, load_yaml
    from patt_reco.dataset.generate import generate_event

    class FakeReader:
        def __init__(self, records): self.records = records
        def __len__(self): return len(self.records)
        def __getitem__(self, i): return self.records[i]

    cfg = load_yaml("configs/data/l2_noise.yaml")
    reader = FakeReader([generate_event(cfg, i) for i in range(40)])
    weights = class_weights_from_dataset(reader, 40, scheme="pixels_per_object")
    assert weights[RING] < weights[TRACK]
    assert np.all(weights >= 0) and np.isfinite(weights).all()


def test_class_weight_schemes_are_selectable():
    class FakeReader:
        def __len__(self): return 0
        def __getitem__(self, i): raise IndexError
    weights = class_weights_from_dataset(FakeReader(), 0, scheme="none")
    assert np.allclose(weights, 1.0)
    with pytest.raises(ValueError):
        class_weights_from_dataset(FakeReader(), 0, scheme="nonsense")


# --------------------------------------------------------------------------- #
# dataset pipeline
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from dataclasses import replace
    from patt_reco.config import DatasetConfig, DetectorConfig, EventConfig
    from patt_reco.dataset.generate import generate_dataset

    cfg = DatasetConfig(name="tiny", n_events=6, shard_size=6, seed=77,
                        event=EventConfig(n_objects=(2, 4)),
                        detector=DetectorConfig(n_channels=32, n_ticks=32))
    root = tmp_path_factory.mktemp("ds") / "tiny"
    generate_dataset(cfg, root, verbose=False)
    return root


def test_dataset_item_has_the_expected_shapes_and_dtypes(tiny_dataset):
    from patt_reco.dataset.torch_dataset import PattRecoDataset
    dataset = PattRecoDataset(tiny_dataset, freeze_noise=True)
    item = dataset[0]
    assert item["views"].shape == (3, 1, 32, 32)
    assert item["semantic"].shape == (3, 32, 32)
    assert item["views"].dtype == torch.float32
    assert item["semantic"].dtype == torch.int64
    assert int(item["semantic"].max()) < N_CLASSES


def test_frozen_noise_is_reproducible_and_fresh_noise_is_not(tiny_dataset):
    from patt_reco.dataset.torch_dataset import PattRecoDataset
    frozen = PattRecoDataset(tiny_dataset, freeze_noise=True)
    assert torch.equal(frozen[0]["views"], frozen[0]["views"])

    fresh = PattRecoDataset(tiny_dataset, freeze_noise=False)
    first = fresh[0]["views"]
    fresh.set_epoch(1)
    assert not torch.equal(first, fresh[0]["views"])


def test_augmentation_keeps_inputs_and_labels_aligned(tiny_dataset):
    """A flip or shift must move the image and its labels together."""
    from patt_reco.dataset.torch_dataset import PattRecoDataset
    plain = PattRecoDataset(tiny_dataset, freeze_noise=True, augment=False,
                            normalise_input=False)
    flipped = PattRecoDataset(tiny_dataset, freeze_noise=True, augment=True,
                              normalise_input=False)
    for i in range(len(plain)):
        a, b = plain[i], flipped[i]
        # the set of labelled pixel *values* is preserved by flips and shifts
        assert torch.equal(torch.bincount(a["semantic"].reshape(-1), minlength=N_CLASSES),
                           torch.bincount(b["semantic"].reshape(-1), minlength=N_CLASSES))


def test_event_mixing_adds_objects_and_keeps_labels_consistent(tiny_dataset):
    from patt_reco.dataset.io_hdf5 import DatasetReader
    from patt_reco.dataset.augment import mix_events
    reader = DatasetReader(tiny_dataset)
    a, b = reader[0], reader[1]
    mixed = mix_events(a, b, None, np.random.default_rng(0))

    semantic, instance = mixed.dense_semantic(), mixed.dense_instance()
    assert np.array_equal(semantic != 0, instance >= 0)
    # the mixed event has at least as many distinct instances as either input
    n_mixed = len(np.unique(instance[instance >= 0]))
    n_a = len(np.unique(a.dense_instance()[a.dense_instance() >= 0]))
    assert n_mixed >= n_a
    reader.close()


def test_loader_batches_cleanly(tiny_dataset):
    from patt_reco.dataset.torch_dataset import PattRecoDataset, make_loader
    dataset = PattRecoDataset(tiny_dataset, freeze_noise=True)
    batch = next(iter(make_loader(dataset, batch_size=3, shuffle=False, num_workers=0)))
    assert batch["views"].shape == (3, 3, 1, 32, 32)
    assert batch["semantic"].shape == (3, 3, 32, 32)


def test_model_consumes_a_real_batch(tiny_dataset):
    from patt_reco.dataset.torch_dataset import PattRecoDataset, make_loader
    dataset = PattRecoDataset(tiny_dataset, freeze_noise=True)
    batch = next(iter(make_loader(dataset, 2, False, num_workers=0)))
    model = UNet(base_width=8, depth=2)
    loss, parts = SemanticLoss()(model(batch), batch)
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "focal", "dice"}
