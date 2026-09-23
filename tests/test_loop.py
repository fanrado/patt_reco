"""The training loop.

Two things here are worth more than the rest. `evaluate` must return the
softmax probability of class 1, because AUC is computed from it and a wrong
column would read as a model that ranks backwards. And the cosine schedule
must span the run that actually happens: PLAN.md records a benchmark whose
verdict was decided by the seed because the learning rate hit zero a quarter
of the way in, and `min_steps` is the fix.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace                              # noqa: E402

from torch.utils.data import DataLoader, TensorDataset       # noqa: E402

from patt_reco.models import CNN                             # noqa: E402
from patt_reco.train.config import TrainConfig               # noqa: E402
from patt_reco.train.loop import Trainer                     # noqa: E402
from patt_reco.train.tracking import Run                     # noqa: E402

HEIGHT = WIDTH = 8


def separable_loader(n=32, batch_size=8, seed=0):
    """A trivially learnable set: class 1 images are brighter than class 0.

    The loop is under test, not the model, so the task is one the baseline
    cannot fail to fit.
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 2
    images = 0.1 * torch.randn(n, 1, HEIGHT, WIDTH, generator=g)
    images += labels.view(-1, 1, 1, 1).float()
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


@pytest.fixture
def cfg(tmp_path):
    base = TrainConfig()
    return replace(
        base,
        optim=replace(base.optim, epochs=2, lr=1e-2),
        run=replace(base.run, out_dir=str(tmp_path / "runs"), name="loop",
                    log_every=0),
    )


def make_trainer(cfg, train_loader=None, val_loader=None, run=None):
    model = CNN(HEIGHT, WIDTH, n_filters=4, pool=2, hidden=8)
    return Trainer(cfg, model,
                   train_loader if train_loader is not None else separable_loader(),
                   val_loader if val_loader is not None else separable_loader(seed=1),
                   torch.device("cpu"),
                   run if run is not None else Run(cfg))


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_the_optimiser_is_adamw_configured_from_the_config(cfg):
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, lr=3e-4,
                                                      weight_decay=0.05)))

    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    group = trainer.optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(0.05)


def test_the_loss_is_cross_entropy_over_class_logits(cfg):
    assert isinstance(make_trainer(cfg).criterion, torch.nn.CrossEntropyLoss)


# --------------------------------------------------------------------------- #
# the step budget
# --------------------------------------------------------------------------- #


def test_without_min_steps_the_configured_epochs_are_used(cfg):
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=5,
                                                      min_steps=0)))

    assert trainer.epochs == 5


def test_min_steps_extends_a_run_that_is_too_short(cfg):
    """4 batches an epoch, so 2 configured epochs is 8 steps; a 50-step floor
    needs 13 epochs."""
    loader = separable_loader(n=32, batch_size=8)       # 4 steps per epoch
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=2,
                                                      min_steps=50)),
                           train_loader=loader)

    assert trainer.epochs == math.ceil(50 / 4)
    assert trainer.epochs * 4 >= 50


def test_min_steps_never_shortens_a_run(cfg):
    """It is a floor, not a target: a generous epoch count stands."""
    loader = separable_loader(n=32, batch_size=8)
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=40,
                                                      min_steps=10)),
                           train_loader=loader)

    assert trainer.epochs == 40


def test_the_same_step_budget_survives_a_change_of_dataset_size(cfg):
    """The bug min_steps exists to fix: an epoch count does not transfer
    across dataset sizes, but a step count does."""
    small = separable_loader(n=16, batch_size=8)        # 2 steps per epoch
    large = separable_loader(n=160, batch_size=8)       # 20 steps per epoch
    optim = replace(cfg.optim, epochs=1, min_steps=100)

    a = make_trainer(replace(cfg, optim=optim), train_loader=small)
    b = make_trainer(replace(cfg, optim=optim), train_loader=large)

    assert a.epochs * 2 >= 100
    assert b.epochs * 20 >= 100


def test_the_schedule_spans_the_run_that_actually_happens(cfg):
    """The warning attached to the fix: annealing over the *nominal* length
    would decay the learning rate to zero mid-run and reintroduce the same
    under-training bug in a new form."""
    loader = separable_loader(n=32, batch_size=8)       # 4 steps per epoch
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=2,
                                                      min_steps=40)),
                           train_loader=loader)

    assert trainer.scheduler.T_max == trainer.epochs * 4
    assert trainer.scheduler.T_max >= 40


def test_the_learning_rate_falls_monotonically_and_only_reaches_zero_at_the_end(cfg):
    """A cosine schedule driven past its T_max turns back upward, so a
    schedule cut short shows up as a learning rate that stops falling."""
    loader = separable_loader(n=32, batch_size=8)
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=1,
                                                      min_steps=20)),
                           train_loader=loader)

    seen = []
    original = trainer.scheduler.step

    def record(*a, **k):
        original(*a, **k)
        seen.append(trainer.scheduler.get_last_lr()[0])

    trainer.scheduler.step = record
    trainer.train()

    assert len(seen) == trainer.scheduler.T_max
    assert all(b <= a + 1e-12 for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(0.0, abs=1e-9)
    # and the rate was still usable through the bulk of the run
    assert seen[len(seen) // 2] > 0.1 * cfg.optim.lr


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def test_evaluate_returns_truth_predictions_and_class_one_probabilities(cfg):
    trainer = make_trainer(cfg)
    loader = separable_loader(n=16, batch_size=4, seed=2)
    loss, y_true, y_pred, scores = trainer.evaluate(loader)

    assert len(y_true) == len(y_pred) == len(scores) == 16
    assert set(np.unique(y_true)) <= {0, 1}
    assert set(np.unique(y_pred)) <= {0, 1}
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert np.isfinite(loss)


def test_the_score_is_the_probability_of_class_one_not_class_zero(cfg):
    """Reading the wrong softmax column inverts every AUC in the project, and
    still produces perfectly plausible numbers."""
    trainer = make_trainer(cfg)
    images = torch.randn(6, 1, HEIGHT, WIDTH)
    labels = torch.randint(0, 2, (6,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=6)

    _, _, y_pred, scores = trainer.evaluate(loader)

    trainer.model.eval()
    with torch.no_grad():
        expected = torch.softmax(trainer.model(images), dim=1)[:, 1].numpy()

    assert scores == pytest.approx(expected, abs=1e-6)
    # and the prediction is the argmax, consistent with the score
    assert np.array_equal(y_pred, (expected > 0.5).astype(int))


def test_evaluate_does_not_train_the_model(cfg):
    trainer = make_trainer(cfg)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.evaluate(separable_loader(seed=3))

    for old, new in zip(before, trainer.model.parameters()):
        assert torch.equal(old, new)


def test_evaluating_nothing_returns_nan_and_empty_arrays(cfg):
    trainer = make_trainer(cfg)
    empty = DataLoader(TensorDataset(torch.empty(0, 1, HEIGHT, WIDTH),
                                     torch.empty(0, dtype=torch.long)),
                       batch_size=4)

    loss, y_true, y_pred, scores = trainer.evaluate(empty)

    assert math.isnan(loss)
    assert len(y_true) == len(y_pred) == len(scores) == 0


def test_evaluate_leaves_the_model_in_eval_mode_so_dropout_is_off(cfg):
    trainer = make_trainer(cfg)
    trainer.model.train()
    trainer.evaluate(separable_loader(seed=4))

    assert not trainer.model.training


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def test_train_logs_one_row_per_effective_epoch(cfg):
    import csv

    run = Run(cfg)
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=3)), run=run)
    trainer.train()

    with open(run.metrics_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [r["epoch"] for r in rows] == ["0", "1", "2"]
    assert set(rows[0]) == {"epoch", "train_loss", "val_loss", "val_acc", "val_auc"}


def test_an_extended_run_logs_every_extended_epoch(cfg):
    import csv

    loader = separable_loader(n=32, batch_size=8)
    run = Run(cfg)
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=1,
                                                      min_steps=20)),
                           train_loader=loader, run=run)
    trainer.train()

    with open(run.metrics_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == trainer.epochs > 1


def test_train_returns_the_best_validation_accuracy_it_logged(cfg):
    import csv

    run = Run(cfg)
    best = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=3)),
                        run=run).train()

    with open(run.metrics_path, newline="") as handle:
        logged = [float(r["val_acc"]) for r in csv.DictReader(handle)]

    assert best == pytest.approx(max(logged))


def test_train_writes_both_checkpoints(cfg):
    run = Run(cfg)
    make_trainer(cfg, run=run).train()

    assert (run.dir / "last.pt").is_file()
    assert (run.dir / "best.pt").is_file()


def test_the_last_checkpoint_holds_the_final_weights(cfg):
    run = Run(cfg)
    trainer = make_trainer(cfg, run=run)
    trainer.train()

    saved = torch.load(run.dir / "last.pt", weights_only=False)["state_dict"]
    for name, p in trainer.model.state_dict().items():
        assert torch.allclose(saved[name], p)


def test_training_actually_reduces_the_loss(cfg):
    """The separable task is trivially learnable, so a loop that does not
    learn it is broken rather than unlucky."""
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, epochs=6)))
    before, *_ = trainer.evaluate(trainer.val_loader)
    trainer.train()
    after, *_ = trainer.evaluate(trainer.val_loader)

    assert after < before


def test_gradient_clipping_bounds_the_update(cfg):
    """With a tiny clip the parameters can barely move in one step; without
    one they move freely."""
    def travel(grad_clip):
        torch.manual_seed(0)
        trainer = make_trainer(replace(cfg, optim=replace(
            cfg.optim, epochs=1, grad_clip=grad_clip, lr=1.0)))
        before = torch.cat([p.detach().flatten().clone()
                            for p in trainer.model.parameters()])
        trainer._train_epoch(0)
        after = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
        return (after - before).norm().item()

    assert travel(1e-6) < travel(0.0)


# --------------------------------------------------------------------------- #
# the wiring check
# --------------------------------------------------------------------------- #


def test_overfitting_one_batch_drives_the_loss_to_near_zero(cfg):
    """`train.py --overfit` calls this the plumbing check and passes it below
    0.05, so the threshold is asserted at the value the script uses."""
    trainer = make_trainer(replace(cfg, optim=replace(cfg.optim, lr=1e-2)))
    final = trainer.overfit_one_batch(steps=200)

    assert final < 0.05


def test_the_wiring_check_fails_when_the_labels_are_detached(cfg):
    """The failure it is meant to catch: images and labels no longer paired.

    Random labels on one fixed batch are still memorisable, so the batch is
    made larger than the model can memorise in the step budget -- what should
    not happen is the loss reaching the `train.py` pass threshold.
    """
    g = torch.Generator().manual_seed(0)
    images = torch.randn(256, 1, HEIGHT, WIDTH, generator=g)
    labels = torch.randint(0, 2, (256,), generator=g)
    scrambled = DataLoader(TensorDataset(images, labels), batch_size=256)

    trainer = make_trainer(cfg, train_loader=scrambled)
    final = trainer.overfit_one_batch(steps=30)

    assert final > 0.05


def test_the_wiring_check_writes_no_run_directory(cfg, tmp_path):
    """`train.py` passes run=None for `--overfit`, so the check must not touch
    the Run at all."""
    trainer = Trainer(cfg, CNN(HEIGHT, WIDTH, n_filters=4, pool=2, hidden=8),
                      separable_loader(), separable_loader(seed=1),
                      torch.device("cpu"), None)
    trainer.overfit_one_batch(steps=5)

    assert not (tmp_path / "runs").exists()
