"""What a training run writes to disk.

The run directory is the only record that survives a run, so these tests care
about whether it can be read back: a checkpoint that reloads without the
training config on hand, and a metrics CSV whose rows line up with its header.
"""
import csv
import json

import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace                          # noqa: E402

from patt_reco.config import config_hash                 # noqa: E402
from patt_reco.models import CNN                         # noqa: E402
from patt_reco.train.config import TrainConfig           # noqa: E402
from patt_reco.train.tracking import Run                 # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return replace(TrainConfig(), run=replace(TrainConfig().run,
                                              out_dir=str(tmp_path / "runs"),
                                              name="unit"))


# --------------------------------------------------------------------------- #
# the run directory
# --------------------------------------------------------------------------- #


def test_a_run_creates_a_timestamped_directory_under_out_dir(cfg, tmp_path):
    run = Run(cfg)

    assert run.dir.is_dir()
    assert run.dir.parent == tmp_path / "runs"
    assert run.dir.name.endswith("_unit")
    # YYYYmmdd_HHMMSS
    stamp = run.dir.name[:-len("_unit")]
    assert len(stamp) == 15 and stamp[8] == "_"
    assert stamp.replace("_", "").isdigit()


def test_an_unnamed_run_falls_back_to_train(cfg):
    run = Run(replace(cfg, run=replace(cfg.run, name="")))

    assert run.dir.name.endswith("_train")


def test_the_config_is_recorded_with_its_hash(cfg):
    run = Run(cfg)
    saved = json.loads((run.dir / "config.json").read_text())

    assert saved["config_hash"] == config_hash(cfg)
    assert saved["config"]["optim"]["epochs"] == cfg.optim.epochs
    assert saved["config"]["run"]["seed"] == cfg.run.seed


def test_the_recorded_config_reflects_overrides_not_the_file(cfg):
    """`train.py` applies `--set` before constructing the Run, so config.json
    describes what was actually trained."""
    overridden = replace(cfg, optim=replace(cfg.optim, epochs=99))
    run = Run(overridden)
    saved = json.loads((run.dir / "config.json").read_text())

    assert saved["config"]["optim"]["epochs"] == 99


def test_the_environment_is_recorded(cfg):
    run = Run(cfg)
    env = json.loads((run.dir / "environment.json").read_text())

    assert set(env) == {"python", "torch", "numpy", "platform", "device", "git_sha"}
    assert env["torch"] == torch.__version__
    assert env["device"]                      # a name, never empty
    assert env["git_sha"]                     # a sha, or the literal 'unknown'


# --------------------------------------------------------------------------- #
# metrics log
# --------------------------------------------------------------------------- #


def test_the_header_follows_the_first_logged_row(cfg):
    run = Run(cfg)
    run.log(0, train_loss=0.5, val_acc=0.8)
    run.log(1, train_loss=0.4, val_acc=0.9)

    with open(run.metrics_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == ["epoch", "train_loss", "val_acc"]
    assert rows[0]["epoch"] == "0" and rows[0]["val_acc"] == "0.8"
    assert rows[1]["epoch"] == "1" and rows[1]["val_acc"] == "0.9"


def test_every_logged_epoch_appears_once(cfg):
    run = Run(cfg)
    for epoch in range(5):
        run.log(epoch, val_acc=0.1 * epoch)

    with open(run.metrics_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [r["epoch"] for r in rows] == ["0", "1", "2", "3", "4"]


def test_a_nan_metric_is_written_rather_than_crashing_the_run(cfg):
    run = Run(cfg)
    run.log(0, val_auc=float("nan"))

    assert "nan" in run.metrics_path.read_text()


def test_two_runs_keep_separate_metrics_files(cfg):
    a, b = Run(cfg), Run(replace(cfg, run=replace(cfg.run, name="other")))
    a.log(0, val_acc=1.0)
    b.log(0, val_acc=0.0)

    assert "1.0" in a.metrics_path.read_text()
    assert "1.0" not in b.metrics_path.read_text()


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


def test_a_checkpoint_can_rebuild_its_model_without_a_training_config(cfg):
    """`evaluate.py` is handed a .pt and nothing else, so the frame size has
    to travel with the weights."""
    run = Run(cfg)
    model = CNN(24, 16, n_filters=4, pool=2, hidden=8)
    path = run.save(model, "ckpt.pt")

    saved = torch.load(path, weights_only=False)
    assert saved["height"] == 24
    assert saved["width"] == 16
    assert saved["config"] == cfg

    rebuilt = CNN(saved["height"], saved["width"], n_filters=4, pool=2, hidden=8)
    rebuilt.load_state_dict(saved["state_dict"])

    model.eval(), rebuilt.eval()
    x = torch.randn(2, 1, 24, 16)
    assert torch.allclose(model(x), rebuilt(x))


def test_best_and_last_are_written_under_their_expected_names(cfg):
    run = Run(cfg)
    model = CNN(16, 16, pool=4)

    assert run.save_best(model).name == "best.pt"
    assert run.save_last(model).name == "last.pt"
    assert (run.dir / "best.pt").is_file()
    assert (run.dir / "last.pt").is_file()


def test_saving_again_overwrites_rather_than_accumulating(cfg):
    run = Run(cfg)
    model = CNN(16, 16, pool=4)
    run.save_last(model)

    with torch.no_grad():
        model.head.bias.add_(1.0)
    run.save_last(model)

    saved = torch.load(run.dir / "last.pt", weights_only=False)
    assert torch.allclose(saved["state_dict"]["head.bias"], model.head.bias)
    assert sorted(p.name for p in run.dir.glob("*.pt")) == ["last.pt"]
