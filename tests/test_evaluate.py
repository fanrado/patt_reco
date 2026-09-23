"""`scripts/evaluate.py` end to end, through the real command line.

This script produces the benchmark's verdict, so it is exercised as a
subprocess against a real checkpoint rather than by importing its parts. Two
of its contracts cannot be checked any other way: that the gate reads the
worst class rather than the mean, and that a FAIL leaves the exit status at
zero -- `scripts/sweep.py` treats a non-zero exit as a failed level, so a
legitimate below-bar measurement that looked like a crash would corrupt every
sweep.
"""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from patt_reco.models import build_model                    # noqa: E402
from patt_reco.train.config import ModelConfig, TrainConfig  # noqa: E402
from patt_reco.train.tracking import Run                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EVALUATE = REPO / "scripts" / "evaluate.py"
SIZE = 16


@pytest.fixture
def scored(tmp_path):
    """A checkpoint and a split it can be scored against.

    The model is untrained, so it predicts near-chance and the gate should
    return FAIL at the default bar -- which is the case worth testing, since
    FAIL is the verdict that must not change the exit code.
    """
    rng = np.random.default_rng(0)
    images = rng.random((40, SIZE, SIZE)).astype(np.float32)
    labels = (np.arange(40) % 2).astype(np.uint8)
    data = tmp_path / "test.npz"
    np.savez_compressed(data, images=images, labels=labels)

    base = TrainConfig()
    cfg = replace(base, model=ModelConfig(n_filters=4, pool=2, hidden=8),
                  run=replace(base.run, out_dir=str(tmp_path / "runs"), name="ev"))
    run = Run(cfg)
    checkpoint = run.save(build_model(cfg.model, SIZE, SIZE), "best.pt")
    return checkpoint, data


@pytest.fixture
def trained(tmp_path):
    """A checkpoint that genuinely separates its split, so the gate can PASS.

    The untrained fixture cannot reach PASS at any threshold: it predicts one
    class, leaving the other's purity undefined, and nan fails every
    comparison. That is correct, but it means a passing gate needs a model
    that has actually learned something.
    """
    g = torch.Generator().manual_seed(0)
    labels = (np.arange(40) % 2).astype(np.uint8)
    images = 0.05 * torch.randn(40, 1, SIZE, SIZE, generator=g)
    images += torch.from_numpy(labels).view(-1, 1, 1, 1).float()

    base = TrainConfig()
    cfg = replace(base, model=ModelConfig(n_filters=4, pool=2, hidden=8),
                  run=replace(base.run, out_dir=str(tmp_path / "runs"), name="fit"))
    model = build_model(cfg.model, SIZE, SIZE)

    y = torch.from_numpy(labels).long()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(120):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(images), y).backward()
        opt.step()
    model.eval()
    assert set(model(images).argmax(1).tolist()) == {0, 1}, "fixture did not learn"

    data = tmp_path / "test.npz"
    np.savez_compressed(data, images=images.squeeze(1).numpy().astype(np.float32),
                        labels=labels)
    return Run(cfg).save(model, "best.pt"), data


@pytest.fixture
def collapsed(tmp_path):
    """A checkpoint that predicts one class for everything.

    Built explicitly rather than by leaving a model untrained: whether random
    weights happen to collapse is luck, and the nan path deserves a fixture
    that always exercises it. The head is zeroed and biased hard toward
    track, so shower is never predicted and its purity is undefined.
    """
    rng = np.random.default_rng(0)
    labels = (np.arange(40) % 2).astype(np.uint8)
    images = rng.random((40, SIZE, SIZE)).astype(np.float32)

    base = TrainConfig()
    cfg = replace(base, model=ModelConfig(n_filters=4, pool=2, hidden=8),
                  run=replace(base.run, out_dir=str(tmp_path / "runs"), name="dead"))
    model = build_model(cfg.model, SIZE, SIZE)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([10.0, -10.0]))
    model.eval()
    assert set(model(torch.from_numpy(images).unsqueeze(1)).argmax(1).tolist()) == {0}

    data = tmp_path / "test.npz"
    np.savez_compressed(data, images=images, labels=labels)
    return Run(cfg).save(model, "best.pt"), data


@pytest.fixture
def asymmetric(tmp_path):
    """A checkpoint that is strong on one class and weak on the other.

    The gate reads the worst class, and a symmetric model cannot show that:
    when both classes score alike, the minimum and the mean coincide and a
    gate computed either way looks identical. This fixture leans the head
    toward shower, reproducing the asymmetry PLAN.md tracks, so the two
    differ.
    """
    g = torch.Generator().manual_seed(0)
    labels = (np.arange(60) % 2).astype(np.uint8)
    images = 0.05 * torch.randn(60, 1, SIZE, SIZE, generator=g)
    images += torch.from_numpy(labels).view(-1, 1, 1, 1).float()

    base = TrainConfig()
    cfg = replace(base, model=ModelConfig(n_filters=4, pool=2, hidden=8),
                  run=replace(base.run, out_dir=str(tmp_path / "runs"), name="skew"))
    model = build_model(cfg.model, SIZE, SIZE)

    y = torch.from_numpy(labels).long()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(120):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(images), y).backward()
        opt.step()
    model.eval()

    # Tilt the head toward shower by a computed amount rather than by
    # searching: adding b to the class-1 logit flips a track exactly when b
    # exceeds that track's margin, so the 70th percentile of those margins
    # misreads 30% of tracks. Raising the class-1 logit can never turn a
    # shower into a track, so shower efficiency stays at 1.0 and the
    # asymmetry is one-sided, as PLAN.md describes it.
    with torch.no_grad():
        logits = model(images)
        margins = (logits[:, 0] - logits[:, 1])[torch.from_numpy(labels) == 0]
        model.head.bias[1] += float(torch.quantile(margins, 0.70))

        pred = model(images).argmax(1).numpy()
    assert 0 < ((labels == 0) & (pred == 1)).sum() < (labels == 0).sum()
    assert ((labels == 1) & (pred == 0)).sum() == 0

    data = tmp_path / "test.npz"
    np.savez_compressed(data, images=images.squeeze(1).numpy().astype(np.float32),
                        labels=labels)
    return Run(cfg).save(model, "best.pt"), data


def evaluate(checkpoint, data, *extra):
    return subprocess.run(
        [sys.executable, str(EVALUATE), str(checkpoint), "--data", str(data), *extra],
        capture_output=True, text=True, cwd=REPO)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def test_a_checkpoint_scores_without_a_training_config_on_hand(scored):
    checkpoint, data = scored
    result = evaluate(checkpoint, data)

    assert result.returncode == 0, result.stderr
    assert "accuracy" in result.stdout
    assert "roc auc" in result.stdout


def test_the_model_is_rebuilt_at_the_checkpoints_depth(tmp_path):
    """The inert-config defect reaching evaluate.py would load a 3-block
    checkpoint into a 1-block model, which cannot even load_state_dict."""
    rng = np.random.default_rng(1)
    data = tmp_path / "test.npz"
    np.savez_compressed(data, images=rng.random((8, 32, 32)).astype(np.float32),
                        labels=(np.arange(8) % 2).astype(np.uint8))

    base = TrainConfig()
    cfg = replace(base, model=ModelConfig(n_filters=4, pool=2, n_blocks=3, hidden=8),
                  run=replace(base.run, out_dir=str(tmp_path / "runs"), name="deep"))
    run = Run(cfg)
    checkpoint = run.save(build_model(cfg.model, 32, 32), "best.pt")

    result = evaluate(checkpoint, data)
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def test_the_gate_reports_all_three_thresholds_and_a_verdict(scored):
    checkpoint, data = scored
    out = evaluate(checkpoint, data).stdout

    assert "gate, threshold 0.90" in out
    assert "accuracy" in out
    assert "purity (min over classes)" in out
    assert "efficiency (min over classes)" in out
    assert "overall" in out


def test_a_failing_gate_still_exits_zero(scored):
    """patt_reco-98z is explicit: sweep.py treats a non-zero exit as a failed
    level, so a below-bar measurement must not look like a crash."""
    checkpoint, data = scored
    result = evaluate(checkpoint, data, "--threshold", "0.999")

    assert result.returncode == 0
    assert "FAIL" in result.stdout

    report = json.loads((checkpoint.parent / "test_report.json").read_text())
    assert report["gate"]["verdict"] == "FAIL"


def test_a_passing_gate_also_exits_zero(trained):
    checkpoint, data = trained
    result = evaluate(checkpoint, data, "--threshold", "0.5")

    assert result.returncode == 0
    report = json.loads((checkpoint.parent / "test_report.json").read_text())
    assert report["gate"]["verdict"] == "PASS"
    assert "PASS" in result.stdout


def test_a_collapsed_model_fails_the_gate_at_every_threshold(collapsed):
    """A class that is never predicted has undefined purity, and nan fails
    every comparison. Lowering the bar must not rescue it -- a model that
    ignores a class has not measured it."""
    checkpoint, data = collapsed
    result = evaluate(checkpoint, data, "--threshold", "0.0")

    assert result.returncode == 0
    report = json.loads((checkpoint.parent / "test_report.json").read_text())
    assert np.isnan(report["gate"]["purity_min"]["value"])
    assert report["gate"]["verdict"] == "FAIL"


def test_the_threshold_is_configurable_rather_than_hard_coded(scored):
    checkpoint, data = scored
    out = evaluate(checkpoint, data, "--threshold", "0.25").stdout

    assert "gate, threshold 0.25" in out
    report = json.loads((checkpoint.parent / "test_report.json").read_text())
    assert report["gate"]["threshold"] == 0.25


def test_the_gate_binds_on_the_worst_class_not_the_mean(asymmetric):
    """A mean hides one class failing, which is the whole reason the rule is
    written per class.

    The model here is deliberately lopsided, so the minimum and the mean are
    different numbers: a gate quietly averaging its classes would report a
    healthier figure than the worst class actually scored.
    """
    checkpoint, data = asymmetric
    evaluate(checkpoint, data)
    report = json.loads((checkpoint.parent / "test_report.json").read_text())

    purity = np.array(list(report["per_class_purity"].values()))
    efficiency = np.array(list(report["per_class_recall"].values()))

    # the fixture is only meaningful if the classes really do differ
    assert purity.min() < purity.mean()
    assert efficiency.min() < efficiency.mean()

    assert report["gate"]["purity_min"]["value"] == pytest.approx(purity.min())
    assert report["gate"]["efficiency_min"]["value"] == pytest.approx(efficiency.min())


def test_a_lopsided_model_fails_a_bar_its_average_would_clear(asymmetric):
    """The concrete consequence: averaging would let a model through on the
    strength of the class it already handles."""
    checkpoint, data = asymmetric
    evaluate(checkpoint, data)
    report = json.loads((checkpoint.parent / "test_report.json").read_text())

    efficiency = np.array(list(report["per_class_recall"].values()))
    bar = 0.5 * (efficiency.min() + efficiency.mean())      # between the two

    result = evaluate(checkpoint, data, "--threshold", f"{bar:.6f}")
    gate = json.loads((checkpoint.parent / "test_report.json").read_text())["gate"]

    assert result.returncode == 0
    assert gate["efficiency_min"]["value"] < bar
    assert gate["verdict"] == "FAIL"


def test_the_overall_verdict_is_the_conjunction_of_the_three(trained):
    checkpoint, data = trained
    for threshold in ("0.0", "0.5", "0.999"):
        evaluate(checkpoint, data, "--threshold", threshold)
        gate = json.loads((checkpoint.parent / "test_report.json").read_text())["gate"]

        expected = all(gate[k]["pass"] for k in
                       ("accuracy", "purity_min", "efficiency_min"))
        assert gate["verdict"] == ("PASS" if expected else "FAIL")


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def test_the_report_records_enough_to_compare_runs_without_rerunning(scored):
    checkpoint, data = scored
    evaluate(checkpoint, data)
    report = json.loads((checkpoint.parent / "test_report.json").read_text())

    assert set(report) >= {"accuracy", "per_class_purity", "per_class_recall",
                           "confusion_matrix", "roc_auc", "gate", "checkpoint",
                           "data", "n_images"}
    assert report["n_images"] == 40
    assert report["confusion_matrix_note"] == "rows are true classes, columns predicted"


def test_the_recorded_confusion_matrix_totals_the_images(scored):
    checkpoint, data = scored
    evaluate(checkpoint, data)
    report = json.loads((checkpoint.parent / "test_report.json").read_text())

    assert np.sum(report["confusion_matrix"]) == report["n_images"]


def test_the_report_is_written_beside_the_checkpoint(scored):
    checkpoint, data = scored
    evaluate(checkpoint, data)

    assert (checkpoint.parent / "test_report.json").is_file()


def test_the_worst_misclassifications_are_plotted(scored):
    """A metric cannot show that the errors are gently curved tracks; the
    grid is what caught the inverted difficulty axis."""
    checkpoint, data = scored
    result = evaluate(checkpoint, data, "--n-worst", "4")

    figure = checkpoint.parent / "misclassified.png"
    assert "misclassified" in result.stdout
    if "no misclassifications" not in result.stdout:
        assert figure.is_file()
