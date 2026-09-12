"""Metric tests, validated against hand-computed cases rather than each other."""
import numpy as np
import pytest

from patt_reco.config import N_CLASSES
from patt_reco.eval.differential import DifferentialMetrics
from patt_reco.eval.metrics_sem import (SemanticMetrics, confusion_matrix,
                                        dice_from_confusion, iou_from_confusion,
                                        purity_efficiency)


def test_confusion_matrix_counts_by_hand():
    true = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 2, 2])
    cm = confusion_matrix(pred, true, n_classes=3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 1 and cm[1, 2] == 1
    assert cm[2, 2] == 1
    assert cm.sum() == 5


def test_iou_matches_the_definition():
    # class 1: TP=4, FP=0, FN=2 -> 4/6 ;  class 2: TP=5, FP=2, FN=0 -> 5/7
    true = np.array([0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 0, 0, 1, 2, 2, 2])
    cm = confusion_matrix(pred, true, n_classes=3)
    iou = iou_from_confusion(cm)
    assert iou[1] == pytest.approx(4 / 6)
    assert iou[2] == pytest.approx(5 / 7)


def test_perfect_prediction_gives_unit_iou():
    true = np.random.default_rng(0).integers(0, N_CLASSES, 500)
    cm = confusion_matrix(true, true)
    iou = iou_from_confusion(cm)
    assert np.allclose(iou[~np.isnan(iou)], 1.0)
    assert np.allclose(dice_from_confusion(cm)[~np.isnan(iou)], 1.0)


def test_absent_classes_are_nan_not_zero():
    """A class nobody predicted and nobody has is undefined, not perfect nor failed."""
    cm = confusion_matrix(np.array([1, 1]), np.array([1, 1]), n_classes=4)
    iou = iou_from_confusion(cm)
    assert np.isnan(iou[2]) and np.isnan(iou[3])
    assert iou[1] == pytest.approx(1.0)


def test_purity_and_efficiency_are_precision_and_recall():
    true = np.array([1, 1, 1, 2])
    pred = np.array([1, 1, 2, 2])
    cm = confusion_matrix(pred, true, n_classes=3)
    purity, efficiency = purity_efficiency(cm)
    assert efficiency[1] == pytest.approx(2 / 3)     # 2 of 3 true 1s found
    assert purity[1] == pytest.approx(1.0)           # everything called 1 was a 1
    assert purity[2] == pytest.approx(1 / 2)


def test_ambiguity_modes_differ_only_where_pixels_are_ambiguous():
    true = np.array([1, 1, 2, 2])
    pred = np.array([1, 2, 2, 2])
    contrib = np.array([1, 3, 1, 1])       # the wrong pixel is the ambiguous one

    m = SemanticMetrics(N_CLASSES)
    m.update(pred, true, contrib)

    # excluding the ambiguous pixel makes the prediction perfect
    assert m.summary("exclusive")["miou"] == pytest.approx(1.0)
    assert m.summary("all")["miou"] < 1.0
    # weighted sits between: the error is present but counts 1/3
    assert m.matrices["weighted"][1, 2] == pytest.approx(1 / 3)


def test_metrics_without_contrib_degenerate_to_all():
    true = np.array([1, 1, 2])
    pred = np.array([1, 2, 2])
    m = SemanticMetrics(N_CLASSES)
    m.update(pred, true)
    assert np.array_equal(m.matrices["all"], m.matrices["exclusive"])
    assert np.array_equal(m.matrices["all"], m.matrices["weighted"])


def test_accumulation_equals_one_shot():
    """Dataset-level IoU must come from summed counts, not averaged per-batch IoU."""
    rng = np.random.default_rng(1)
    true = rng.integers(0, N_CLASSES, (10, 64))
    pred = rng.integers(0, N_CLASSES, (10, 64))

    batched = SemanticMetrics(N_CLASSES)
    for i in range(10):
        batched.update(pred[i], true[i])
    one_shot = SemanticMetrics(N_CLASSES)
    one_shot.update(pred, true)
    assert np.allclose(batched.matrices["all"], one_shot.matrices["all"])


def test_foreground_summary_collapses_classes():
    true = np.array([0, 0, 1, 2, 3])
    pred = np.array([0, 1, 2, 2, 0])       # one false positive, one false negative
    m = SemanticMetrics(N_CLASSES)
    m.update(pred, true)
    fg = m.foreground_summary("all")
    # hit-hit = 2 (the 1->2 and 2->2 pixels), fp = 1, fn = 1
    assert fg["iou"] == pytest.approx(2 / 4)


def test_differential_bins_by_covariate():
    diff = DifferentialMetrics("n", np.array([0, 2, 4, 6]))
    perfect = np.array([1, 1, 1, 1])
    wrong = np.array([2, 2, 2, 2])
    diff.update(1.0, perfect, perfect)           # bin 0: perfect
    diff.update(5.0, wrong, perfect)             # bin 2: entirely wrong
    curve = diff.curve("all")
    assert curve["n_events"] == [1, 0, 1]
    assert curve["miou"][0] == pytest.approx(1.0)
    assert curve["miou"][2] == pytest.approx(0.0)
    assert np.isnan(curve["miou"][1])            # empty bin, not zero


def test_differential_clips_out_of_range_values():
    diff = DifferentialMetrics("n", np.array([0, 5, 10]))
    assert diff.bin_index(-3) == 0
    assert diff.bin_index(1e6) == 1
