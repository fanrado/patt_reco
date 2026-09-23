"""Classification metrics.

`roc_auc` is checked against an independent brute-force implementation of the
definition it claims to compute, rather than against hand-picked expected
numbers: the rank trick is exactly the kind of code that looks right and is
off by half a tie.
"""
import warnings

import numpy as np
import pytest

from patt_reco.config import CLASS_NAMES, SHOWER, TRACK
from patt_reco.train.metrics import (accuracy, confusion_matrix, format_report,
                                     per_class_precision, per_class_recall,
                                     roc_auc)


def brute_force_auc(y_true, scores):
    """AUC straight from its definition: the probability that a random
    positive outranks a random negative, ties counting a half."""
    pos = [s for s, y in zip(scores, y_true) if y == 1]
    neg = [s for s, y in zip(scores, y_true) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #


def test_accuracy_counts_exact_matches():
    assert accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0
    assert accuracy([0, 1, 1, 0], [1, 0, 0, 1]) == 0.0
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 1]) == 0.5


def test_accuracy_of_nothing_is_nan_and_says_nothing():
    """numpy would return nan here anyway, but with a RuntimeWarning. The
    guard exists to keep an empty split from spamming the per-epoch log, so
    silence is part of the contract."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(accuracy([], []))


def test_accuracy_accepts_any_shape():
    assert accuracy(np.array([[0, 1], [1, 0]]), np.array([0, 1, 1, 0])) == 1.0


# --------------------------------------------------------------------------- #
# confusion matrix
# --------------------------------------------------------------------------- #


def test_rows_are_true_classes_and_columns_are_predictions():
    """The orientation is the whole contract. An asymmetric case is the only
    way to catch a transpose, so this uses one deliberately."""
    # three tracks, all predicted shower; one shower, predicted shower
    cm = confusion_matrix([TRACK, TRACK, TRACK, SHOWER],
                          [SHOWER, SHOWER, SHOWER, SHOWER])

    assert cm[TRACK, SHOWER] == 3       # true track, called shower
    assert cm[SHOWER, TRACK] == 0       # true shower, called track -- never
    assert cm[TRACK, TRACK] == 0
    assert cm[SHOWER, SHOWER] == 1


def test_the_confusion_matrix_totals_the_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    cm = confusion_matrix(y_true, y_pred)

    assert cm.sum() == 200
    assert np.array_equal(cm.sum(axis=1), np.bincount(y_true, minlength=2))
    assert np.array_equal(cm.sum(axis=0), np.bincount(y_pred, minlength=2))
    assert np.trace(cm) == (y_true == y_pred).sum()


def test_a_perfect_classifier_is_purely_diagonal():
    y = [0, 1, 1, 0, 1]
    assert np.array_equal(confusion_matrix(y, y), np.diag(np.bincount(y)))


def test_more_classes_can_be_requested():
    cm = confusion_matrix([0, 2], [2, 2], n_classes=3)

    assert cm.shape == (3, 3)
    assert cm[0, 2] == 1 and cm[2, 2] == 1


# --------------------------------------------------------------------------- #
# per-class recall
# --------------------------------------------------------------------------- #


def test_recall_is_row_wise():
    # 4 true tracks, 3 caught; 2 true showers, 1 caught
    cm = np.array([[3, 1], [1, 1]])

    assert per_class_recall(cm) == pytest.approx([0.75, 0.5])


def test_a_class_absent_from_the_truth_has_undefined_recall():
    cm = np.array([[5, 0], [0, 0]])
    recall = per_class_recall(cm)

    assert recall[0] == 1.0
    assert np.isnan(recall[1])


def test_recall_catches_a_collapsed_classifier():
    """The failure this project actually hits: everything called one class."""
    cm = confusion_matrix([TRACK] * 50 + [SHOWER] * 50, [SHOWER] * 100)
    recall = per_class_recall(cm)

    assert recall[TRACK] == 0.0
    assert recall[SHOWER] == 1.0


# --------------------------------------------------------------------------- #
# per-class precision -- "purity"
# --------------------------------------------------------------------------- #


def test_purity_is_column_wise_where_efficiency_is_row_wise():
    """The pair is the transpose trap again, and the gate reads both. An
    asymmetric matrix is the only case that tells them apart."""
    # 4 true tracks: 3 called track, 1 called shower
    # 2 true showers: 1 called track, 1 called shower
    cm = np.array([[3, 1], [1, 1]])

    # of the 4 predicted track, 3 were track; of the 2 predicted shower, 1 was
    assert per_class_precision(cm) == pytest.approx([0.75, 0.5])
    # of the 4 true track, 3 were caught; of the 2 true shower, 1 was
    assert per_class_recall(cm) == pytest.approx([0.75, 0.5])

    # now break the symmetry so the two genuinely differ
    cm = np.array([[3, 1], [3, 1]])
    assert per_class_precision(cm) == pytest.approx([0.5, 0.5])
    assert per_class_recall(cm) == pytest.approx([0.75, 0.25])


def test_a_class_never_predicted_has_undefined_purity():
    """Mirrors per_class_recall's nan convention, and the gate relies on it:
    nan propagates through min and fails the comparison."""
    cm = np.array([[5, 0], [3, 0]])          # nothing was ever called shower

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        purity = per_class_precision(cm)

    assert purity[TRACK] == pytest.approx(5 / 8)
    assert np.isnan(purity[SHOWER])


def test_an_undefined_purity_fails_the_gate_rather_than_being_skipped():
    """`evaluate.py` takes the min over classes and compares it to the bar. A
    class that was never predicted must not quietly drop out of that min."""
    cm = confusion_matrix([TRACK] * 5 + [SHOWER] * 3, [TRACK] * 8)
    worst = per_class_precision(cm).min()

    assert np.isnan(worst)
    assert not (worst >= 0.90)               # nan comparisons are False: FAIL


def test_purity_and_efficiency_read_a_shower_biased_model_from_each_side():
    """The asymmetry PLAN.md tracks: over-predicting shower costs shower
    purity and track efficiency, and leaves the other two comfortable."""
    # every track image called shower half the time; showers all caught
    cm = confusion_matrix([TRACK] * 100 + [SHOWER] * 100,
                          [TRACK] * 50 + [SHOWER] * 50 + [SHOWER] * 100)
    purity = per_class_precision(cm)
    efficiency = per_class_recall(cm)

    assert purity[TRACK] == 1.0                      # comfortable
    assert efficiency[SHOWER] == 1.0                 # comfortable
    assert purity[SHOWER] == pytest.approx(2 / 3)    # binding
    assert efficiency[TRACK] == 0.5                  # binding


def test_a_perfect_classifier_has_unit_purity_and_efficiency():
    y = [0, 1, 1, 0, 1]
    cm = confusion_matrix(y, y)

    assert per_class_precision(cm) == pytest.approx([1.0, 1.0])
    assert per_class_recall(cm) == pytest.approx([1.0, 1.0])


# --------------------------------------------------------------------------- #
# roc auc
# --------------------------------------------------------------------------- #


def test_perfect_separation_scores_one():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_perfectly_inverted_separation_scores_zero():
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_all_scores_tied_is_one_half():
    """Every pair is a tie, and a tie is worth exactly half."""
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_is_indifferent_to_a_monotonic_rescaling():
    y = [0, 1, 0, 1, 1, 0]
    scores = np.array([0.1, 0.9, 0.4, 0.6, 0.55, 0.2])

    assert roc_auc(y, scores) == roc_auc(y, 3.0 * scores + 7.0)
    assert roc_auc(y, scores) == roc_auc(y, np.exp(scores))


def test_auc_is_the_probability_a_positive_outranks_a_negative():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(4, 60))
        y = rng.integers(0, 2, n)
        if len(set(y.tolist())) < 2:
            continue
        # coarse quantisation, so ties are common rather than incidental
        scores = np.round(rng.uniform(size=n), 1)

        assert roc_auc(y, scores) == pytest.approx(brute_force_auc(y, scores))


def test_auc_is_nan_when_a_class_is_missing_and_says_nothing():
    """Same contract as `accuracy`: a one-class batch is nan, quietly. Without
    the guard the division is 0/0, which yields nan *and* a RuntimeWarning
    every epoch."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3]))
        assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
        assert np.isnan(roc_auc([], []))


def test_auc_still_reads_a_collapsed_classifier():
    """A model predicting one class can still rank well -- the separation the
    plan relies on to tell a collapsed head from useless features."""
    y = [0] * 20 + [1] * 20
    # every score above 0.5, so argmax calls everything class 1, yet the
    # ordering is perfect
    scores = np.r_[np.linspace(0.51, 0.60, 20), np.linspace(0.61, 0.70, 20)]

    assert roc_auc(y, scores) == 1.0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def test_the_report_names_purity_and_efficiency_with_their_statistical_names():
    """patt_reco-5mi: the user's vocabulary, with the statistical name in
    parentheses once, so the mapping is unambiguous."""
    report = format_report([TRACK, SHOWER], [TRACK, SHOWER], [0.1, 0.9])

    assert "purity (precision)" in report
    assert "efficiency (recall)" in report


def test_the_report_shows_both_per_class_columns():
    cm_true = [TRACK] * 4 + [SHOWER] * 2
    cm_pred = [TRACK] * 3 + [SHOWER] * 3
    report = format_report(cm_true, cm_pred, [0.1, 0.2, 0.3, 0.8, 0.7, 0.9])

    cm = confusion_matrix(cm_true, cm_pred)
    for value in (*per_class_precision(cm), *per_class_recall(cm)):
        assert f"{value:.4f}" in report


def test_the_report_carries_the_numbers_and_the_class_names():
    y_true = [TRACK] * 3 + [SHOWER] * 3
    y_pred = [TRACK, TRACK, SHOWER, SHOWER, SHOWER, SHOWER]
    scores = [0.1, 0.2, 0.7, 0.6, 0.8, 0.9]
    report = format_report(y_true, y_pred, scores)

    for name in CLASS_NAMES.values():
        assert name in report
    assert f"{accuracy(y_true, y_pred):.4f}" in report
    assert f"{roc_auc(y_true, scores):.4f}" in report
    assert "rows = true" in report


def test_the_report_survives_a_degenerate_run():
    """nan AUC and nan recall must format rather than crash the run that
    produced them."""
    report = format_report([0, 0], [0, 0], [0.4, 0.6])

    assert "nan" in report
