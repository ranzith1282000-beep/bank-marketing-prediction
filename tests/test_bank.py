import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bank  # noqa: E402


def test_data():
    df = bank.load()
    assert len(df) == 45211 and df.isna().sum().sum() == 0
    assert abs(df["subscribed"].mean() - 0.117) < 0.001


def test_call_duration_is_never_used():
    # Duration is only known after the call ends, so using it would be cheating.
    assert "duration" not in bank.FEATURES


def test_swapped_report_arguments_swap_precision_and_recall():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert recall_score(y_pred, y_true) == precision_score(y_true, y_pred) == 0.5
    assert recall_score(y_true, y_pred) == 0.25


def test_call_top_picks_highest_scores():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
    r = bank.call_top(y, scores, 0.3)   # calls the 3 highest: rows 0, 1, 2
    assert r["conversion"] == 2 / 3 and r["share_of_all_yes"] == 2 / 3


def test_gains_curve_runs_from_zero_to_all():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    x, g = bank.gains_curve(y, rng.random(200))
    assert g[-1] == 1 and np.all(np.diff(g) >= 0) and x[-1] == 1
