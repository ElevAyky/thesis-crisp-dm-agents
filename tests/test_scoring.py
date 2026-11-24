"""Scorer-Units: Der Harness rechnet F1/Accuracy selbst aus einer Predictions-Datei
gegen holdout/test.parquet — Self-Report der Worker zählt nicht (Spec §1.2)."""
import os
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from crispfsm.scoring import ScoreResult, score_predictions


@pytest.fixture
def holdout(tmp_path):
    """holdout/test.parquet mit 6 Zeilen, Target 'col_2' (String-Labels wie nach
    Neutralisierung)."""
    path = os.path.join(tmp_path, "test.parquet")
    pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "col_2": ["a", "b", "a", "b", "a", "b"],
    }).to_parquet(path, index=False)
    return path


def _write_preds(tmp_path, values):
    path = os.path.join(tmp_path, "predictions.parquet")
    pd.DataFrame({"prediction": values}).to_parquet(path, index=False)
    return path


def test_perfect_predictions_score_1(tmp_path, holdout):
    pred = _write_preds(tmp_path, ["a", "b", "a", "b", "a", "b"])
    res = score_predictions(pred, holdout, target="col_2")
    assert res.ok is True
    assert res.f1 == 1.0
    assert res.accuracy == 1.0


def test_scores_match_sklearn_reference(tmp_path, holdout):
    y_true = ["a", "b", "a", "b", "a", "b"]
    y_pred = ["a", "a", "a", "b", "b", "b"]
    pred = _write_preds(tmp_path, y_pred)
    res = score_predictions(pred, holdout, target="col_2")
    assert res.ok is True
    assert res.f1 == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert res.accuracy == pytest.approx(accuracy_score(y_true, y_pred))


def test_f1_is_macro_not_weighted_on_imbalance(tmp_path):
    """Decker 14.09.: Schwelle = best-Baseline-F1, und die Baseline misst macro-F1
    (scripts/baseline_difficulty.py). Auf unbalancierten Sets divergieren macro und
    weighted deutlich — der Harness muss dieselbe Größe messen wie die Schwelle."""
    holdout_path = os.path.join(tmp_path, "test.parquet")
    y_true = ["a"] * 8 + ["b"] * 2
    y_pred = ["a"] * 10                      # Majority-Klasse geraten
    pd.DataFrame({"f": range(10), "target": y_true}).to_parquet(holdout_path, index=False)
    pred = _write_preds(tmp_path, y_pred)
    res = score_predictions(pred, holdout_path, target="target")
    assert res.ok is True
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")
    assert macro != pytest.approx(weighted)  # Testfall diskriminiert wirklich
    assert res.f1 == pytest.approx(macro)


def test_int_predictions_scored_against_int_labels(tmp_path):
    """Ohne Neutralisierung sind Labels Ints — Typ-Normierung (str) beidseitig."""
    holdout_path = os.path.join(tmp_path, "test.parquet")
    pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 1]}).to_parquet(holdout_path, index=False)
    pred = _write_preds(tmp_path, [0, 1])
    res = score_predictions(pred, holdout_path, target="target")
    assert res.ok is True and res.f1 == 1.0


def test_missing_predictions_file(tmp_path, holdout):
    res = score_predictions(os.path.join(tmp_path, "nope.parquet"), holdout, target="col_2")
    assert res.ok is False
    assert "missing" in res.feedback
    assert res.f1 is None


def test_missing_prediction_column(tmp_path, holdout):
    path = os.path.join(tmp_path, "predictions.parquet")
    pd.DataFrame({"pred": ["a"] * 6}).to_parquet(path, index=False)
    res = score_predictions(path, holdout, target="col_2")
    assert res.ok is False
    assert "'prediction'" in res.feedback


def test_row_count_mismatch(tmp_path, holdout):
    pred = _write_preds(tmp_path, ["a", "b"])          # 2 statt 6
    res = score_predictions(pred, holdout, target="col_2")
    assert res.ok is False
    assert "2" in res.feedback and "6" in res.feedback


def test_float_predictions_against_int_labels_score_1(tmp_path):
    """Dtype-Drift (Worker schreibt 1.0 statt 1) darf keinen stillen 0-Score erzeugen."""
    holdout_path = os.path.join(tmp_path, "test.parquet")
    pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 1]}).to_parquet(holdout_path, index=False)
    pred = _write_preds(tmp_path, [0.0, 1.0])
    res = score_predictions(pred, holdout_path, target="target")
    assert res.ok is True and res.f1 == 1.0


def test_disjoint_label_space_is_form_error(tmp_path, holdout):
    """Scan-Befund 08.08. (jm1): Int-Codes statt String-Labels liefen still als
    F1=0.0 durch — Komplett-Mismatch des Label-Raums ist ein Formfehler."""
    pred = _write_preds(tmp_path, [0, 1, 0, 1, 0, 1])
    res = score_predictions(pred, holdout, target="col_2")
    assert res.ok is False
    assert res.f1 is None
    assert "target label space" in res.feedback


def test_probabilities_instead_of_labels_is_form_error(tmp_path, holdout):
    """Scan-Befund 08.08. (bank_marketing): Wahrscheinlichkeiten statt Labels."""
    pred = _write_preds(tmp_path, [0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    res = score_predictions(pred, holdout, target="col_2")
    assert res.ok is False
    assert "target label space" in res.feedback


def test_int_predictions_for_digit_string_labels_still_score(tmp_path):
    """Regression: Int-Predictions gegen String-Digit-Labels ('1'/'2', z. B.
    bank_marketing) sind KEIN Mismatch — die str-Normierung macht sie vergleichbar."""
    holdout_path = os.path.join(tmp_path, "test.parquet")
    pd.DataFrame({"x": [1.0] * 6, "col_1": ["1", "2", "1", "2", "1", "2"]}
                 ).to_parquet(holdout_path, index=False)
    pred = _write_preds(tmp_path, [1, 2, 1, 2, 1, 2])
    res = score_predictions(pred, holdout_path, target="col_1")
    assert res.ok is True
    assert res.f1 == 1.0


def test_partial_label_overlap_scores_normally(tmp_path, holdout):
    """Einzelne unbekannte Labels (Tippfehler-Klasse) sind kein Formfehler —
    sie zählen als falsch, der Rest scored normal."""
    pred = _write_preds(tmp_path, ["a", "b", "a", "b", "a", "typo"])
    res = score_predictions(pred, holdout, target="col_2")
    assert res.ok is True
    assert 0 < res.f1 < 1.0


def test_nan_predictions_fail_with_diagnosis(tmp_path, holdout):
    pred = _write_preds(tmp_path, ["a", None, "a", "b", "a", "b"])
    res = score_predictions(pred, holdout, target="col_2")
    assert res.ok is False
    assert "NaN" in res.feedback


def test_corrupt_parquet(tmp_path, holdout):
    path = os.path.join(tmp_path, "predictions.parquet")
    with open(path, "w") as f:
        f.write("das ist kein parquet")
    res = score_predictions(path, holdout, target="col_2")
    assert res.ok is False
    assert res.f1 is None


# --- Gepaarter Nicht-Unterlegenheits-Test gegen die Baseline (23.09.) --------

import numpy as np
from crispfsm.scoring import (compare_to_baseline, macro_f1_from_confusion,
                              paired_bootstrap_f1_diff)


def test_macro_f1_from_confusion_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = np.array(["a", "b", "c"])
    for _ in range(20):
        y_t = labels[rng.integers(0, 3, 50)]
        y_p = labels[rng.integers(0, 3, 50)]
        idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((3, 3))
        for t, p in zip(y_t, y_p):
            cm[idx[t], idx[p]] += 1
        assert macro_f1_from_confusion(cm) == pytest.approx(
            f1_score(y_t, y_p, average="macro", zero_division=0))


def test_macro_f1_ignores_classes_absent_in_truth_and_prediction():
    """sklearn-Semantik: nur Klassen aus y_true ∪ y_pred zaehlen (Bootstrap-
    Stichproben koennen eine Klasse komplett verlieren)."""
    cm = np.array([[3, 1, 0], [0, 2, 0], [0, 0, 0]], dtype=float)
    assert macro_f1_from_confusion(cm) == pytest.approx(
        f1_score(["a"] * 4 + ["b"] * 2, ["a"] * 3 + ["b"] + ["b"] * 2, average="macro"))


def test_bootstrap_one_error_vs_perfect_baseline_is_noninferior():
    """wine-Fall: Baseline 36/36, Agent 35/36 -> nicht signifikant schlechter."""
    y = np.array(["a"] * 12 + ["b"] * 12 + ["c"] * 12)
    base = y.copy()
    agent = y.copy(); agent[0] = "b"
    r = paired_bootstrap_f1_diff(y, agent, base)
    assert r.diff < 0
    assert r.ci_hi >= 0 and r.noninferior


def test_bootstrap_systematic_gap_is_inferior():
    """credit_g-artig: Agent auf vielen Zeilen schlechter -> FAIL, obwohl beide
    F1-Werte einzeln stark streuen (gepaart kuerzt sich die Test-Schwere raus)."""
    rng = np.random.default_rng(1)
    y = rng.choice(["good", "bad"], 400, p=[0.7, 0.3])
    base = y.copy(); flip = rng.random(400) < 0.20
    base[flip] = np.where(y[flip] == "good", "bad", "good")
    agent = base.copy(); worse = (base == y) & (rng.random(400) < 0.12)
    agent[worse] = np.where(y[worse] == "good", "bad", "good")
    r = paired_bootstrap_f1_diff(y, agent, base)
    assert r.ci_hi < 0 and not r.noninferior


def test_bootstrap_never_stricter_than_k1():
    """Agent >= Baseline (Punktschaetzung) ist immer SUCCESS — die Regel ist eine
    Obermenge der strikten k = 1-Regel, nie strenger."""
    rng = np.random.default_rng(2)
    for _ in range(30):
        y = rng.choice(["a", "b", "c"], 120)
        base = np.where(rng.random(120) < 0.7, y, rng.choice(["a", "b", "c"], 120))
        agent = np.where(rng.random(120) < 0.7, y, rng.choice(["a", "b", "c"], 120))
        r = paired_bootstrap_f1_diff(y, agent, base)
        if r.diff >= 0:
            assert r.noninferior


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(3)
    y = rng.choice(["a", "b"], 300)
    a = np.where(rng.random(300) < 0.8, y, "a")
    b = np.where(rng.random(300) < 0.8, y, "b")
    assert paired_bootstrap_f1_diff(y, a, b) == paired_bootstrap_f1_diff(y, a, b)


def test_bootstrap_is_fast_on_large_holdout():
    """Kern-Sets bis 138k Holdout-Zeilen, Eval-Gate bis 3x pro Run: Aggregation
    ueber (wahr, Agent, Baseline)-Zellen statt Zeilen -> kein Zeilen-Resampling."""
    import time
    rng = np.random.default_rng(4)
    y = rng.choice(list("abcdefg"), 140_000)
    a = np.where(rng.random(140_000) < 0.9, y, "a")
    b = np.where(rng.random(140_000) < 0.9, y, "b")
    t0 = time.perf_counter()
    paired_bootstrap_f1_diff(y, a, b)
    assert time.perf_counter() - t0 < 5.0


def test_compare_to_baseline_reads_files(tmp_path, holdout):
    agent = _write_preds(tmp_path, ["a", "b", "a", "b", "a", "a"])
    base_path = os.path.join(tmp_path, "baseline.parquet")
    pd.DataFrame({"prediction": ["a", "b", "a", "b", "a", "b"]}).to_parquet(base_path, index=False)
    c = compare_to_baseline(agent, holdout, base_path, target="col_2")
    assert c.baseline_f1 == 1.0
    assert c.f1 == pytest.approx(f1_score(["a", "b"] * 3, ["a", "b", "a", "b", "a", "a"], average="macro"))
    assert c.diff == pytest.approx(c.f1 - 1.0)


def test_compare_to_baseline_none_on_form_error(tmp_path, holdout):
    agent = _write_preds(tmp_path, ["a", "b"])                  # falsche Zeilenzahl
    base_path = os.path.join(tmp_path, "baseline.parquet")
    pd.DataFrame({"prediction": ["a", "b"] * 3}).to_parquet(base_path, index=False)
    assert compare_to_baseline(agent, holdout, base_path, target="col_2") is None


# --- Holdout mit Validierungsanteil (S1, 23.09.) ------------------------------

import json as _json
from crispfsm.scoring import compare_part_to_baseline, gate_part, score_part


@pytest.fixture
def split_holdout(tmp_path):
    """val = 4 Zeilen, test = 6 Zeilen, gemeinsame Feature-Datei gemischt (10 Zeilen)."""
    h = tmp_path / "holdout"
    h.mkdir()
    pd.DataFrame({"x": range(4), "t": ["a", "b", "a", "b"]}).to_parquet(h / "val.parquet", index=False)
    pd.DataFrame({"x": range(6), "t": ["a", "a", "b", "b", "a", "b"]}).to_parquet(h / "test.parquet", index=False)
    idx = {"val": [7, 2, 9, 0], "test": [1, 3, 4, 5, 6, 8]}
    (h / "eval_index.json").write_text(_json.dumps(idx), encoding="utf-8")
    return h, idx


def _combined(idx, val_preds, test_preds):
    out = [None] * 10
    for pos, v in zip(idx["val"], val_preds):
        out[pos] = v
    for pos, v in zip(idx["test"], test_preds):
        out[pos] = v
    return out


def test_score_part_reads_the_right_rows(tmp_path, split_holdout):
    h, idx = split_holdout
    # val perfekt, test komplett falsch -> val F1 1.0, test F1 0.0
    preds = _combined(idx, ["a", "b", "a", "b"], ["b", "b", "a", "a", "b", "a"])
    p = _write_preds(tmp_path, preds)
    assert score_part(p, str(h), "val", target="t").f1 == 1.0
    assert score_part(p, str(h), "test", target="t").f1 == 0.0


def test_score_part_requires_full_combined_file(tmp_path, split_holdout):
    h, _ = split_holdout
    p = _write_preds(tmp_path, ["a"] * 6)                    # nur Test-Zeilen geliefert
    res = score_part(p, str(h), "test", target="t")
    assert res.ok is False and "Row count" in res.feedback and "expected 10" in res.feedback


def test_gate_part_is_val_only_with_index(tmp_path, split_holdout, holdout):
    h, _ = split_holdout
    assert gate_part(str(h)) == "val"
    assert gate_part(os.path.dirname(holdout)) == "test"


def test_score_part_without_index_equals_legacy(tmp_path, holdout):
    p = _write_preds(tmp_path, ["a", "a", "a", "b", "b", "b"])
    assert score_part(p, os.path.dirname(holdout), "test", target="col_2") == \
        score_predictions(p, holdout, target="col_2")
    with pytest.raises(ValueError):
        score_part(p, os.path.dirname(holdout), "val", target="col_2")


def test_compare_part_uses_part_specific_baseline(tmp_path, split_holdout):
    h, idx = split_holdout
    preds = _combined(idx, ["a", "b", "a", "b"], ["a", "a", "b", "b", "a", "b"])
    p = _write_preds(tmp_path, preds)
    base_val = str(tmp_path / "bval.parquet")
    pd.DataFrame({"prediction": ["a", "a", "a", "a"]}).to_parquet(base_val, index=False)
    c = compare_part_to_baseline(p, str(h), "val", base_val, target="t")
    assert c.f1 == 1.0 and c.baseline_f1 < 1.0 and c.noninferior
    with pytest.raises(ValueError):                          # Test-Baseline passt nicht zu val
        base_test = str(tmp_path / "btest.parquet")
        pd.DataFrame({"prediction": ["a"] * 6}).to_parquet(base_test, index=False)
        compare_part_to_baseline(p, str(h), "val", base_test, target="t")
