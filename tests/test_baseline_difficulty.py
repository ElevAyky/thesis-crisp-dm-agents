# tests/test_baseline_difficulty.py
"""Baseline auf dem Run-Holdout (23.09.): die Erfolgsschwelle muss genauso
gemessen werden wie der Agent — gleicher Split (prepare_from_manifest, dieselbe
Funktion wie run.py), Fit auf train, Score auf holdout. Vorher: 5-fold-CV-Mittel
vs. Agent auf EINEM Holdout -> die Baseline bestand ihre eigene Schwelle nicht
(phoneme 0.878 < 0.890, cnae_9 0.940 < 0.942)."""
from __future__ import annotations
import pytest

import baseline_difficulty as bd
import crispfsm.datasets as ds
from crispfsm.manifests import load_manifest


def test_holdout_baseline_uses_the_run_split(monkeypatch):
    """Paritaet by construction: dieselbe Setup-Funktion wie run.py, ohne Leakage."""
    calls = []
    real = ds.prepare_from_manifest

    def spy(manifest, workdir, **kw):
        calls.append((manifest.dataset, kw))
        return real(manifest, workdir, **kw)

    monkeypatch.setattr(bd, "prepare_from_manifest", spy)
    r = bd.score_holdout("breast_cancer", "manifests")
    assert calls == [("breast_cancer", {"inject_leakage": False})]
    assert (r["n_train"], r["n_val"], r["n_test"]) == (398, 57, 114)   # 70/10/20 von 569
    assert len(r["predictions_val"]) == 57


def test_holdout_baseline_model_is_chosen_on_validation():
    """S1: Referenzmodell wird auf dem Validierungsanteil gewaehlt (keine Auswahl auf dem
    Test-Set, auch nicht fuer die Baseline); berichtet wird sein Test-F1."""
    r = bd.score_holdout("breast_cancer", "manifests")
    test_f1 = {"logreg": r["f1_logreg_holdout"], "rf": r["f1_rf_holdout"]}
    assert r["f1_best_holdout"] == test_f1[r["best_model"]]
    assert r["best_model"] in ("logreg", "rf")
    assert 0.9 < r["f1_best_val"] <= 1.0 and 0.9 < r["f1_best_holdout"] <= 1.0


def test_run_uses_prepare_from_manifest():
    """run.py darf den Split nicht selbst zusammenbauen — sonst driftet die
    Baseline-Messung wieder weg."""
    src = open("src/crispfsm/run.py", encoding="utf-8").read()
    assert "prepare_from_manifest(" in src
    assert "prepare_dataset(" not in src


YAML = """dataset: x
task_prompt: "t"
canonical_spec:
  task_type: classification
  target_column: y
  metric: f1
baseline_f1: 0.890   # alter CV-Wert
"""


def test_write_manifest_baseline_sets_holdout_and_keeps_cv(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text(YAML, encoding="utf-8")
    bd.write_manifest_baseline(str(p), holdout_f1=0.8781, cv_f1=0.8904, val_f1=0.8612)
    m = load_manifest("x", manifests_dir=str(tmp_path))
    assert m.baseline_f1 == pytest.approx(0.878)
    assert m.baseline_f1_cv == pytest.approx(0.890)
    assert m.baseline_f1_val == pytest.approx(0.861)
    text = p.read_text(encoding="utf-8")
    assert 'target_column: y' in text and text.count("baseline_f1:") == 1
    # idempotent: zweiter Lauf dupliziert nichts
    bd.write_manifest_baseline(str(p), holdout_f1=0.8781, cv_f1=0.8904, val_f1=0.8612)
    text2 = p.read_text(encoding="utf-8")
    assert text2 == text


def test_holdout_baseline_returns_predictions_of_best_model():
    """Die Predictions der besten Baseline sind der Vergleichspartner im gepaarten
    Bootstrap (scoring.compare_to_baseline) — eine pro Holdout-Zeile, als Labels."""
    from sklearn.metrics import f1_score as f1
    r = bd.score_holdout("breast_cancer", "manifests")
    assert len(r["predictions"]) == r["n_test"]
    assert set(r["predictions"]) <= {"malignant", "benign"}


def test_write_baseline_predictions(tmp_path):
    import pandas as pd
    path = bd.write_baseline_predictions(str(tmp_path), "x", ["a", "b", "a"])
    assert path == str(tmp_path / "baseline_preds" / "x.test.parquet")
    assert bd.write_baseline_predictions(str(tmp_path), "x", ["a"], "val") ==         str(tmp_path / "baseline_preds" / "x.val.parquet")
    df = pd.read_parquet(path)
    assert list(df.columns) == ["prediction"] and df["prediction"].tolist() == ["a", "b", "a"]
