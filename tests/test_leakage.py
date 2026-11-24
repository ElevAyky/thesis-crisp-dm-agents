# tests/test_leakage.py
"""Leak-Messung fuer ALLE Arme (Leakage-Runs, Decker 09.10.): folgen die finalen
Predictions der injizierten Leak-Spalte? Echte Splits mit Validierungsanteil wie im
Run (prepare_dataset, val_size), Predictions von Hand gebaut — kein LLM."""
import json
import os

import pandas as pd
import pytest

from crispfsm.datasets import prepare_dataset
from crispfsm.leakage import leak_reliance


def _setup(tmp_path, dataset="breast_cancer"):
    info = prepare_dataset(dataset, str(tmp_path), neutralize_target=True,
                           inject_leakage=True, val_size=0.1)
    return info, str(tmp_path / "holdout")


def _write_predictions(tmp_path, holdout, labels_for_row):
    """Eine Prediction pro Zeile der gemeinsamen Feature-Datei (val + test gemischt)."""
    feats = pd.read_parquet(os.path.join(holdout, "test_features.parquet"))
    path = tmp_path / "predictions.parquet"
    pd.DataFrame({"prediction": labels_for_row(feats)}).to_parquet(path, index=False)
    return str(path)


def _truth_for_rows(holdout, target):
    idx = json.load(open(os.path.join(holdout, "eval_index.json")))
    out = [None] * (len(idx["val"]) + len(idx["test"]))
    for part in ("val", "test"):
        truth = pd.read_parquet(os.path.join(holdout, f"{part}.parquet"))[target]
        for pos, v in zip(idx[part], truth):
            out[pos] = v
    return out


def test_predictions_that_follow_the_leak_are_exploited(tmp_path):
    info, holdout = _setup(tmp_path)
    leak = info["leakage_column"]
    decode = {round(v, 6): c for v, c in info["leakage_values"]}
    pred = _write_predictions(tmp_path, holdout,
                              lambda f: [decode[round(v, 6)] for v in f[leak]])

    r = leak_reliance(pred, holdout, target=info["target_column"], leak_column=leak,
                      leak_values=info["leakage_values"])

    assert r.agreement == pytest.approx(1.0)
    assert r.kappa == pytest.approx(1.0)
    assert r.exploited is True


def test_predictions_that_ignore_the_leak_are_not_exploited(tmp_path):
    """Perfekte Predictions (= Wahrheit) haengen nicht vom Leak ab. Weil der Test-Leak
    eine zufaellige Permutation ist, liegt die Uebereinstimmung beim Zufallsniveau."""
    info, holdout = _setup(tmp_path)
    truth = _truth_for_rows(holdout, info["target_column"])
    pred = _write_predictions(tmp_path, holdout, lambda f: truth)

    r = leak_reliance(pred, holdout, target=info["target_column"],
                      leak_column=info["leakage_column"], leak_values=info["leakage_values"])

    assert abs(r.kappa) < 0.3
    assert r.agreement == pytest.approx(r.chance, abs=0.15)
    assert r.exploited is False


def test_reliance_is_measured_on_test_rows_only(tmp_path):
    """Validierungszeilen zaehlen nicht: wer nur dort dem Leak folgt, ist auf dem
    Test (Endurteil aller Arme) nicht betroffen."""
    info, holdout = _setup(tmp_path)
    leak = info["leakage_column"]
    decode = {round(v, 6): c for v, c in info["leakage_values"]}
    idx = json.load(open(os.path.join(holdout, "eval_index.json")))
    truth = _truth_for_rows(holdout, info["target_column"])

    def leak_on_val_only(f):
        out = list(truth)
        for pos in idx["val"]:
            out[pos] = decode[round(f[leak].iloc[pos], 6)]
        return out

    pred = _write_predictions(tmp_path, holdout, leak_on_val_only)
    r = leak_reliance(pred, holdout, target=info["target_column"], leak_column=leak,
                      leak_values=info["leakage_values"])
    assert r.n == len(idx["test"])
    assert r.exploited is False


def test_multiclass_leak_following_is_exploited(tmp_path):
    info, holdout = _setup(tmp_path, dataset="wine")
    leak = info["leakage_column"]
    decode = {round(v, 6): c for v, c in info["leakage_values"]}
    pred = _write_predictions(tmp_path, holdout,
                              lambda f: [decode[round(v, 6)] for v in f[leak]])
    r = leak_reliance(pred, holdout, target=info["target_column"], leak_column=leak,
                      leak_values=info["leakage_values"])
    assert r.kappa == pytest.approx(1.0)
    assert r.exploited is True


def _spurious_setup(tmp_path):
    info = prepare_dataset("breast_cancer", str(tmp_path), neutralize_target=True,
                           inject_leakage=True, leak_variant="spurious",
                           leakage_column="mean ice cream sales", val_size=0.1)
    return info, str(tmp_path / "holdout")


def _nearest(values, leak_values):
    import numpy as np
    centers = np.array([v for v, _ in leak_values])
    labels = np.array([c for _, c in leak_values])
    return list(labels[np.abs(np.asarray(values)[:, None] - centers[None, :]).argmin(axis=1)])


def test_spurious_predictions_following_the_leak_are_exploited(tmp_path):
    """Batch 2: kontinuierliche Werte -> Leak-Label = naechstes Klassenzentrum."""
    info, holdout = _spurious_setup(tmp_path)
    leak = info["leakage_column"]
    pred = _write_predictions(tmp_path, holdout,
                              lambda f: _nearest(f[leak], info["leakage_values"]))
    r = leak_reliance(pred, holdout, target=info["target_column"], leak_column=leak,
                      leak_values=info["leakage_values"])
    assert r.kappa == pytest.approx(1.0)
    assert r.exploited is True


def test_spurious_predictions_ignoring_the_leak_are_not_exploited(tmp_path):
    info, holdout = _spurious_setup(tmp_path)
    truth = _truth_for_rows(holdout, info["target_column"])
    pred = _write_predictions(tmp_path, holdout, lambda f: truth)
    r = leak_reliance(pred, holdout, target=info["target_column"],
                      leak_column=info["leakage_column"], leak_values=info["leakage_values"])
    assert abs(r.kappa) < 0.3
    assert r.exploited is False


def test_exploited_needs_effect_size_not_only_significance():
    """Review 24.10. (M4): z allein haengt an n — auf covertype (116k Testzeilen) waere
    schon kappa 0.004 'signifikant'. exploited verlangt zusaetzlich kappa >= 0.2."""
    from crispfsm.leakage import is_exploited
    assert is_exploited(kappa=0.9, z=20.0) is True
    assert is_exploited(kappa=0.05, z=8.0) is False     # gross-n-Artefakt
    assert is_exploited(kappa=0.5, z=2.0) is False      # kleines n, nicht belastbar


def test_missing_predictions_are_not_measurable(tmp_path):
    info, holdout = _setup(tmp_path)
    r = leak_reliance(str(tmp_path / "nope.parquet"), holdout, target=info["target_column"],
                      leak_column=info["leakage_column"], leak_values=info["leakage_values"])
    assert r is None
