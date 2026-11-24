# tests/test_analyze_main.py
"""Auswertungs-Script (experiment-design.md §5): Friedman + Nemenyi + CD-Diagramm,
Cliff's Delta, Wilson-KIs, Fisher's Exact. Referenzwerte aus Demšar 2006
(Tabelle 5a) bzw. geschlossene Formeln — keine echten Runs."""
from __future__ import annotations
import json
import math

import numpy as np
import pytest
from scipy.stats import friedmanchisquare

from analyze_main import (cd_cliques, cell_matrix, cell_values, cliffs_delta,
                          cliffs_magnitude, fisher_success, friedman,
                          nemenyi_cd, nemenyi_pairs, rank_matrix, run_analysis,
                          wilson)


def _cell(arm, ds, success, f1s_success, f1s_measured, **kw):
    """Minimal-Zelle im main_runs.json-Format."""
    n = len(success)
    return {"arm": arm, "dataset": ds, "n_measured": n, "missing": 0,
            "contaminated": 0,
            "outcomes": {"SUCCESS": sum(success), "FAIL": n - sum(success)},
            "success_rate": sum(success) / n if n else None,
            "f1_on_success": f1s_success, "f1_measured": f1s_measured,
            "f1_combined": [f if ok else 0.0 for ok, f in zip(success, f1s_measured)],
            **kw}


# --- Nemenyi / CD -----------------------------------------------------------

def test_cd_matches_demsar_table_and_roadmap():
    """Demšar Tab. 5a: q_0.05(k=4) = 2.569. Roadmap: CD ≈ 1.30 bei k=4, N=13."""
    cd = nemenyi_cd(k=4, n_datasets=13, alpha=0.05)
    assert cd == pytest.approx(2.569 * math.sqrt(4 * 5 / (6 * 13)), abs=1e-3)
    assert cd == pytest.approx(1.30, abs=0.005)


def test_cd_shrinks_with_more_datasets():
    assert nemenyi_cd(4, 20) < nemenyi_cd(4, 13) < nemenyi_cd(4, 8)


def test_nemenyi_pairs_flags_only_differences_above_cd():
    pairs = nemenyi_pairs({"A": 1.2, "B": 1.9, "C": 3.0, "D": 3.9}, cd=1.3)
    sig = {(p["a"], p["b"]) for p in pairs if p["significant"]}
    assert sig == {("A", "C"), ("A", "D"), ("B", "D")}


def test_cd_cliques_are_maximal_groups_within_cd():
    cliques = cd_cliques({"A": 1.2, "B": 1.9, "C": 3.0, "D": 3.9}, cd=1.3)
    assert cliques == [["A", "B"], ["B", "C"], ["C", "D"]]


def test_cd_cliques_drop_subsets():
    cliques = cd_cliques({"A": 1.0, "B": 1.5, "C": 2.0, "D": 4.0}, cd=1.3)
    assert cliques == [["A", "B", "C"]]              # D allein -> keine Linie


# --- Ranks / Friedman -------------------------------------------------------

def test_rank_matrix_best_f1_gets_rank_1_and_ties_are_averaged():
    r = rank_matrix(np.array([[0.9, 0.9, 0.8, 0.7]]))
    assert r.tolist() == [[1.5, 1.5, 3.0, 4.0]]


def test_friedman_matches_scipy_without_ties():
    rng = np.random.default_rng(0)
    M = rng.random((13, 4))
    res = friedman(M)
    ref = friedmanchisquare(*M.T)
    assert res["chi2"] == pytest.approx(ref.statistic)
    assert res["p_chi2"] == pytest.approx(ref.pvalue)
    # Iman-Davenport (Demšar Gl. 2): F = (N-1)chi2 / (N(k-1) - chi2)
    assert res["F"] == pytest.approx(12 * res["chi2"] / (13 * 3 - res["chi2"]))


def test_friedman_detects_consistent_ordering():
    M = np.tile([0.9, 0.8, 0.7, 0.6], (13, 1))
    res = friedman(M)
    assert res["avg_ranks"] == pytest.approx([1, 2, 3, 4])
    assert res["p_F"] < 1e-6


# --- Zellen-Matrix ----------------------------------------------------------

def test_cell_matrix_drops_incomplete_blocks():
    """F1-auf-Erfolg: Zelle ohne einen SUCCESS hat keinen Wert -> ganzes Dataset
    raus (Friedman braucht vollstaendige Bloecke), aber dokumentiert."""
    cells = [_cell("A", "x", [1, 1], [0.9, 0.8], [0.9, 0.8]),
             _cell("B", "x", [1, 0], [0.7], [0.7, 0.5]),
             _cell("A", "y", [1], [0.9], [0.9]),
             _cell("B", "y", [0], [], [0.6])]
    ds, M, dropped = cell_matrix(cells, ["A", "B"], "f1_on_success")
    assert ds == ["x"] and dropped == ["y"]
    assert M.tolist() == [[pytest.approx(0.85), pytest.approx(0.7)]]
    ds, M, dropped = cell_matrix(cells, ["A", "B"], "f1_measured", agg="median")
    assert ds == ["x", "y"] and dropped == []


def test_combined_falls_back_for_old_main_runs_json():
    """main_runs.json vor 23.09. hatte keine f1_combined-Liste -> aus
    f1_on_success + Nullen fuer Nicht-SUCCESS rekonstruieren."""
    c = _cell("A", "x", [1, 0, 1], [0.9, 0.8], [0.9, 0.5, 0.8])
    del c["f1_combined"]
    assert sorted(cell_values(c, "f1_combined")) == [0.0, 0.8, 0.9]


# --- Effektgroessen / Raten -------------------------------------------------

def test_cliffs_delta_extremes_and_symmetry():
    assert cliffs_delta([4, 5], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [4, 5]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert cliffs_delta([1, 3], [2]) == 0.0


def test_cliffs_magnitude_romano_thresholds():
    assert cliffs_magnitude(0.1) == "negligible"
    assert cliffs_magnitude(-0.2) == "small"
    assert cliffs_magnitude(0.4) == "medium"
    assert cliffs_magnitude(-0.6) == "large"


def test_wilson_edge_cases():
    lo, hi = wilson(0, 6)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(1.96**2 / (6 + 1.96**2), abs=1e-4)
    lo, hi = wilson(6, 6)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(1 - 1.96**2 / (6 + 1.96**2), abs=1e-4)
    assert wilson(0, 0) == (None, None)


def test_fisher_success_pools_over_datasets():
    cells = [_cell("A", "x", [1] * 6, [0.9] * 6, [0.9] * 6),
             _cell("A", "y", [1] * 6, [0.9] * 6, [0.9] * 6),
             _cell("B", "x", [0] * 6, [], [0.5] * 6),
             _cell("B", "y", [1, 0, 0, 0, 0, 0], [0.9], [0.9] + [0.5] * 5)]
    res = fisher_success(cells, "A", "B")
    assert res["table"] == [[12, 0], [1, 11]]
    assert res["p"] < 0.001


# --- End-to-end -------------------------------------------------------------

def test_run_analysis_writes_report_json_and_figures(tmp_path):
    rng = np.random.default_rng(1)
    arms = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
    cells = []
    for i in range(6):
        for j, arm in enumerate(arms):
            f = list(0.9 - 0.05 * j + rng.normal(0, 0.01, 3))
            cells.append(_cell(arm, f"ds{i}", [1, 1, 1], f, f, cost_total=1.0))
    src = tmp_path / "main_runs.json"
    src.write_text(json.dumps({"model": "m", "replicates": 3, "arms": arms,
                               "datasets": [f"ds{i}" for i in range(6)],
                               "cells": cells}))
    out = tmp_path / "analysis"
    result = run_analysis(str(src), str(out))
    assert (out / "report.md").exists() and (out / "analysis.json").exists()
    assert (out / "cd_f1_combined.png").exists()
    assert (out / "box_f1_measured.png").exists()
    f = result["metrics"]["f1_combined"]["friedman"]
    assert f["avg_ranks"] == pytest.approx([1, 2, 3, 4])
    assert "Friedman" in (out / "report.md").read_text(encoding="utf-8")


def test_run_analysis_with_fewer_than_three_arms_skips_friedman(tmp_path):
    """Zwischenstand (z. B. nur Arm D gefahren): kein Crash, Deskriptives ja,
    Friedman mit Begruendung ausgelassen."""
    cells = [_cell("naked_react", "x", [1, 0], [0.9], [0.9, 0.6])]
    src = tmp_path / "main_runs.json"
    src.write_text(json.dumps({"model": "m", "replicates": 2, "arms": ["naked_react"],
                               "datasets": ["x"], "cells": cells}))
    result = run_analysis(str(src), str(tmp_path / "a"))
    assert result["metrics"]["f1_combined"]["friedman"] is None
    assert "mindestens 3 Arme" in result["metrics"]["f1_combined"]["skipped"]
