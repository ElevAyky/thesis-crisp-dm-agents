# tests/test_analyze_h1.py
"""Auswertung H1-Nachtrag (scripts/analyze_h1.py, docs/h1-single-session.md §4)."""
from __future__ import annotations
import json
import os

import numpy as np
import pytest
from scipy.stats import wilcoxon

from analyze_h1 import (A, A_OLD, S, core_analysis, post_cutoff, run_analysis,
                        signed_rank_test)

DS = [f"ds{i:02d}" for i in range(13)]


def _cell(arm, ds, f1s, success=None, cost=6.0, calls=5.0, missing=0):
    n = len(f1s)
    success = n if success is None else success
    return {"arm": arm, "dataset": ds, "n_measured": n, "missing": missing, "contaminated": 0,
            "outcomes": {"SUCCESS": success, **({"FAIL": n - success} if n - success else {})},
            "fail_reasons": {}, "f1_measured": list(f1s), "f1_on_success": list(f1s[:success]),
            "f1_combined": list(f1s[:success]) + [0.0] * (n - success),
            "f1_measured_sd": 0.001, "calls_mean": calls, "cost_total": cost}


def _suite(arm, base, step=0.01, jitter=0.0):
    return [_cell(arm, d, [base + i * step + i * jitter] * 6) for i, d in enumerate(DS)]


# --- Test-Statistik ------------------------------------------------------------

def test_signed_rank_all_zero_is_no_difference():
    t = signed_rank_test([0.8] * 13, [0.8] * 13)
    assert t["p"] == 1.0 and t["ties"] == 13


def test_signed_rank_matches_scipy_exact_without_zeros():
    rng = np.random.default_rng(0)
    x = rng.normal(0.8, 0.01, 13)
    y = x + rng.normal(0.002, 0.004, 13)
    ours = signed_rank_test(x.tolist(), y.tolist())["p"]
    ref = wilcoxon(x, y, method="exact").pvalue
    assert ours == pytest.approx(ref, abs=1e-12)


def test_signed_rank_with_zeros_is_conditional_permutation():
    """7 Null-Differenzen + 6 positive: exakt 2/2^6 (beide Extreme), unabhaengig
    davon, wie scipy die Null-Differenzen behandelt (Review 13.11.)."""
    x = [0.8] * 7 + [0.81, 0.82, 0.83, 0.84, 0.85, 0.86]
    t = signed_rank_test(x, [0.8] * 13)
    assert t["p"] == pytest.approx(2 / 64)
    assert t["r_plus"] > t["r_minus"] and t["median_diff"] == 0.0


# --- Kern -----------------------------------------------------------------------

def test_core_identical_arms_not_supported():
    res = core_analysis(_suite("hard_fsm", 0.8), _suite(A, 0.8) + _suite(S, 0.8))
    assert res["verdicts"] == {"H1a": "nicht gestuetzt", "H1b": "nicht gestuetzt"}
    assert res["primary"]["rates"][A]["success"] == res["primary"]["rates"][S]["success"] == 78
    assert res["primary"]["metrics"]["f1_measured"]["datasets"] == DS


def test_core_a_better_is_supported_even_with_median_zero():
    """Richtung kommt aus R+/R-, nicht aus dem Median (Review 13.11.)."""
    a = [_cell(A, d, [0.8] * 6) if i < 7 else _cell(A, d, [0.8 + i / 100] * 6)
         for i, d in enumerate(DS)]
    s = [_cell(S, d, [0.8] * 6) for d in DS]
    res = core_analysis(_suite("hard_fsm", 0.8), a + s)
    t = res["primary"]["metrics"]["f1_measured"]["test"]
    assert t["median_diff"] == 0.0 and t["p"] < 0.05
    assert res["verdicts"]["H1b"] == "gestuetzt"


def test_core_single_better_is_reversed():
    res = core_analysis(_suite("hard_fsm", 0.8),
                        _suite(A, 0.80, jitter=0.0001) + _suite(S, 0.85))
    assert res["verdicts"]["H1b"] == "umgekehrt"


def test_core_pass_rate_is_its_own_hypothesis():
    s = [_cell(S, d, [0.8 + i * 0.01] * 6, success=3) for i, d in enumerate(DS)]   # 39/78
    res = core_analysis(_suite("hard_fsm", 0.8), _suite(A, 0.8) + s)
    assert res["verdicts"]["H1a"] == "gestuetzt"
    assert res["verdicts"]["H1b"] == "nicht gestuetzt"             # f1_measured gleich


def test_core_primary_uses_concurrent_a_not_october_a():
    """Primaer = A aus dem H1-Batch; A(Okt.) nur ergaenzend + Replikation."""
    res = core_analysis(_suite("hard_fsm", 0.90), _suite(A, 0.8) + _suite(S, 0.8))
    assert res["verdicts"]["H1b"] == "nicht gestuetzt"
    assert res["old_vs_single"]["f1_direction"] == "x vorn"
    assert res["replication"]["f1_direction"] == "x vorn"


def test_core_ignores_other_arms_in_main_file():
    main = _suite("hard_fsm", 0.8) + _suite("naked_react", 0.1)
    res = core_analysis(main, _suite(A, 0.8) + _suite(S, 0.8))
    assert res["old_vs_single"]["rates"][A_OLD]["n"] == 78


def test_core_incomplete_data_gives_no_verdict():
    """Review 13.11.: 6 von 13 Datasets mit 2 Runs ergaben sonst 'gestuetzt'."""
    a = [_cell(A, d, [0.85] * 2) for d in DS[:6]]
    s = [_cell(S, d, [0.80] * 2) for d in DS[:6]]
    res = core_analysis(_suite("hard_fsm", 0.8), a + s)
    assert res["verdicts"]["H1b"].startswith("vorlaeufig")
    assert not res["primary"]["completeness"]["complete"]


def test_core_missing_runs_give_no_verdict():
    s = _suite(S, 0.8)
    s[3] = _cell(S, DS[3], [0.8] * 6, missing=1)
    res = core_analysis(_suite("hard_fsm", 0.8), _suite(A, 0.8) + s)
    assert res["verdicts"]["H1a"].startswith("vorlaeufig")


# --- Post-Cutoff ----------------------------------------------------------------

def test_post_cutoff_keeps_old_and_new_a_apart():
    old = [_cell("hard_fsm", "ev", [0.82] * 6), _cell("naked_react", "ev", [0.80] * 6, success=4)]
    new = [_cell(A, "ev", [0.821] * 6, success=5), _cell(S, "ev", [0.82] * 6)]
    pc = post_cutoff(old, new)
    assert [r["arm"] for r in pc["rows"]] == [A_OLD, A, S, "naked_react"]
    assert pc["pooled"][A_OLD]["success"] == 6 and pc["pooled"][A]["success"] == 5
    assert pc["f1_drift_new_minus_old"]["ev"] == pytest.approx(0.001)
    assert pc["missing_arms"] == []
    assert old[0]["arm"] == "hard_fsm" and new[0]["arm"] == A    # Eingaben unveraendert


def test_post_cutoff_flags_missing_arm():
    pc = post_cutoff([_cell("hard_fsm", "ev", [0.82] * 6)], [_cell(S, "ev", [0.82] * 6)])
    assert pc["missing_arms"] == [A]


# --- Ende-zu-Ende ---------------------------------------------------------------

def test_run_analysis_writes_report(tmp_path):
    for sub, cells in (("_main", _suite("hard_fsm", 0.8)),
                       ("_h1", _suite(A, 0.8) + _suite(S, 0.8)),
                       ("_post_cutoff", [_cell("hard_fsm", "ev", [0.82] * 6)]),
                       ("_post_cutoff_h1", [_cell(A, "ev", [0.82] * 6), _cell(S, "ev", [0.82] * 6)])):
        os.makedirs(tmp_path / sub)
        (tmp_path / sub / "main_runs.json").write_text(json.dumps({"cells": cells}))
    res = run_analysis(str(tmp_path))
    assert res["core"]["verdicts"]["H1b"] == "nicht gestuetzt"
    report = (tmp_path / "_h1" / "analysis" / "report.md").read_text(encoding="utf-8")
    assert "H1a (Bestanden): nicht gestuetzt" in report and "A-Single" in report
    assert json.loads((tmp_path / "_h1" / "analysis" / "analysis.json").read_text())["agg"] == "mean"


def test_run_analysis_without_h1_results_does_not_crash(tmp_path):
    os.makedirs(tmp_path / "_main")
    (tmp_path / "_main" / "main_runs.json").write_text(json.dumps({"cells": []}))
    res = run_analysis(str(tmp_path))
    assert "core" not in res and "post_cutoff" not in res
