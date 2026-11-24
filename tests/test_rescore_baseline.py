# tests/test_rescore_baseline.py
"""Offline-Neubewertung C/D nach der Bootstrap-Regel (23.09.): exakt, weil kein
Retry an der Schwelle hing; Originalwerte bleiben, Ergebnis als rescore-Block."""
from __future__ import annotations
import json
import os

import pandas as pd
import pytest

import rescore_baseline as rb
from main_runs import aggregate, build_plan, summarize

Y = ["a"] * 12 + ["b"] * 12 + ["c"] * 12          # 36 Zeilen, wine-artig


def _setup(tmp_path, run_id, agent_preds, summary, baseline=Y):
    mdir = tmp_path / "manifests"
    (mdir / "baseline_preds").mkdir(parents=True, exist_ok=True)
    (mdir / "toy.yaml").write_text(
        "dataset: toy\ntask_prompt: t\ncanonical_spec:\n  task_type: classification\n"
        "  target_column: y\n  metric: f1\nbaseline_f1: 1.000\n", encoding="utf-8")
    pd.DataFrame({"prediction": baseline}).to_parquet(mdir / "baseline_preds" / "toy.parquet",
                                                      index=False)
    run = tmp_path / "runs" / run_id
    (run / "holdout").mkdir(parents=True)
    (run / "work").mkdir()
    pd.DataFrame({"x": range(36), "col_1": Y}).to_parquet(run / "holdout" / "test.parquet",
                                                          index=False)
    if agent_preds is not None:
        pd.DataFrame({"prediction": agent_preds}).to_parquet(run / "work" / "predictions.parquet",
                                                             index=False)
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return str(run), str(mdir)


def test_one_error_below_threshold_becomes_success(tmp_path):
    agent = list(Y); agent[0] = "b"
    run, mdir = _setup(tmp_path, "main-toy-naked_react-r1", agent,
                       {"outcome": "FAIL", "success": False, "fail_reason": "below_threshold",
                        "final_f1": None})
    block = rb.rescore_run(run, mdir)
    assert block["outcome"] == "SUCCESS" and block["changed"] is True
    assert block["original"]["outcome"] == "FAIL"
    assert block["final_f1"] == pytest.approx(block["baseline_comparison"]["f1"])


def test_significantly_worse_success_becomes_fail(tmp_path):
    agent = ["a"] * 36
    run, mdir = _setup(tmp_path, "main-toy-crisp_react-r2", agent,
                       {"outcome": "SUCCESS", "success": True, "fail_reason": None, "final_f1": 0.3})
    block = rb.rescore_run(run, mdir)
    assert block["outcome"] == "FAIL" and block["fail_reason"] == "below_threshold"
    assert block["final_f1"] is None


def test_form_error_is_not_touched(tmp_path):
    run, mdir = _setup(tmp_path, "main-toy-naked_react-r3", None,
                       {"outcome": "FAIL", "success": False, "fail_reason": "form_error",
                        "final_f1": None})
    block = rb.rescore_run(run, mdir)
    assert block["changed"] is False and block["fail_reason"] == "form_error"


def test_fsm_arms_are_refused(tmp_path):
    run, mdir = _setup(tmp_path, "main-toy-hard_fsm-r1", Y,
                       {"outcome": "SUCCESS", "success": True, "final_f1": 1.0})
    with pytest.raises(ValueError, match="neu fahren"):
        rb.rescore_run(run, mdir)


def test_misaligned_baseline_is_refused(tmp_path):
    run, mdir = _setup(tmp_path, "main-toy-naked_react-r4", Y,
                       {"outcome": "SUCCESS", "success": True, "final_f1": 1.0},
                       baseline=list(reversed(Y)))
    with pytest.raises(ValueError, match="passen nicht"):
        rb.rescore_run(run, mdir)


def test_main_runs_aggregate_prefers_rescore(tmp_path):
    runs = tmp_path / "runs"
    plan = build_plan(["toy"], replicates=1, arms=["naked_react"])
    d = runs / "main-toy-naked_react-r1"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "outcome": "FAIL", "success": False, "fail_reason": "below_threshold", "final_f1": None,
        "best_f1_measured": 0.97, "holdout_intact": True,
        "rescore": {"outcome": "SUCCESS", "success": True, "fail_reason": None,
                    "final_f1": 0.97, "changed": True}}), encoding="utf-8")
    row = summarize(aggregate(plan, str(runs)))[0]
    assert row["success_rate"] == 1.0
    assert row["f1_on_success"] == [0.97] and row["f1_combined"] == [0.97]
    assert row["rescored"] == 1
