# tests/test_variance_scan.py
"""Der Varianz-Scan leitet das finale n pro Zelle ab (experiment-design.md §5.2).
Die Rechnung landet als begruendete Zahl in der Thesis — sie muss reproduzierbar
bleiben, auch wenn jemand am Script schraubt.
"""
from __future__ import annotations
import json
import math
import os

import pytest

from variance_scan import (build_plan, collect, required_n, run_id_for,
                           summarize, target_se)


def test_se_ziel_entspricht_dokumentierter_heuristik():
    # experiment-design.md §5.2: SE <= 1/3 des kleinsten Arm-Effekts (0.05)
    assert target_se() == pytest.approx(0.05 / 3)


def test_reproduziert_kalibrierungslauf_n6():
    """Die treibende Zelle credit_g x Naked ReAct hatte SD 0.0395 -> n = 6.
    Aendert sich diese Zahl, ist die publizierte n-Begruendung hinfaellig."""
    assert required_n(0.0395) == 6
    # Nachbarwerte aus der dokumentierten Rechnung: n=5 knapp drueber, n=7 konservativ
    assert 0.0395 / math.sqrt(5) > target_se()
    assert 0.0395 / math.sqrt(6) <= target_se()


@pytest.mark.parametrize("sd", [0.001, 0.0036, 0.02, 0.0395, 0.06, 0.15])
def test_erfuellt_se_bedingung(sd):
    n = required_n(sd)
    assert sd / math.sqrt(n) <= target_se() + 1e-12
    if n > 1:  # n ist minimal — eins weniger reisst die Bedingung
        assert sd / math.sqrt(n - 1) > target_se()


def test_bit_stabile_zelle_fordert_kein_n():
    # covertype x Hard-FSM lieferte 3x bit-identische F1 (SD 0.0)
    assert required_n(0.0) == 1


def test_plan_faehrt_arm_d_breit_und_arm_a_als_stichprobe():
    plan = build_plan(["a", "b", "c"], replicates=3, arm_a_datasets=["b"])
    assert len(plan) == 3 * 3 + 1 * 3
    d = [j for j in plan if j["arm"] == "naked_react"]
    a = [j for j in plan if j["arm"] == "hard_fsm"]
    assert {j["dataset"] for j in d} == {"a", "b", "c"}
    assert {j["dataset"] for j in a} == {"b"}
    # Replikate sind durchnummeriert, nicht dupliziert
    assert sorted(j["replicate"] for j in a) == [1, 2, 3]


def test_run_id_ist_deterministisch_und_kollisionsfrei():
    """Resume haengt daran: derselbe Job muss dieselbe id ergeben."""
    plan = build_plan(["a", "b"], replicates=2, arm_a_datasets=["a"])
    ids = [run_id_for(j) for j in plan]
    assert len(set(ids)) == len(ids)
    assert run_id_for(plan[0]) == run_id_for(dict(plan[0]))


def _write_summary(runs_dir, job, **fields):
    d = os.path.join(runs_dir, run_id_for(job))
    os.makedirs(d, exist_ok=True)
    payload = {"outcome": "SUCCESS", "final_f1": 0.9, "holdout_intact": True,
               "total_cost_usd": 0.35, **fields}
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(payload, f)


def test_sammelt_f1_pro_zelle_und_rechnet_sd(tmp_path):
    runs = str(tmp_path)
    plan = build_plan(["credit_g"], replicates=3, arm_a_datasets=[])
    for job, f1 in zip(plan, [0.6879, 0.7562, 0.7562]):  # echte Kalibrierungswerte
        _write_summary(runs, job, final_f1=f1)

    row = summarize(collect(plan, runs))[0]
    assert row["n_f1"] == 3
    assert row["f1_mean"] == pytest.approx(0.7334, abs=1e-4)
    assert row["f1_sd"] == pytest.approx(0.0395, abs=1e-4)
    assert row["n_required"] == 6


def test_kontaminierte_und_fehlende_runs_fliegen_aus_der_varianz(tmp_path):
    """holdout_intact=False heisst: Messung unbrauchbar (run.py warnt bereits).
    So ein Run darf die SD nicht mitpraegen, muss aber sichtbar bleiben."""
    runs = str(tmp_path)
    plan = build_plan(["wine"], replicates=3, arm_a_datasets=[])
    _write_summary(runs, plan[0], final_f1=0.90)
    _write_summary(runs, plan[1], final_f1=0.92)
    _write_summary(runs, plan[2], final_f1=0.10, holdout_intact=False)

    row = summarize(collect(plan, runs))[0]
    assert row["n_runs"] == 3          # alle drei gelaufen
    assert row["n_f1"] == 2            # nur zwei zaehlen in die Varianz
    assert row["f1_values"] == [0.90, 0.92]
    assert row["missing"] == 0


def test_abstain_ohne_f1_zaehlt_als_lauf_nicht_als_messwert(tmp_path):
    runs = str(tmp_path)
    plan = build_plan(["cmc"], replicates=2, arm_a_datasets=[])
    _write_summary(runs, plan[0], final_f1=0.5)
    _write_summary(runs, plan[1], outcome="ABSTAIN", final_f1=None)

    row = summarize(collect(plan, runs))[0]
    assert row["n_runs"] == 2 and row["n_f1"] == 1
    assert "ABSTAIN" in row["outcomes"]
    assert row["f1_sd"] == 0.0  # ein Wert -> keine Streuung ableitbar


def test_error_summaries_gelten_als_nicht_gemessen(tmp_path):
    """ERROR (run.py, Decker 14.09.) = Harness-Crash, kein Agent-Befund: zaehlt
    weder als Lauf noch als Messwert, und Resume faehrt den Run neu."""
    from variance_scan import is_measured, summary_path
    runs = str(tmp_path)
    plan = build_plan(["phoneme"], replicates=2, arm_a_datasets=[])
    _write_summary(runs, plan[0], final_f1=0.85)
    _write_summary(runs, plan[1], outcome="ERROR", final_f1=None,
                   fail_reason="crash", error={"reason": "crash", "type": "RuntimeError"})

    row = summarize(collect(plan, runs))[0]
    assert row["n_runs"] == 1 and row["n_f1"] == 1
    assert row["missing"] == 1
    assert "ERROR" not in row["outcomes"]
    assert is_measured(summary_path(runs, plan[0])) is True
    assert is_measured(summary_path(runs, plan[1])) is False
    assert is_measured(summary_path(runs, {**plan[1], "replicate": 99})) is False


def test_runner_wartet_bei_rate_limit_und_wiederholt_denselben_job(monkeypatch, tmp_path):
    """Abo-Kontingent leer (run.py Exit 75): nicht als Fehlschlag zählen, nicht
    zum nächsten Job springen — warten und exakt diesen Run wiederholen."""
    from variance_scan import execute_with_rate_limit_wait, EXIT_RATE_LIMITED
    job = {"arm": "naked_react", "dataset": "wine", "replicate": 1}
    codes = iter([EXIT_RATE_LIMITED, EXIT_RATE_LIMITED, 0])
    calls, sleeps = [], []

    def fake_execute(job_, runs_dir, model, manifests_dir, timeout, log_dir, run_id=None):
        calls.append(job_)
        rc = next(codes)
        if rc == EXIT_RATE_LIMITED:
            return False, f"exit {rc}", rc
        _write_summary(runs_dir, job_, final_f1=0.9)
        return True, "ok", 0

    monkeypatch.setattr("variance_scan.execute", fake_execute)
    monkeypatch.setattr("variance_scan.time.sleep", lambda s: sleeps.append(s))
    ok, note = execute_with_rate_limit_wait(job, str(tmp_path), "m", "manifests", 10,
                                            str(tmp_path / "logs"), wait_s=600)
    assert ok is True
    assert len(calls) == 3 and all(c == job for c in calls)
    assert sleeps == [600, 600]
    assert "rate_limited" in note


def test_noch_nicht_gelaufene_runs_werden_als_offen_gemeldet(tmp_path):
    runs = str(tmp_path)
    plan = build_plan(["adult"], replicates=3, arm_a_datasets=[])
    _write_summary(runs, plan[0], final_f1=0.8)

    row = summarize(collect(plan, runs))[0]
    assert row["missing"] == 2
    assert row["n_runs"] == 1
