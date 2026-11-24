# tests/test_main_runs.py
"""Haupt-Runner (roadmap §4): 4 Arme x 13 Sets x n=6, replikat-weise balanciert,
resumebar, Auswertung nach fail_reason + best_f1_measured. Keine echten Runs —
die Ausfuehrung ist gemockt."""
from __future__ import annotations
import json
import os

import pytest

from main_runs import (ARMS, aggregate, build_plan, run_id_for, summarize,
                       suite_datasets)


def test_suite_datasets_are_the_13_curated_manifests():
    ds = suite_datasets("manifests")
    assert len(ds) == 13
    assert ds == sorted(ds)
    assert {"cmc", "credit_g", "wine", "covertype"} <= set(ds)


def test_plan_is_replicate_major_and_balanced():
    """Replikat aussen: nach r Replikaten hat JEDE Zelle genau r Runs — so ist
    ein Zwischenstand (05.10.) ueber alle Zellen gleich tief."""
    plan = build_plan(["a", "b"], replicates=3, arms=list(ARMS))
    assert len(plan) == 2 * 4 * 3
    reps = [j["replicate"] for j in plan]
    assert reps == sorted(reps)                      # 1,1,...,2,2,...,3,3,...
    first = plan[:8]
    assert {(j["arm"], j["dataset"]) for j in first} == {(a, d) for a in ARMS for d in ("a", "b")}


def test_plan_respects_arm_and_dataset_subsets():
    plan = build_plan(["cmc"], replicates=2, arms=["hard_fsm", "naked_react"])
    assert len(plan) == 4
    assert {j["arm"] for j in plan} == {"hard_fsm", "naked_react"}


def test_run_id_is_deterministic_and_distinct_from_scan_ids():
    job = {"arm": "hard_fsm", "dataset": "cmc", "replicate": 4}
    assert run_id_for(job) == run_id_for(dict(job)) == "main-cmc-hard_fsm-r4"
    assert not run_id_for(job).startswith("scan-")


def _write_summary(runs_dir, job, **fields):
    d = os.path.join(runs_dir, run_id_for(job))
    os.makedirs(d, exist_ok=True)
    payload = {"outcome": "SUCCESS", "success": True, "final_f1": 0.9,
               "best_f1_measured": 0.9, "fail_reason": None, "holdout_intact": True,
               "total_cost_usd": 0.5, "total_calls": 5, **fields}
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(payload, f)


def test_aggregate_counts_fail_reasons_and_keeps_full_f1_distribution(tmp_path):
    runs = str(tmp_path)
    plan = build_plan(["credit_g"], replicates=4, arms=["hard_fsm"])
    _write_summary(runs, plan[0], final_f1=0.70, best_f1_measured=0.70)
    _write_summary(runs, plan[1], outcome="FAIL", success=False, final_f1=None,
                   best_f1_measured=0.676, fail_reason="below_threshold")
    _write_summary(runs, plan[2], outcome="FAIL", success=False, final_f1=None,
                   best_f1_measured=None, fail_reason="form_error")
    # plan[3] fehlt noch

    row = summarize(aggregate(plan, runs))[0]
    assert row["arm"] == "hard_fsm" and row["dataset"] == "credit_g"
    assert row["n_measured"] == 3 and row["missing"] == 1
    assert row["success_rate"] == pytest.approx(1 / 3)
    assert row["f1_on_success"] == [0.70]
    assert row["f1_measured"] == [0.70, 0.676]            # Verteilung unabhaengig von der Schwelle
    assert row["f1_combined_mean"] == pytest.approx((0.70 + 0 + 0) / 3)  # Fail = 0
    assert row["fail_reasons"] == {"below_threshold": 1, "form_error": 1}
    assert row["cost_total"] == pytest.approx(1.5)


def test_aggregate_ignores_error_and_contaminated_runs(tmp_path):
    runs = str(tmp_path)
    plan = build_plan(["wine"], replicates=3, arms=["naked_react"])
    _write_summary(runs, plan[0], final_f1=0.95, best_f1_measured=0.95)
    _write_summary(runs, plan[1], outcome="ERROR", success=False, final_f1=None,
                   best_f1_measured=None, fail_reason="crash")
    _write_summary(runs, plan[2], final_f1=1.0, best_f1_measured=1.0, holdout_intact=False)

    row = summarize(aggregate(plan, runs))[0]
    assert row["n_measured"] == 1            # ERROR = nicht gemessen, kontaminiert = raus
    assert row["missing"] == 1               # der ERROR-Run ist noch zu fahren
    assert row["contaminated"] == 1
    assert row["f1_measured"] == [0.95]


def test_main_loop_runs_only_missing_jobs_and_uses_rate_limit_wait(tmp_path, monkeypatch):
    import main_runs
    runs = str(tmp_path / "runs")
    plan = build_plan(["wine"], replicates=2, arms=["naked_react", "hard_fsm"])
    _write_summary(runs, plan[0])                       # liegt schon vor
    started = []

    def fake_run(job, runs_dir, model, manifests_dir, timeout, log_dir, wait_s, run_id=None):
        started.append(run_id_for(job))
        _write_summary(runs_dir, job)
        return True, "ok"

    monkeypatch.setattr(main_runs, "execute_with_rate_limit_wait", fake_run)
    failures = main_runs.run_plan(plan, runs, "m", "manifests", timeout=10,
                                  log_dir=str(tmp_path / "logs"), wait_s=1, max_failures=0)
    assert failures == 0
    assert started == [run_id_for(j) for j in plan[1:]]


def test_main_loop_passes_main_run_id_to_subprocess(tmp_path, monkeypatch):
    """Regression 23.09.: der Runner delegierte an variance_scan.execute, das die
    run_id selbst baute (scan-<arm>-<set>-r<k>) — Summaries landeten unter
    scan-*, Resume/Aggregation suchten unter main-* und fanden nie etwas.
    Hier laeuft der echte Pfad bis zum Subprocess (nur subprocess.run gemockt)."""
    import subprocess
    import main_runs
    runs = str(tmp_path / "runs")
    plan = build_plan(["wine"], replicates=1, arms=["naked_react"])
    seen = []

    def fake_subprocess_run(cmd, **kw):
        rid = cmd[cmd.index("--run-id") + 1]
        seen.append(rid)
        os.makedirs(os.path.join(runs, rid), exist_ok=True)
        with open(os.path.join(runs, rid, "summary.json"), "w") as f:
            json.dump({"outcome": "SUCCESS", "success": True, "final_f1": 0.9,
                       "best_f1_measured": 0.9, "holdout_intact": True}, f)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("variance_scan.subprocess.run", fake_subprocess_run)
    failures = main_runs.run_plan(plan, runs, "m", "manifests", timeout=10,
                                  log_dir=str(tmp_path / "logs"), wait_s=1, max_failures=0)
    assert failures == 0
    assert seen == ["main-wine-naked_react-r1"]
    rows = summarize(aggregate(plan, runs))
    assert rows[0]["n_measured"] == 1 and rows[0]["missing"] == 0


# --- Leakage-Runs (Decker 09.10., docs/leakage-runs.md) ----------------------

def test_leakage_plan_has_own_run_ids_not_colliding_with_core():
    """Gleiche Zellen wie der Kern — ohne eigenes Praefix hielte der Resume die
    gemessenen main-*-Runs fuer erledigt und fuehre keinen einzigen Leakage-Run."""
    plan = build_plan(["cmc"], replicates=1, arms=["hard_fsm"], leakage=True)
    assert plan[0]["leakage"] is True
    assert run_id_for(plan[0]) == "leak-cmc-hard_fsm-r1"
    core = build_plan(["cmc"], replicates=1, arms=["hard_fsm"])
    assert run_id_for(core[0]) == "main-cmc-hard_fsm-r1"


def test_leakage_flag_reaches_the_subprocess(tmp_path, monkeypatch):
    import subprocess
    import main_runs
    runs = str(tmp_path / "runs")
    plan = build_plan(["wine"], replicates=1, arms=["naked_react"], leakage=True)
    seen = []

    def fake_subprocess_run(cmd, **kw):
        seen.append(cmd)
        rid = cmd[cmd.index("--run-id") + 1]
        os.makedirs(os.path.join(runs, rid), exist_ok=True)
        with open(os.path.join(runs, rid, "summary.json"), "w") as f:
            json.dump({"outcome": "SUCCESS", "success": True, "holdout_intact": True}, f)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("variance_scan.subprocess.run", fake_subprocess_run)
    main_runs.run_plan(plan, runs, "m", "manifests", timeout=10,
                       log_dir=str(tmp_path / "logs"), wait_s=1, max_failures=0)
    assert "--leakage" in seen[0]
    assert seen[0][seen[0].index("--run-id") + 1] == "leak-wine-naked_react-r1"


def test_core_plan_does_not_pass_leakage(tmp_path, monkeypatch):
    import subprocess
    import main_runs
    runs = str(tmp_path / "runs")
    seen = []

    def fake_subprocess_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr("variance_scan.subprocess.run", fake_subprocess_run)
    main_runs.run_plan(build_plan(["wine"], 1, ["naked_react"]), runs, "m", "manifests",
                       timeout=10, log_dir=str(tmp_path / "logs"), wait_s=1, max_failures=0)
    assert "--leakage" not in seen[0]


def test_leakage_output_dir_is_separate():
    from main_runs import out_subdir
    assert out_subdir("manifests", leakage=True) == "_leakage"
    assert out_subdir("manifests/post_cutoff", leakage=True) == "_post_cutoff_leakage"


def test_aggregate_counts_leak_exploitation_and_du_detection(tmp_path):
    runs = str(tmp_path)
    plan = build_plan(["credit_g"], replicates=3, arms=["hard_fsm"], leakage=True)
    _write_summary(runs, plan[0], leakage_exploited=False, leakage_detected=True,
                   leakage_reliance={"kappa": 0.02})
    _write_summary(runs, plan[1], outcome="FAIL", success=False, final_f1=None,
                   fail_reason="below_threshold", leakage_exploited=True,
                   leakage_detected=False, leakage_reliance={"kappa": 0.97})
    _write_summary(runs, plan[2], leakage_exploited=False, leakage_detected=None,
                   leakage_reliance={"kappa": -0.01})

    row = summarize(aggregate(plan, runs))[0]
    assert row["leak_exploited"] == [False, True, False]
    assert row["leak_exploited_rate"] == pytest.approx(1 / 3)
    assert row["leak_detected_rate"] == pytest.approx(1 / 2)   # None = nicht messbar
    assert row["leak_kappa"] == [0.02, 0.97, -0.01]


def test_cli_leakage_dry_run_plans_n3_over_the_core_suite(tmp_path, monkeypatch, capsys):
    """Design (docs/leakage-runs.md): 4 Arme x 13 Sets x n = 3 = 156 Runs."""
    import main_runs
    monkeypatch.setattr("sys.argv", ["prog", "--dry-run", "--leakage",
                                     "--runs-dir", str(tmp_path)])
    main_runs.main()
    out = capsys.readouterr().out
    assert "Plan: 156 Runs" in out and "n=3" in out and "Leakage" in out


def test_cli_core_dry_run_keeps_n6(tmp_path, monkeypatch, capsys):
    import main_runs
    monkeypatch.setattr("sys.argv", ["prog", "--dry-run", "--runs-dir", str(tmp_path)])
    main_runs.main()
    assert "Plan: 312 Runs" in capsys.readouterr().out


def test_table_shows_leak_rates_only_for_leakage_cells(tmp_path, capsys):
    from main_runs import print_table
    runs = str(tmp_path)
    plan = build_plan(["cmc"], replicates=2, arms=["hard_fsm"], leakage=True)
    _write_summary(runs, plan[0], leakage_exploited=True, leakage_detected=False)
    _write_summary(runs, plan[1], leakage_exploited=False, leakage_detected=True)
    print_table(summarize(aggregate(plan, runs)))
    out = capsys.readouterr().out
    assert "leak 1/2" in out and "DU 1/2" in out

    core = build_plan(["cmc"], replicates=1, arms=["hard_fsm"])
    _write_summary(runs, core[0])
    print_table(summarize(aggregate(core, runs)))
    assert "leak" not in capsys.readouterr().out


def test_spurious_plan_has_own_run_ids_and_output_dir():
    """Batch 2 (docs/leakage-runs.md §7) faehrt dieselben Zellen wie Batch 1 -> eigenes
    Praefix, sonst gaelten die leak-*-Runs als erledigt."""
    from main_runs import out_subdir
    plan = build_plan(["cmc"], replicates=1, arms=["hard_fsm"], leakage="spurious")
    assert plan[0]["leakage_variant"] == "spurious"
    assert run_id_for(plan[0]) == "spur-cmc-hard_fsm-r1"
    perfect = build_plan(["cmc"], replicates=1, arms=["hard_fsm"], leakage="perfect")
    assert run_id_for(perfect[0]) == "leak-cmc-hard_fsm-r1"
    assert out_subdir("manifests", leakage="spurious") == "_spurious"
    assert out_subdir("manifests", leakage="perfect") == "_leakage"


def test_spurious_variant_reaches_the_subprocess(tmp_path, monkeypatch):
    import subprocess
    import main_runs
    runs = str(tmp_path / "runs")
    seen = []

    def fake_subprocess_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr("variance_scan.subprocess.run", fake_subprocess_run)
    main_runs.run_plan(build_plan(["wine"], 1, ["naked_react"], leakage="spurious"), runs, "m",
                       "manifests", timeout=10, log_dir=str(tmp_path / "logs"), wait_s=1,
                       max_failures=0)
    cmd = seen[0]
    assert "--leakage" in cmd and cmd[cmd.index("--leakage-variant") + 1] == "spurious"


def test_cli_spurious_dry_run(tmp_path, monkeypatch, capsys):
    import main_runs
    monkeypatch.setattr("sys.argv", ["prog", "--dry-run", "--leakage", "spurious",
                                     "--arms", "naked_react", "--runs-dir", str(tmp_path)])
    main_runs.main()
    out = capsys.readouterr().out
    assert "Plan: 39 Runs" in out and "spurious" in out and "_spurious" in out


def test_core_aggregate_has_no_leak_rates(tmp_path):
    runs = str(tmp_path)
    plan = build_plan(["wine"], replicates=1, arms=["naked_react"])
    _write_summary(runs, plan[0])
    row = summarize(aggregate(plan, runs))[0]
    assert row["leak_exploited_rate"] is None and row["leak_detected_rate"] is None


def test_out_subdir_follows_manifests_dir():
    """Post-Cutoff-Lauf (manifests/post_cutoff) darf die Kern-Aggregation in
    _main/ nicht ueberschreiben."""
    from main_runs import out_subdir
    assert out_subdir("manifests") == "_main"
    assert out_subdir("manifests/") == "_main"
    assert out_subdir("manifests/post_cutoff") == "_post_cutoff"
    assert out_subdir(r"manifests\post_cutoff") == "_post_cutoff"


# --- H1-Nachtrag (13.11., docs/h1-single-session.md) --------------------------


def test_h1_batch_has_own_run_ids_and_output_dirs():
    """Die A-Wiederholung auf Post-Cutoff darf nicht als main-*-hard_fsm-r* laufen:
    der Resume hielte sie fuer gemessen (Kern-Runs vom 27.10.) und fuehre nichts."""
    from main_runs import out_subdir
    plan = build_plan(["ev_purchase"], replicates=2, arms=["hard_fsm_single", "hard_fsm"],
                      batch="h1")
    ids = [run_id_for(j) for j in plan]
    assert ids == ["h1-ev_purchase-hard_fsm_single-r1", "h1-ev_purchase-hard_fsm-r1",
                   "h1-ev_purchase-hard_fsm_single-r2", "h1-ev_purchase-hard_fsm-r2"]
    assert out_subdir("manifests", batch="h1") == "_h1"
    assert out_subdir("manifests/post_cutoff", batch="h1") == "_post_cutoff_h1"
    with pytest.raises(ValueError):
        out_subdir("manifests", leakage=True, batch="h1")
    with pytest.raises(ValueError):
        build_plan(["wine"], replicates=1, arms=["hard_fsm"], leakage=True, batch="h1")


def test_h1_batch_does_not_touch_core_summaries(tmp_path):
    """Ein gemessener Kern-Run main-wine-hard_fsm-r1 zaehlt NICHT fuer h1-wine-hard_fsm-r1."""
    runs = str(tmp_path)
    core = build_plan(["wine"], replicates=1, arms=["hard_fsm"])
    _write_summary(runs, core[0])
    h1 = build_plan(["wine"], replicates=1, arms=["hard_fsm"], batch="h1")
    row = summarize(aggregate(h1, runs))[0]
    assert row["n_measured"] == 0 and row["missing"] == 1


def test_h1_arm_reaches_the_subprocess(tmp_path, monkeypatch):
    import subprocess
    import main_runs
    runs = str(tmp_path / "runs")
    plan = build_plan(["wine"], replicates=1, arms=["hard_fsm_single"], batch="h1")
    seen = []

    def fake_subprocess_run(cmd, **kw):
        rid = cmd[cmd.index("--run-id") + 1]
        seen.append((rid, cmd[cmd.index("--arm") + 1], "--leakage" in cmd))
        os.makedirs(os.path.join(runs, rid), exist_ok=True)
        with open(os.path.join(runs, rid, "summary.json"), "w") as f:
            json.dump({"outcome": "SUCCESS", "success": True, "final_f1": 0.9,
                       "best_f1_measured": 0.9, "holdout_intact": True}, f)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("variance_scan.subprocess.run", fake_subprocess_run)
    main_runs.run_plan(plan, runs, "m", "manifests", timeout=10,
                       log_dir=str(tmp_path / "logs"), wait_s=1, max_failures=0)
    assert seen == [("h1-wine-hard_fsm_single-r1", "hard_fsm_single", False)]


def test_cli_h1_dry_runs(tmp_path, monkeypatch, capsys):
    """Kern: 1 Arm x 13 Sets x 6 = 78; Post-Cutoff: 2 Arme x 2 Sets x 6 = 24."""
    import main_runs
    monkeypatch.setattr("sys.argv", ["prog", "--dry-run", "--batch", "h1",
                                     "--arms", "hard_fsm_single", "--runs-dir", str(tmp_path)])
    main_runs.main()
    out = capsys.readouterr().out
    assert "Plan: 78 Runs" in out and "_h1" in out
    monkeypatch.setattr("sys.argv", ["prog", "--dry-run", "--batch", "h1",
                                     "--arms", "hard_fsm_single", "hard_fsm",
                                     "--manifests-dir", "manifests/post_cutoff",
                                     "--runs-dir", str(tmp_path)])
    main_runs.main()
    out = capsys.readouterr().out
    assert "Plan: 24 Runs" in out and "_post_cutoff_h1" in out


def test_cli_single_arm_requires_batch(tmp_path, monkeypatch):
    """Ohne --batch liefen A-Single-Runs als main-* und die Aggregation ueberschriebe _main/."""
    import main_runs
    monkeypatch.setattr("sys.argv", ["prog", "--dry-run", "--arms", "hard_fsm_single",
                                     "--runs-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        main_runs.main()


def test_default_plan_still_has_only_the_four_arms(tmp_path, monkeypatch, capsys):
    import main_runs
    assert main_runs.ARMS == ("hard_fsm", "soft_fsm", "crisp_react", "naked_react")
    monkeypatch.setattr("sys.argv", ["prog", "--dry-run", "--runs-dir", str(tmp_path)])
    main_runs.main()
    out = capsys.readouterr().out
    assert "Plan: 312 Runs" in out and "hard_fsm_single" not in out
