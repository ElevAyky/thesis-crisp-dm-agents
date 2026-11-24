"""Facts der zweiten Stufe (H1: Arm A gegen A-Single) -> thesis/facts/derived/h1.json.

Liest die h1-Runs (Kern + Post-Cutoff) direkt aus dem runs-Repo und die Auswertung von
scripts/analyze_h1.py (<runs>/_h1/analysis/analysis.json). Ergaenzt collect_facts.py, ohne
dessen abgeleitete Dateien neu zu schreiben.

Aufruf (Repo-Root):
    uv run python scripts/analyze_h1.py --runs-dir ../thesis-crisp-dm-runs
    uv run python thesis/scripts/collect_facts_h1.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_facts as cf  # noqa: E402

H1_ARMS = ["hard_fsm", "hard_fsm_single"]
# Vorab festgelegte Erwartungen, docs/h1-single-session.md §5
EXP_F1_DIFF_MAX = 0.01
EXP_POST_SINGLE_MIN = 10
EXP_COST_DELTA = (0.30, 0.65)
EXP_REPL_POST = (10, 12)
EXP_REPL_F1_TOL = 0.005


def cost_block(runs):
    out = {}
    for a in H1_ARMS:
        rs = [r for r in runs if r["arm"] == a]
        if not rs:
            continue
        tok = {k: sum(r["usage"].get(k, 0) for r in rs) for k in cf.TOKEN_KEYS}
        tok_cost = {k: tok[k] * cf.PRICE[k.replace("_tokens", "")] for k in cf.TOKEN_KEYS}
        total = sum(r["cost"] for r in rs)
        out[a] = {
            "runs": len(rs), "cost_total": total, "cost_per_run_mean": total / len(rs),
            "cost_per_run_median": cf.median([r["cost"] for r in rs]),
            "cost_per_run_max": max(r["cost"] for r in rs),
            "tokens_per_run": {k: v / len(rs) for k, v in tok.items()},
            "cost_by_token_type_per_run": {k: v / len(rs) for k, v in tok_cost.items()},
            "repriced_equals_summary": abs(sum(tok_cost.values()) - total) < 1e-6,
            "calls_mean": cf.mean([r["total_calls"] for r in rs]),
            "sessions_mean": cf.mean([r["n_sessions"] for r in rs]),
            "sessions_max": max(r["n_sessions"] for r in rs),
        }
    return out


def span(runs):
    starts = [r["transcripts"]["start"] for r in runs if r["transcripts"]["start"]]
    ends = [r["transcripts"]["end"] for r in runs if r["transcripts"]["end"]]
    return {"first_start_utc": min(starts), "last_end_utc": max(ends),
            "first_start_local": cf.to_local(min(starts)), "last_end_local": cf.to_local(max(ends))}


def model_counts(runs_dir, runs):
    c = collections.Counter()
    for r in runs:
        d = os.path.join(runs_dir, r["run_id"], "transcripts")
        for f in os.listdir(d):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                for line in fh:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("type") == "assistant":
                        m = (e.get("message") or {}).get("model")
                        if m:
                            c[m] += 1
    return dict(c)


def compactions(runs_dir, runs):
    """Runs, in deren Transkripten der Client den Kontext kompaktiert hat (Marker compact_boundary)."""
    hit = []
    for r in runs:
        d = os.path.join(runs_dir, r["run_id"], "transcripts")
        for f in os.listdir(d):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                if "compact_boundary" in fh.read():
                    hit.append(r["run_id"])
                    break
    return {"runs_with_compaction": hit, "runs_checked": len(runs)}


def flagged_access(runs_dir, runs):
    """Markierte Zugriffe: zielt jeder Treffer auf die eigene ausgelagerte Tool-Ausgabe des Runs
    (eigener Projektordner des Clients, eigene Session-ID, Unterordner tool-results)?"""
    out = []
    for r in runs:
        if not r["access_flagged"]:
            continue
        s = cf.jload(os.path.join(runs_dir, r["run_id"], "summary.json"))
        own_dir = ("C--projects-thesis-crisp-dm-runs-" + r["run_id"].replace("_", "-") + "-work").lower()
        sessions = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(runs_dir, r["run_id"], "transcripts"))}
        ex = []
        for e in (s.get("access_audit") or {}).get("examples") or []:
            inp = e.get("input", "")
            low = inp.lower().replace("\\\\", "\\")
            ex.append({"kinds": e.get("kinds"), "input": inp[:300],
                       "own_project_dir": own_dir in low,
                       "own_session": any(sid.lower() in low for sid in sessions),
                       "tool_results": "tool-results" in low})
        out.append({"run_id": r["run_id"],
                    "counts": {k: v for k, v in (r["access_counts"] or {}).items() if v},
                    "examples": ex,
                    "all_own_tool_output": all(x["own_project_dir"] and x["own_session"] and x["tool_results"] for x in ex)})
    return out


def harness_code_changes(commits, base="0b1ccca"):
    """Dateien ausserhalb von thesis/ und docs/, die sich zwischen base und jedem Commit unterscheiden."""
    if not all(cf.has_commit(cf.REPO, c) for c in [base, *commits]):
        # Pruefer-Schnappschuss ohne Historie (Anhang B): Wert aus der im Arbeits-Repo berechneten h1.json
        print("  Hinweis: keine Git-Historie -> harness_code_changes_vs_0b1ccca aus gespeicherter h1.json")
        stored = cf.jload(os.path.join(cf.OUT, "h1.json"))["harness_code_changes_vs_0b1ccca"]
        assert set(stored) == set(commits), (sorted(stored), sorted(commits))
        return stored
    out = {}
    for c in sorted(commits):
        p = subprocess.run(["git", "-C", cf.REPO, "diff", "--name-only", base, c, "--", ".", ":!thesis", ":!docs"],
                           capture_output=True, text=True, check=True)
        out[c] = [x for x in p.stdout.splitlines() if x]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(cf.REPO), "thesis-crisp-dm-runs"))
    args = ap.parse_args()
    runs_dir = os.path.abspath(args.runs_dir)
    core_man, post_man = cf.manifests()
    core = cf.load_batch(runs_dir, "h1", set(core_man), "h1_core")
    post = cf.load_batch(runs_dir, "h1", set(post_man), "h1_post")
    print({"h1_core": len(core), "h1_post": len(post)})
    ana = cf.jload(os.path.join(runs_dir, "_h1", "analysis", "analysis.json"))

    prim = ana["core"]["primary"]
    repl = ana["core"]["replication"]
    means = {}
    for key, block in (("primary", prim), ("replication", repl), ("old_vs_single", ana["core"]["old_vs_single"])):
        m = block["metrics"]["f1_measured"]
        means[key] = {ds: row for ds, row in zip(m["datasets"], m["matrix"])}
    diffs = {ds: means["primary"][ds][0] - means["primary"][ds][1] for ds in means["primary"]}
    repl_diffs = {ds: means["replication"][ds][1] - means["replication"][ds][0] for ds in means["replication"]}

    costs = {"h1_core": cost_block(core), "h1_post": cost_block(post)}
    cost_delta_core = (costs["h1_core"]["hard_fsm_single"]["cost_per_run_mean"]
                       - costs["h1_core"]["hard_fsm"]["cost_per_run_mean"])
    post_pool = ana["post_cutoff"]["pooled"]

    expectations = {
        "e1_core_no_difference_both_levels": ana["core"]["verdicts"],
        "e1_abs_f1_diff_below_0.01": {"count": sum(abs(d) < EXP_F1_DIFF_MAX for d in diffs.values()),
                                       "of": len(diffs),
                                       "above": {ds: d for ds, d in diffs.items() if abs(d) >= EXP_F1_DIFF_MAX}},
        "e2_post_single_success": {"success": post_pool["hard_fsm_single"]["success"],
                                    "n": post_pool["hard_fsm_single"]["n"],
                                    "met": post_pool["hard_fsm_single"]["success"] >= EXP_POST_SINGLE_MIN},
        "e3_cost_delta_single_minus_a_core": {"delta_per_run": cost_delta_core,
                                               "expected": list(EXP_COST_DELTA),
                                               "met": EXP_COST_DELTA[0] <= cost_delta_core <= EXP_COST_DELTA[1]},
        "e4_replication": {
            "core_success_new": prim["rates"]["hard_fsm"]["success"], "core_n_new": prim["rates"]["hard_fsm"]["n"],
            "post_success_new": post_pool["hard_fsm"]["success"], "post_n_new": post_pool["hard_fsm"]["n"],
            "post_in_range": EXP_REPL_POST[0] <= post_pool["hard_fsm"]["success"] <= EXP_REPL_POST[1],
            "core_cells_within_0.005": sum(abs(d) <= EXP_REPL_F1_TOL for d in repl_diffs.values()),
            "core_cells": len(repl_diffs),
            "outside": {ds: d for ds, d in repl_diffs.items() if abs(d) > EXP_REPL_F1_TOL},
            "max_abs_diff": max(abs(d) for d in repl_diffs.values()),
            "post_f1_drift_new_minus_old": ana["post_cutoff"]["f1_drift_new_minus_old"],
        },
    }

    out = {
        "source": "runs h1-*/summary.json, events.jsonl, transcripts; _h1/analysis/analysis.json (analyze_h1.py)",
        "n_runs": {"h1_core": len(core), "h1_post": len(post)},
        "span": {"h1_core": span(core), "h1_post": span(post)},
        "controls": cf.controls({"h1_core": core, "h1_post": post}),
        "access_flagged": flagged_access(runs_dir, core + post),
        "harness_code_changes_vs_0b1ccca": harness_code_changes(
            {str(r["harness"].get("commit"))[:7] for r in core + post}),
        "models_in_transcripts": model_counts(runs_dir, core + post),
        "compaction": compactions(runs_dir, core + post),
        "kgit": {"runs_with_git_calls": cf.kgit({"h1_core": core, "h1_post": post}),
                 "scanned_runs": len(core) + len(post)},
        "completeness": {k: ana["core"][k]["completeness"] for k in ("primary", "replication", "old_vs_single")},
        "failed_runs": [{"run_id": r["run_id"], "outcome": r["outcome"], "fail_reason": r["fail_reason"],
                         "f1": r["f1"], "baseline_f1": r["baseline_f1"], "ci_hi": r["ci_hi"]}
                        for r in core + post if not r["success"]],
        "costs": costs,
        "cost_total_all_h1": sum(r["cost"] for r in core + post),
        "cost_delta_single_minus_a_core_per_run": cost_delta_core,
        "analysis": ana,
        "cell_means_f1_measured": {"A_new_vs_single": means["primary"], "A_old_vs_A_new": means["replication"]},
        "diff_A_new_minus_single": diffs,
        "diff_A_new_minus_A_old": repl_diffs,
        "expectations": expectations,
    }
    cf.jdump("h1.json", cf.clean(json.loads(json.dumps(out, default=cf._default))))
    print("geschrieben:", os.path.join(cf.OUT, "h1.json"))


if __name__ == "__main__":
    main()
