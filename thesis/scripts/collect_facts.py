# thesis/scripts/collect_facts.py
"""Faktenbasis fuer Kap. 5 (Experiment Execution) und Kap. 6 (Results).

Liest NUR die Rohdaten im runs-Repo (summary.json, events.jsonl, Transkript-Zeitstempel
der sauberen Runs main-*/leak-*/spur-*, dazu calib-*/scan-* fuer die Kalibrierung und die
Aggregate _main/, _post_cutoff/, _leakage/, _spurious/) plus Manifeste und Code des
Thesis-Repos. Schreibt abgeleitete Kennzahlen nach thesis/facts/derived/*.json.
Ins runs-Repo wird nichts geschrieben.

Statistik-Funktionen (Wilson, Friedman, Nemenyi, Cliff's delta) kommen aus
scripts/analyze_main.py, damit die Zahlen identisch zur Auswertung der Runs sind.
Die Median-Sensitivitaet erzeugt dieses Script ueber den CLI-Aufruf
    analyze_main.py --input <runs>/_main/main_runs.json --out-dir thesis/facts/derived/median_core --agg median
(analog fuer _post_cutoff).

Archivierte Runs (automem_*, promptv1_*, gitctx_*) werden nur gezaehlt bzw. als
ausdruecklich benannter Vergleich fuer den Kontaminations-Abschnitt in Kap. 5 gelesen
(contamination.json, Schluessel "archived_comparison").

Aufruf (Repo-Root):
    uv run python thesis/scripts/collect_facts.py
    uv run python thesis/scripts/collect_facts.py --runs-dir ../thesis-crisp-dm-runs --skip-median
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import itertools
import json
import math
import os
import statistics
import subprocess
import sys

import numpy as np
import yaml
from scipy.stats import fisher_exact

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import analyze_main as am  # noqa: E402

try:
    from crispfsm.pricing import PRICING  # noqa: E402
    PRICE = PRICING["claude-opus-4-8"]
except Exception:  # pragma: no cover - Fallback = Werte aus src/crispfsm/pricing.py
    PRICE = {"input": 5e-6, "output": 25e-6, "cache_read": 0.5e-6, "cache_creation": 6.25e-6}

ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
LETTER = {"hard_fsm": "A", "soft_fsm": "B", "crisp_react": "C", "naked_react": "D"}
FSM_ARMS = ["hard_fsm", "soft_fsm"]
REACT_ARMS = ["crisp_react", "naked_react"]
PHASES = ["bu", "du", "prep", "model", "eval"]
TOKEN_KEYS = ["input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"]
OUT = os.path.join(REPO, "thesis", "facts", "derived")
# Sommerzeit 2026 endet am 25.10.2026 01:00 UTC (EU-Regel: letzter Sonntag im Oktober)
DST_END_UTC = dt.datetime(2026, 10, 25, 1, 0, tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------- helpers

def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jdump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=False, default=_default)
    print(f"  -> {os.path.relpath(path, REPO)}")


def _default(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


def clean(o):
    """NaN/inf -> String, damit json strikt bleibt."""
    if isinstance(o, float):
        if math.isnan(o):
            return "nan"
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def price(usage: dict) -> float:
    return (usage.get("input_tokens", 0) * PRICE["input"]
            + usage.get("output_tokens", 0) * PRICE["output"]
            + usage.get("cache_read_tokens", 0) * PRICE["cache_read"]
            + usage.get("cache_creation_tokens", 0) * PRICE["cache_creation"])


def to_local(ts_utc: str) -> str:
    t = dt.datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    off = 2 if t < DST_END_UTC else 1
    loc = t + dt.timedelta(hours=off)
    return loc.strftime("%Y-%m-%d %H:%M") + (" CEST" if off == 2 else " CET")


def parse_ts(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni (step-down), adjustierte p-Werte in Eingabe-Reihenfolge."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def fisher(a_succ, a_n, b_succ, b_n):
    table = [[a_succ, a_n - a_succ], [b_succ, b_n - b_succ]]
    odds, p = fisher_exact(table)
    return {"table": table, "odds_ratio": float(odds), "p": float(p)}


def median(xs):
    return statistics.median(xs) if xs else None


def mean(xs):
    return statistics.fmean(xs) if xs else None


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else None


def git_log(repo: str, fmt="%ad %h %s", extra=None):
    cmd = ["git", "-C", repo, "log", "--date=iso", f"--format={fmt}"] + (extra or [])
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8").stdout
    return [line for line in out.splitlines() if line.strip()]


def has_commit(repo: str, rev: str) -> bool:
    return subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{rev}^{{commit}}"],
                          capture_output=True).returncode == 0


def history_available(runs_dir: str) -> bool:
    """Die veroeffentlichten Pruefer-Repos sind Schnappschuesse mit je einem Commit (Anhang B). Dort fehlen die
    Commits, aus denen einige Felder abgeleitet werden; diese Felder werden dann aus den gespeicherten
    Dateien in thesis/facts/derived/ uebernommen, die im Arbeits-Repo mit Historie berechnet wurden."""
    return has_commit(REPO, "14e99e4") and has_commit(runs_dir, "22684c4")


# --------------------------------------------------------------------------- loading

def manifests():
    core, post = {}, {}
    for p in sorted(glob.glob(os.path.join(REPO, "manifests", "*.yaml"))):
        with open(p, encoding="utf-8") as f:
            core[os.path.splitext(os.path.basename(p))[0]] = yaml.safe_load(f)
    for p in sorted(glob.glob(os.path.join(REPO, "manifests", "post_cutoff", "*.yaml"))):
        with open(p, encoding="utf-8") as f:
            post[os.path.splitext(os.path.basename(p))[0]] = yaml.safe_load(f)
    return core, post


def transcript_info(run_dir: str):
    """Start/Ende (UTC) aus den Transkript-Zeitstempeln + Git-Aufrufe des Workers."""
    lo = hi = None
    files = sorted(glob.glob(os.path.join(run_dir, "transcripts", "*.jsonl")))
    nonempty = 0
    git_cmds = []
    for fpath in files:
        if os.path.getsize(fpath) > 0:
            nonempty += 1
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                t = e.get("timestamp")
                if t:
                    lo = t if lo is None or t < lo else lo
                    hi = t if hi is None or t > hi else hi
                for cmd in _bash_commands(e):
                    toks = cmd.replace(";", " ").replace("&&", " ").replace("|", " ").split()
                    if "git" in toks:
                        git_cmds.append(cmd[:200])
    return {"n_files": len(files), "n_nonempty": nonempty, "start": lo, "end": hi,
            "git_cmds": git_cmds}


def _bash_commands(obj):
    """Alle tool_use-Eingaben mit Schluessel 'command' (Bash) rekursiv einsammeln."""
    out = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            if o.get("type") == "tool_use" and isinstance(o.get("input"), dict):
                c = o["input"].get("command")
                if isinstance(c, str):
                    out.append(c)
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return out


def load_run(runs_dir: str, rid: str, batch: str, with_transcripts=True) -> dict:
    d = os.path.join(runs_dir, rid)
    s = jload(os.path.join(d, "summary.json"))
    events = []
    ep = os.path.join(d, "events.jsonl")
    if os.path.exists(ep):
        with open(ep, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
    parts = rid.split("-")
    ds, arm, rep = parts[1], parts[2], int(parts[3].lstrip("r"))
    bc = s.get("baseline_comparison") or {}
    worker_usage = collections.Counter()
    gate_usage = collections.Counter()
    for e in events:
        worker_usage.update(e.get("usage") or {})
        gate_usage.update(e.get("gate_usage") or {})
    sessions = {e.get("session_id") for e in events if e.get("session_id")}
    sessions |= {e.get("gate_session_id") for e in events if e.get("gate_session_id")}
    rec = {
        "run_id": rid, "batch": batch, "dataset": ds, "arm": arm, "rep": rep,
        "outcome": s.get("outcome"), "success": bool(s.get("success")),
        "fail_reason": s.get("fail_reason"), "fail_phase": s.get("fail_phase"),
        "f1": s.get("best_f1_measured"), "final_f1": s.get("final_f1"),
        "baseline_f1": bc.get("baseline_f1"), "diff": bc.get("diff"),
        "ci_lo": bc.get("ci_lo"), "ci_hi": bc.get("ci_hi"),
        "sig_better": (bc.get("ci_lo") is not None and bc["ci_lo"] > 0),
        "cost": s.get("total_cost_usd"), "usage": s.get("usage_total") or {},
        "worker_cost_events": price(worker_usage), "gate_cost_events": price(gate_usage),
        "gate_usage_events": dict(gate_usage), "worker_usage_events": dict(worker_usage),
        "total_calls": s.get("total_calls"), "gate_calls": s.get("gate_calls"),
        "worker_turns_summary": s.get("worker_turns"),
        "worker_turns_events": sum(int(e.get("worker_num_turns") or 0) for e in events),
        "latency_ms": sum(float(e.get("latency_ms") or 0) for e in events),
        "gate_latency_ms": sum(float(e.get("gate_latency_ms") or 0) for e in events),
        "budget_exhausted": s.get("budget_exhausted"),
        "attempts": s.get("attempts") or {}, "phase_seq": [e.get("phase") for e in events],
        "events": events, "n_sessions": len(sessions),
        "harness": s.get("harness") or {}, "holdout_intact": s.get("holdout_intact"),
        "access_flagged": (s.get("access_audit") or {}).get("flagged"),
        "access_counts": (s.get("access_audit") or {}).get("counts"),
        "transcripts_listed": len(s.get("transcripts") or []),
        "bu_correct": s.get("bu_correct"), "bu_correct_first": s.get("bu_correct_first"),
        "target_ok": s.get("target_ok"),
        "du_excluded": [c.get("column") for c in ((s.get("du_report") or {}).get("excluded_columns") or [])],
        "class_balance": (s.get("du_report") or {}).get("class_balance"),
        "leak_exploited": s.get("leakage_exploited"),
        "leak_kappa": (s.get("leakage_reliance") or {}).get("kappa"),
        "leak_z": (s.get("leakage_reliance") or {}).get("z"),
        "leak_column": s.get("leakage_column"), "leak_variant": s.get("leakage_variant"),
        "leak_detected": s.get("leakage_detected"),
        "success_rule": s.get("success_rule"), "abstained": s.get("abstained"),
    }
    if with_transcripts:
        rec["transcripts"] = transcript_info(d)
    return rec


def load_batch(runs_dir, prefix, datasets, batch):
    runs = []
    for d in sorted(os.listdir(runs_dir)):
        if not d.startswith(prefix + "-"):
            continue
        parts = d.split("-")
        if len(parts) != 4 or parts[1] not in datasets:
            continue
        runs.append(load_run(runs_dir, d, batch))
    return runs


# --------------------------------------------------------------------------- cells

def to_cells(runs):
    """Zellen im Format von main_runs.json (fuer analyze_main-Funktionen)."""
    by = collections.defaultdict(list)
    for r in runs:
        by[(r["arm"], r["dataset"])].append(r)
    cells = []
    for (arm, ds), rs in by.items():
        rs = sorted(rs, key=lambda r: r["rep"])
        outcomes = collections.Counter(r["outcome"] for r in rs)
        cells.append({
            "arm": arm, "dataset": ds, "n_measured": len(rs),
            "outcomes": dict(outcomes),
            "fail_reasons": dict(collections.Counter(r["fail_reason"] for r in rs if r["fail_reason"])),
            "f1_measured": [r["f1"] for r in rs if r["f1"] is not None],
            "f1_on_success": [r["final_f1"] for r in rs if r["final_f1"] is not None],
            "f1_combined": [r["final_f1"] if r["success"] and r["final_f1"] is not None else 0.0 for r in rs],
            "run_ids": [r["run_id"] for r in rs],
        })
    return cells


def level1(runs, arms):
    per_arm = {}
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        k, n = sum(r["success"] for r in rs), len(rs)
        lo, hi = am.wilson(k, n)
        per_arm[a] = {"success": k, "n": n, "rate": k / n if n else None, "wilson": [lo, hi],
                      "fail_reasons": dict(collections.Counter(r["fail_reason"] for r in rs if r["fail_reason"]))}
    pairs = []
    for a, b in itertools.combinations(arms, 2):
        f = fisher(per_arm[a]["success"], per_arm[a]["n"], per_arm[b]["success"], per_arm[b]["n"])
        pairs.append({"a": a, "b": b, **f})
    adj = holm([p["p"] for p in pairs])
    for p, h in zip(pairs, adj):
        p["p_holm"] = h
    per_cell = []
    for ds in sorted({r["dataset"] for r in runs}):
        for a in arms:
            rs = [r for r in runs if r["arm"] == a and r["dataset"] == ds]
            k, n = sum(r["success"] for r in rs), len(rs)
            per_cell.append({"dataset": ds, "arm": a, "success": k, "n": n,
                             "wilson": list(am.wilson(k, n)),
                             "fail_reasons": dict(collections.Counter(r["fail_reason"] for r in rs if r["fail_reason"])),
                             "fails": [{"run_id": r["run_id"], "fail_reason": r["fail_reason"],
                                        "f1": r["f1"], "diff": r["diff"], "ci_hi": r["ci_hi"]}
                                       for r in rs if not r["success"]]})
    return {"per_arm": per_arm, "pairs": pairs, "per_cell": per_cell}


def secondary(runs, arms):
    """experiment-design §5.3b: signifikant besser <=> baseline_comparison.ci_lo > 0."""
    per_arm, per_cell = {}, []
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        k, n = sum(r["sig_better"] for r in rs), len(rs)
        per_arm[a] = {"sig_better": k, "n": n, "rate": k / n if n else None,
                      "wilson": list(am.wilson(k, n))}
    for ds in sorted({r["dataset"] for r in runs}):
        row = {"dataset": ds}
        for a in arms:
            rs = [r for r in runs if r["arm"] == a and r["dataset"] == ds]
            row[a] = sum(r["sig_better"] for r in rs)
        per_cell.append(row)
    pairs = []
    for a, b in itertools.combinations(arms, 2):
        pairs.append({"a": a, "b": b, **fisher(per_arm[a]["sig_better"], per_arm[a]["n"],
                                              per_arm[b]["sig_better"], per_arm[b]["n"])})
    for p, h in zip(pairs, holm([p["p"] for p in pairs])):
        p["p_holm"] = h
    zero_cells = {a: [c["dataset"] for c in per_cell if c[a] == 0] for a in arms}
    return {"per_arm": per_arm, "pairs": pairs, "per_cell": per_cell, "cells_with_zero": zero_cells}


def level2(runs, arms, agg="mean"):
    cells = to_cells(runs)
    out = {}
    for metric in am.METRICS:
        res = am.analyze_metric(cells, arms, metric, agg, am.ALPHA)
        out[metric] = res
    return out, cells


def run_sd_table(cells, arms):
    per_cell = {}
    for c in cells:
        v = c["f1_measured"]
        per_cell.setdefault(c["dataset"], {})[c["arm"]] = sd(v)
    per_arm = {}
    for a in arms:
        vals = [per_cell[ds][a] for ds in per_cell if per_cell[ds].get(a) is not None]
        per_arm[a] = {"median_sd": median(vals), "max_sd": max(vals) if vals else None,
                      "max_sd_dataset": max(per_cell, key=lambda ds: per_cell[ds].get(a) or -1),
                      "min_sd": min(vals) if vals else None, "cells": len(vals)}
    # Datasets, in denen Arm x allein die groesste SD hat (Gleichstaende zaehlen nicht)
    largest = collections.Counter()
    ties = []
    for ds, row in per_cell.items():
        top = max(v for v in row.values() if v is not None)
        winners = [a for a in arms if row.get(a) is not None and abs(row[a] - top) < 1e-12]
        if len(winners) == 1:
            largest[winners[0]] += 1
        else:
            ties.append(ds)
    zero = {a: sorted(ds for ds in per_cell if per_cell[ds].get(a) is not None and per_cell[ds][a] < 1e-12)
            for a in arms}
    return {"per_cell": per_cell, "per_arm": per_arm, "arm_with_largest_sd_count": dict(largest),
            "sets_with_tied_largest_sd": ties, "zero_sd_cells": zero}


def cell_means(cells, metric="f1_measured"):
    m = {}
    for c in cells:
        v = c[metric]
        m.setdefault(c["dataset"], {})[c["arm"]] = mean(v) if v else None
    return m


# --------------------------------------------------------------------------- sections

def controls(batches: dict) -> dict:
    out = {}
    for name, runs in batches.items():
        c = collections.Counter()
        for r in runs:
            h = r["harness"]
            c["harness_commit:" + str(h.get("commit"))[:7]] += 1
            c["claude_version:" + str(h.get("claude_version"))] += 1
            c["prompt_sha256:" + str(h.get("prompt_sha256"))[:8]] += 1
            c["dirty_false"] += h.get("dirty") is False
            c["auto_memory_disabled"] += h.get("auto_memory") == "disabled"
            c["git_context_disabled"] += h.get("git_context") == "disabled"
            c["holdout_intact"] += r["holdout_intact"] is True
            c["access_audit_flagged"] += r["access_flagged"] is True
            c["access_audit_not_flagged"] += r["access_flagged"] is False
            t = r["transcripts"]
            c["transcripts_nonempty"] += (t["n_files"] > 0 and t["n_nonempty"] == t["n_files"])
            c["transcripts_match_sessions"] += t["n_files"] == r["n_sessions"]
            c["outcome:" + str(r["outcome"])] += 1
            c["bu_or_target_ok"] += r["target_ok"] is True
            hits = sum((r["access_counts"] or {}).values()) if r["access_counts"] else 0
            c["access_audit_hits"] += hits
            c["success_rule:" + str(r["success_rule"])] += 1
            c["abstained"] += r["abstained"] is True
            c["budget_exhausted"] += r["budget_exhausted"] is True
        out[name] = {"n_runs": len(runs), **dict(sorted(c.items()))}
    return out


def code_constants():
    """Konstanten direkt aus dem Harness-Code (nicht aus der Doku)."""
    import inspect
    from crispfsm import config as cfgmod, datasets as dsmod, leakage as lkmod, scoring as scmod
    rc = cfgmod.RunConfig(run_id="x", workdir="x")
    split_fn = [f for _, f in inspect.getmembers(dsmod, inspect.isfunction)
                if f.__module__ == dsmod.__name__ and "test_size" in inspect.signature(f).parameters]
    test_size = inspect.signature(split_fn[0]).parameters["test_size"].default
    return {"run_budget_usd": rc.run_budget_usd, "run_max_turns": rc.run_max_turns,
            "budget_usd_per_call": rc.budget_usd_per_call, "max_turns_per_call": rc.max_turns,
            "max_retries_per_phase": rc.max_retries, "safety_max_calls": rc.safety_max_calls,
            "react_max_turns_default": rc.react_max_turns, "fsm_phases": list(cfgmod.FSM_PHASES),
            "val_fraction": cfgmod.VAL_FRACTION, "test_fraction": test_size,
            "threshold_k": cfgmod.THRESHOLD_K,
            "n_boot": scmod.N_BOOT, "boot_seed": scmod.BOOT_SEED, "alpha": scmod.ALPHA,
            "leak_z_crit": lkmod.Z_CRIT, "leak_kappa_min": lkmod.KAPPA_MIN,
            "_source": "src/crispfsm/config.py (RunConfig, FSM_PHASES, VAL_FRACTION, THRESHOLD_K), "
                       "src/crispfsm/scoring.py:17-19, src/crispfsm/leakage.py:25-29, "
                       f"src/crispfsm/datasets.py ({split_fn[0].__name__}: test_size)"}


def _int_field(text: str, key: str) -> int:
    """Ganzzahl hinter "'key': " aus der [setup]-Zeile lesen (Python-dict-Repr)."""
    i = text.index(f"'{key}': ") + len(key) + 4
    j = i
    while j < len(text) and text[j].isdigit():
        j += 1
    return int(text[i:j])


def splits(runs_dir):
    """n_train/n_val/n_test pro Dataset aus den [setup]-Zeilen der Runner-Logs."""
    out = {}
    for sub in ("_main", "_post_cutoff", "_leakage", "_spurious"):
        for lp in sorted(glob.glob(os.path.join(runs_dir, sub, "logs", "*.log"))):
            with open(lp, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("[setup] ") and ": {" in line:
                        name = line[len("[setup] "):].split(": ", 1)[0]
                        vals = {k: _int_field(line, k) for k in
                                ("n_features", "n_classes", "n_train", "n_val", "n_test")}
                        prev = out.setdefault(f"{sub}|{name}", {**vals, "logs": 0, "consistent": True})
                        prev["logs"] += 1
                        if any(prev[k] != vals[k] for k in vals):
                            prev["consistent"] = False
                        break
    for v in out.values():
        tot = v["n_train"] + v["n_val"] + v["n_test"]
        v["total"] = tot
        v["fractions"] = [v["n_train"] / tot, v["n_val"] / tot, v["n_test"] / tot]
    return out


def kgit(batches):
    rows = []
    for name, runs in batches.items():
        for r in runs:
            cmds = r["transcripts"]["git_cmds"]
            if cmds:
                rows.append({"batch": name, "run_id": r["run_id"], "commands": cmds})
    return rows


def timeline(batches, runs_dir):
    out = {}
    for name, runs in batches.items():
        rs = sorted([r for r in runs if r["transcripts"]["start"]], key=lambda r: r["transcripts"]["start"])
        start, end = rs[0]["transcripts"]["start"], max(r["transcripts"]["end"] for r in rs)
        durs = [(parse_ts(r["transcripts"]["end"]) - parse_ts(r["transcripts"]["start"])).total_seconds() / 60
                for r in rs]
        gaps, prev_end = [], None
        for r in rs:
            st = r["transcripts"]["start"]
            if prev_end is not None:
                g = (parse_ts(st) - parse_ts(prev_end)).total_seconds() / 60
                if g > 60:
                    gaps.append({"from_utc": prev_end, "to_utc": st, "from_local": to_local(prev_end),
                                 "to_local": to_local(st), "minutes": g, "next_run": r["run_id"]})
            en = r["transcripts"]["end"]
            prev_end = en if prev_end is None or en > prev_end else prev_end
        wall_h = (parse_ts(end) - parse_ts(start)).total_seconds() / 3600
        per_arm = {}
        for a in ARMS:
            ra = [r for r in rs if r["arm"] == a]
            if not ra:
                continue
            dd = [(parse_ts(r["transcripts"]["end"]) - parse_ts(r["transcripts"]["start"])).total_seconds() / 60
                  for r in ra]
            per_arm[a] = {"runs": len(ra), "mean_min": mean(dd), "median_min": median(dd), "max_min": max(dd),
                          "sum_h": sum(dd) / 60,
                          "mean_model_latency_min": mean([(r["latency_ms"] + r["gate_latency_ms"]) / 60000 for r in ra])}
        out[name] = {
            "n_runs": len(runs), "start_utc": start, "end_utc": end,
            "start_local": to_local(start), "end_local": to_local(end),
            "wall_clock_h": wall_h, "sum_run_minutes_h": sum(durs) / 60,
            "wall_clock_minus_gaps_h": wall_h - sum(g["minutes"] for g in gaps) / 60,
            "gaps_over_60min": gaps, "per_arm": per_arm,
            "cost_usd": sum(r["cost"] for r in runs),
            "harness_commits": sorted({r["harness"].get("commit", "")[:7] for r in runs}),
            "claude_versions": sorted({str(r["harness"].get("claude_version")) for r in runs}),
            "arms": sorted({r["arm"] for r in runs}, key=ARMS.index),
            "datasets": sorted({r["dataset"] for r in runs}),
            "errors_in_final_summaries": sum(r["outcome"] == "ERROR" for r in runs),
        }
    if not history_available(runs_dir):
        stored = jload(os.path.join(OUT, "timeline.json"))
        for k in ("_runs_repo_git_log", "_harness_repo_key_commits", "_diff_9bb6a91_14e99e4",
                  "_summaries_per_runs_repo_commit"):
            out[k] = stored[k]
        print("  Hinweis: keine Git-Historie (Pruefer-Schnappschuss) -> Historienfelder aus gespeicherter timeline.json")
    else:
        out["_runs_repo_git_log"] = git_log(runs_dir)
        keys = ["4845fc0", "b6fcc73", "559eafc", "63200e3", "e62ea25", "d848587", "9bb6a91", "14e99e4", "de857cd"]
        log = git_log(REPO)
        out["_harness_repo_key_commits"] = [l for l in log if l.split()[3] in keys]
        diff = subprocess.run(["git", "-C", REPO, "diff", "--stat", "9bb6a91", "14e99e4"],
                              capture_output=True, text=True, encoding="utf-8").stdout.strip().splitlines()
        out["_diff_9bb6a91_14e99e4"] = diff
        # Wie viele saubere Run-Summaries lagen in welchem runs-Repo-Commit (Zwischenstaende)?
        counts = {}
        for c in ("868b43c", "94d9212", "22684c4", "3085c8b", "4b46569", "25644b7"):
            names = subprocess.run(["git", "-C", runs_dir, "ls-tree", "-r", "--name-only", c],
                                   capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
            cnt = collections.Counter(n.split("-")[0] for n in names
                                      if n.endswith("/summary.json") and n.count("/") == 1
                                      and n.split("-")[0] in ("main", "leak", "spur"))
            counts[c] = dict(cnt)
        out["_summaries_per_runs_repo_commit"] = counts
    out["_interruption_note"] = ("Runner-Logs werden pro Versuch ueberschrieben (variance_scan.execute, open(..., 'w')), "
                                 "Warte-Meldungen bei rate_limited gehen nur auf die Konsole -> Rate-Limit-Ereignisse "
                                 "sind aus den Daten nicht rekonstruierbar.")
    logs = []
    for sub in ("_main", "_post_cutoff", "_leakage", "_spurious"):
        logs += glob.glob(os.path.join(runs_dir, sub, "logs", "*.log"))
    markers = ("rate_limited", "RUN ERROR", "session limit", "usage limit", "weekly limit")
    hits = []
    for lp in logs:
        with open(lp, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        if any(m in txt for m in markers):
            hits.append(os.path.basename(lp))
    out["_logs_scanned"] = len(logs)
    out["_logs_with_error_or_limit_marker"] = hits
    return out


def calibration(runs_dir, core_sets):
    rows = collections.defaultdict(list)
    missing = []
    for d in sorted(os.listdir(runs_dir)):
        if not (d.startswith("calib-") or d.startswith("scan-")):
            continue
        p = os.path.join(runs_dir, d, "summary.json")
        if not os.path.exists(p):
            missing.append(d)
            continue
        s = jload(p)
        parts = d.split("-")
        phase, ds, arm = parts[0], parts[1], parts[2]
        rows[(phase, ds, arm)].append({"run": d, "outcome": s.get("outcome"), "f1": s.get("final_f1"),
                                       "cost": s.get("total_cost_usd"), "holdout_intact": s.get("holdout_intact"),
                                       "threshold": s.get("eval_f1_threshold")})
    cells = []
    for (phase, ds, arm), rs in sorted(rows.items()):
        f1s = [r["f1"] for r in rs if r["f1"] is not None]
        cells.append({"phase": phase, "dataset": ds, "arm": arm, "n": len(rs),
                      "in_core_suite": ds in core_sets,
                      "outcomes": dict(collections.Counter(r["outcome"] for r in rs)),
                      "f1_values": f1s, "mean": mean(f1s), "sd": sd(f1s),
                      "cost_total": sum(r["cost"] for r in rs), "cost_per_run": mean([r["cost"] for r in rs]),
                      "holdout_intact": all(r["holdout_intact"] for r in rs),
                      "thresholds": sorted({r["threshold"] for r in rs})})
    calib = [c for c in cells if c["phase"] == "calib"]
    scan = [c for c in cells if c["phase"] == "scan"]
    target_se = 0.05 / 3
    # gepoolte SD fuer Zellen, die in Kalibrierung UND Scan gemessen wurden (df 2+2)
    pooled = {}
    for c in calib:
        twin = [s for s in scan if s["dataset"] == c["dataset"] and s["arm"] == c["arm"]]
        if twin and c["sd"] is not None and twin[0]["sd"] is not None:
            p = math.sqrt((c["sd"] ** 2 + twin[0]["sd"] ** 2) / 2)
            pooled[f"{c['dataset']}|{c['arm']}"] = {"calib_sd": c["sd"], "scan_sd": twin[0]["sd"], "pooled_sd": p,
                                                   "n_required": (p / target_se) ** 2}
    # massgebliche SD je Zelle (Suite): gepoolt wenn doppelt gemessen, sonst Einzelwert
    governing = {}
    for c in calib + scan:
        if not c["in_core_suite"] or c["sd"] is None:
            continue
        key = f"{c['dataset']}|{c['arm']}"
        governing[key] = pooled[key]["pooled_sd"] if key in pooled else max(governing.get(key, 0), c["sd"])
    worst_key = max(governing, key=governing.get)
    scan_d_suite = [c for c in scan if c["arm"] == "naked_react" and c["in_core_suite"]]
    scan_a = [c for c in scan if c["arm"] == "hard_fsm" and c["in_core_suite"]]
    all_a = [c for c in calib + scan if c["arm"] == "hard_fsm" and c["in_core_suite"] and c["sd"] is not None]
    single_max = max([c for c in calib + scan if c["in_core_suite"] and c["sd"] is not None], key=lambda c: c["sd"])
    out = {
        "cells": cells, "runs_without_summary": missing,
        "calibration": {"runs": sum(c["n"] for c in calib), "cost": sum(c["cost_total"] for c in calib),
                        "success": sum(c["outcomes"].get("SUCCESS", 0) for c in calib),
                        "datasets": sorted({c["dataset"] for c in calib}), "arms": sorted({c["arm"] for c in calib}),
                        "holdout_intact_all": all(c["holdout_intact"] for c in calib)},
        "scan": {"runs": sum(c["n"] for c in scan), "cost": sum(c["cost_total"] for c in scan),
                 "runs_suite": sum(c["n"] for c in scan if c["in_core_suite"]),
                 "cost_suite": sum(c["cost_total"] for c in scan if c["in_core_suite"]),
                 "runs_out_of_suite": sum(c["n"] for c in scan if not c["in_core_suite"]),
                 "cost_out_of_suite": sum(c["cost_total"] for c in scan if not c["in_core_suite"]),
                 "out_of_suite_cells": [f"{c['dataset']}|{c['arm']}" for c in scan if not c["in_core_suite"]],
                 "success": sum(c["outcomes"].get("SUCCESS", 0) for c in scan),
                 "fail": sum(c["outcomes"].get("FAIL", 0) for c in scan),
                 "d_suite_cells": len(scan_d_suite), "a_suite_cells": len(scan_a),
                 "d_cells_sd_gt_0_01": sorted(c["dataset"] for c in scan_d_suite if (c["sd"] or 0) > 0.01),
                 "d_cells_sd_ge_0_015": sorted(c["dataset"] for c in scan_d_suite if (c["sd"] or 0) >= 0.015)},
        "heuristic": {"target_se": target_se, "pooled": pooled, "governing_sd": governing,
                      "worst_cell": worst_key, "worst_sd": governing[worst_key],
                      "n_required_worst": (governing[worst_key] / target_se) ** 2,
                      "n_chosen": math.ceil((governing[worst_key] / target_se) ** 2),
                      "se_at_n6": governing[worst_key] / math.sqrt(6),
                      "strict_single_max_cell": f"{single_max['dataset']}|{single_max['arm']}|{single_max['phase']}",
                      "strict_single_max_sd": single_max["sd"],
                      "strict_n_required": (single_max["sd"] / target_se) ** 2,
                      "calib_credit_g_D_over_A_sd_ratio": None,
                      "a_cells_max_sd": max(c["sd"] for c in all_a)},
    }
    cg = {c["arm"]: c["sd"] for c in calib if c["dataset"] == "credit_g"}
    if cg.get("hard_fsm"):
        out["heuristic"]["calib_credit_g_D_over_A_sd_ratio"] = cg["naked_react"] / cg["hard_fsm"]
    out["calibration"]["mean_diff_A_minus_D"] = {
        ds: next(c["mean"] for c in calib if c["dataset"] == ds and c["arm"] == "hard_fsm")
        - next(c["mean"] for c in calib if c["dataset"] == ds and c["arm"] == "naked_react")
        for ds in out["calibration"]["datasets"]}
    out["calibration"]["cost_per_run_by_arm"] = {
        a: sum(c["cost_total"] for c in calib if c["arm"] == a) / sum(c["n"] for c in calib if c["arm"] == a)
        for a in out["calibration"]["arms"]}
    return out


def terciles(core_man, core_runs, cells, arms):
    ranked = sorted(core_man, key=lambda ds: -core_man[ds]["baseline_f1_cv"])
    assign = {}
    for i, ds in enumerate(ranked):
        assign[ds] = "easy" if i < 4 else "medium" if i < 8 else "hard"
    means = cell_means(cells)
    base = {ds: next(r["baseline_f1"] for r in core_runs if r["dataset"] == ds) for ds in core_man}
    out = {"assignment": assign, "order_by_cv": ranked,
           "baseline_cv": {ds: core_man[ds]["baseline_f1_cv"] for ds in ranked}, "per_tercile": {}}
    for t in ["easy", "medium", "hard"]:
        dss = [ds for ds in ranked if assign[ds] == t]
        res = {"datasets": dss, "arms": {}}
        for a in arms:
            ranks = []
            for ds in dss:
                row = [means[ds][x] for x in arms]
                r = am.rank_matrix(np.array([row]))[0]
                ranks.append(r[arms.index(a)])
            rs = [r for r in core_runs if r["arm"] == a and r["dataset"] in dss]
            res["arms"][a] = {"mean_rank": mean(ranks),
                              "mean_f1": mean([means[ds][a] for ds in dss]),
                              "mean_f1_minus_baseline": mean([means[ds][a] - base[ds] for ds in dss]),
                              "success": sum(r["success"] for r in rs), "n": len(rs),
                              "sig_better": sum(r["sig_better"] for r in rs)}
        out["per_tercile"][t] = res
    return out


def phases(runs, arms):
    out = {}
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        if not rs:
            continue
        att_tot = {p: sum(r["attempts"].get(p, 0) for r in rs) for p in PHASES}
        loops = collections.Counter()
        runs_with = collections.Counter()
        gate_fail = collections.Counter()
        du_fail_msgs = collections.Counter()
        model_fail_msgs = collections.Counter()
        eval_loop_runs, spec_runs = [], []
        for r in rs:
            seq = r["phase_seq"]
            flags = set()
            for prev, cur in zip(seq, seq[1:]):
                if cur == prev:
                    loops[f"retry_{cur}"] += 1
                    flags.add(f"retry_{cur}")
                elif cur in PHASES and prev in PHASES and PHASES.index(cur) < PHASES.index(prev):
                    loops[f"back_{prev}_to_{cur}"] += 1
                    flags.add(f"back_{prev}_to_{cur}")
                    flags.add("any_backjump")
            if any(v > 1 for k, v in r["attempts"].items() if k in PHASES):
                flags.add("any_repeat")
            for f in flags:
                runs_with[f] += 1
            if "back_eval_to_model" in flags:
                eval_loop_runs.append(r["run_id"])
            if "back_du_to_bu" in flags:
                spec_runs.append(r["run_id"])
            for e in r["events"]:
                if e.get("gate_passed") is False:
                    gate_fail[e["phase"]] += 1
                    fb = (e.get("gate_feedback") or "")
                    if e["phase"] == "du":
                        # A: "Target column '<t>' must not be excluded." (gates.check_du);
                        # B: Pruefer-Freitext, z. B. "criterion 7 ... target ... excluded_columns"
                        low = fb.lower()
                        du_fail_msgs["target_excluded" if ("target" in low and "exclud" in low) else fb[:60]] += 1
                    if e["phase"] == "model" and a == "hard_fsm":
                        model_fail_msgs[fb[:45]] += 1
        turns = [r["worker_turns_events"] for r in rs]
        calls = [r["total_calls"] for r in rs]
        out[a] = {
            "runs": len(rs),
            "attempts_total": att_tot,
            "attempts_mean": {p: att_tot[p] / len(rs) for p in PHASES},
            "loops": dict(loops), "runs_with": dict(runs_with),
            "eval_to_model_runs": eval_loop_runs, "du_to_bu_runs": spec_runs,
            "gate_fail_events_by_phase": dict(gate_fail),
            "du_gate_fail_feedback": dict(du_fail_msgs),
            "model_gate_fail_feedback_A": dict(model_fail_msgs),
            "total_calls_mean": mean(calls), "total_calls_max": max(calls), "total_calls_min": min(calls),
            "gate_calls_total": sum(r["gate_calls"] or 0 for r in rs),
            "gate_calls_mean": mean([r["gate_calls"] or 0 for r in rs]),
            "worker_turns_mean": mean(turns), "worker_turns_median": median(turns), "worker_turns_max": max(turns),
            "worker_turns_summary_equals_events": sum(r["worker_turns_summary"] == r["worker_turns_events"] for r in rs),
            "budget_exhausted": sum(bool(r["budget_exhausted"]) for r in rs),
            "bu_correct": sum(r["bu_correct"] is True for r in rs),
            "bu_correct_first": sum(r["bu_correct_first"] is True for r in rs),
            "target_ok": sum(r["target_ok"] is True for r in rs),
            "per_dataset_backjump_runs": dict(collections.Counter(
                r["dataset"] for r in rs if any(f.startswith("back_") for f in _flags(r)))),
        }
    return out


def _flags(r):
    seq = r["phase_seq"]
    fl = set()
    for prev, cur in zip(seq, seq[1:]):
        if cur in PHASES and prev in PHASES and PHASES.index(cur) < PHASES.index(prev):
            fl.add(f"back_{prev}_to_{cur}")
    return fl


def du_exclusions(runs):
    out = {}
    for ds in ["bank_marketing", "eucalyptus"]:
        out[ds] = {}
        for a in FSM_ARMS:
            rs = [r for r in runs if r["arm"] == a and r["dataset"] == ds]
            cnt = collections.Counter(c for r in rs for c in r["du_excluded"])
            out[ds][a] = {"runs": len(rs), "excluded_column_counts": dict(cnt.most_common()),
                          "per_run": {r["run_id"]: r["du_excluded"] for r in rs},
                          "f1_mean": mean([r["f1"] for r in rs])}
    v12 = {a: out["bank_marketing"][a]["excluded_column_counts"].get("V12", 0) for a in FSM_ARMS}
    out["v12_excluded_A_B"] = v12
    return out


def leakage(leak_runs, spur_runs, core_runs):
    out = {}
    core_d = {ds: mean([r["f1"] for r in core_runs if r["arm"] == "naked_react" and r["dataset"] == ds])
              for ds in {r["dataset"] for r in core_runs}}
    for name, runs in [("perfect", leak_runs), ("spurious", spur_runs)]:
        per_set = {}
        for ds in sorted({r["dataset"] for r in runs}):
            rs = sorted([r for r in runs if r["dataset"] == ds], key=lambda r: r["rep"])
            per_set[ds] = {"runs": len(rs), "column": rs[0]["leak_column"],
                           "exploited": sum(bool(r["leak_exploited"]) for r in rs),
                           "exploited_runs": [r["run_id"] for r in rs if r["leak_exploited"]],
                           "kappas": [r["leak_kappa"] for r in rs], "kappa_mean": mean([r["leak_kappa"] for r in rs]),
                           "kappa_max": max(r["leak_kappa"] for r in rs),
                           "success": sum(r["success"] for r in rs),
                           "fail_reasons": dict(collections.Counter(r["fail_reason"] for r in rs if r["fail_reason"])),
                           "f1_mean": mean([r["f1"] for r in rs if r["f1"] is not None]),
                           "f1_core_D_mean": core_d.get(ds),
                           "delta_f1_vs_core_D": (mean([r["f1"] for r in rs if r["f1"] is not None]) - core_d[ds])
                           if ds in core_d else None,
                           "cost": sum(r["cost"] for r in rs)}
        exploited = [r for r in runs if r["leak_exploited"]]
        k, n = len(exploited), len(runs)
        out[name] = {"arms": sorted({r["arm"] for r in runs}), "runs": n, "exploited": k,
                     "exploited_wilson": list(am.wilson(k, n)),
                     "exploited_detail": [{"run_id": r["run_id"], "kappa": r["leak_kappa"], "z": r["leak_z"],
                                           "success": r["success"], "diff": r["diff"], "ci_hi": r["ci_hi"]}
                                          for r in exploited],
                     "success": sum(r["success"] for r in runs), "fail": sum(not r["success"] for r in runs),
                     "fail_reasons": dict(collections.Counter(r["fail_reason"] for r in runs if r["fail_reason"])),
                     "cost": sum(r["cost"] for r in runs),
                     "rule_threshold": 8, "rule_triggered": k >= 8,
                     "kappa_max_not_exploited": max(r["leak_kappa"] for r in runs if not r["leak_exploited"]),
                     "per_set": per_set,
                     "harness_commits": sorted({r["harness"].get("commit", "")[:7] for r in runs})}
    return out


def costs(batches):
    out = {}
    for name, runs in batches.items():
        per_arm = {}
        for a in ARMS:
            rs = [r for r in runs if r["arm"] == a]
            if not rs:
                continue
            tok = {k: sum(r["usage"].get(k, 0) for r in rs) for k in TOKEN_KEYS}
            gate = sum(r["gate_cost_events"] for r in rs)
            worker = sum(r["worker_cost_events"] for r in rs)
            total = sum(r["cost"] for r in rs)
            tok_cost = {k: tok[k] * PRICE[k.replace("_tokens", "")] for k in TOKEN_KEYS}
            per_arm[a] = {
                "runs": len(rs), "cost_total": total, "cost_per_run_mean": total / len(rs),
                "cost_per_run_median": median([r["cost"] for r in rs]),
                "cost_per_run_max": max(r["cost"] for r in rs),
                "tokens": tok, "tokens_per_run": {k: v / len(rs) for k, v in tok.items()},
                "tokens_total": sum(tok.values()),
                "cost_by_token_type": tok_cost,
                "cost_share_by_token_type": {k: v / total for k, v in tok_cost.items()},
                "worker_cost_events": worker, "gate_cost_events": gate,
                "gate_cost_share": gate / total if total else None,
                "repriced_equals_summary": abs(worker + gate - total) < 1e-6,
                "calls_mean": mean([r["total_calls"] for r in rs]),
            }
        ratios = {}
        for x, y in itertools.product(FSM_ARMS, REACT_ARMS):
            if x in per_arm and y in per_arm:
                ratios[f"{LETTER[x]}/{LETTER[y]}"] = per_arm[x]["cost_per_run_mean"] / per_arm[y]["cost_per_run_mean"]
        if "soft_fsm" in per_arm and "hard_fsm" in per_arm:
            ratios["B/A"] = per_arm["soft_fsm"]["cost_per_run_mean"] / per_arm["hard_fsm"]["cost_per_run_mean"]
        if "crisp_react" in per_arm and "naked_react" in per_arm:
            ratios["C/D"] = per_arm["crisp_react"]["cost_per_run_mean"] / per_arm["naked_react"]["cost_per_run_mean"]
        out[name] = {"per_arm": per_arm, "ratios_cost_per_run": ratios,
                     "total": sum(r["cost"] for r in runs), "runs": len(runs)}
    out["_all_clean_batches_total"] = sum(v["total"] for k, v in out.items() if not k.startswith("_"))
    out["_pricing_usd_per_mtok"] = {k: v * 1e6 for k, v in PRICE.items()}
    return out


def diff_to_baseline(runs, arms):
    return {a: {"median_diff": median([r["diff"] for r in runs if r["arm"] == a and r["diff"] is not None]),
                "mean_diff": mean([r["diff"] for r in runs if r["arm"] == a and r["diff"] is not None]),
                "n": sum(1 for r in runs if r["arm"] == a and r["diff"] is not None)} for a in arms}


def postcutoff(post_runs, core_runs, arms, runs_dir):
    l1 = level1(post_runs, arms)
    ab = sum(l1["per_arm"][a]["success"] for a in FSM_ARMS), sum(l1["per_arm"][a]["n"] for a in FSM_ARMS)
    cd = sum(l1["per_arm"][a]["success"] for a in REACT_ARMS), sum(l1["per_arm"][a]["n"] for a in REACT_ARMS)
    pooled = {"AB": list(ab), "CD": list(cd), **fisher(ab[0], ab[1], cd[0], cd[1]),
              "note": "post hoc gruppiert (nach Sicht der Daten), nur explorativ"}
    l2, cells = level2(post_runs, arms)
    rsd = run_sd_table(cells, arms)
    means = cell_means(cells)
    base = {ds: next(r["baseline_f1"] for r in post_runs if r["dataset"] == ds) for ds in means}
    fails = [{"run_id": r["run_id"], "arm": r["arm"], "dataset": r["dataset"], "f1": r["f1"], "diff": r["diff"],
              "ci_hi": r["ci_hi"], "fail_reason": r["fail_reason"]} for r in post_runs if not r["success"]]
    eval_gate_runs = [r["run_id"] for r in post_runs if r["arm"] in FSM_ARMS and "back_eval_to_model" in _flags(r)]
    ana = jload(os.path.join(runs_dir, "_post_cutoff", "analysis", "analysis.json"))
    return {
        "level1": l1, "pooled_fsm_vs_react": pooled,
        "secondary": secondary(post_runs, arms),
        "f1_cell_means": means, "baseline_test": base,
        "f1_cell_minus_baseline": {ds: {a: means[ds][a] - base[ds] for a in arms} for ds in means},
        "run_sd": rsd,
        "friedman_f1_measured": {k: l2["f1_measured"][k] for k in ("friedman", "cd", "avg_ranks")},
        "median_diff_to_baseline_post": diff_to_baseline(post_runs, arms),
        "median_diff_to_baseline_core": diff_to_baseline(core_runs, arms),
        "fails": fails,
        "fails_by_arm_set": dict(collections.Counter(f"{f['arm']}|{f['dataset']}" for f in fails)),
        "eval_to_model_runs_fsm": eval_gate_runs,
        "eval_to_model_runs_fsm_by_set": dict(collections.Counter(x.split("-")[1] for x in eval_gate_runs)),
        "fsm_runs_without_eval_loop_student_health": sorted(
            r["run_id"] for r in post_runs if r["arm"] in FSM_ARMS and r["dataset"] == "student_health"
            and "back_eval_to_model" not in _flags(r)),
        "class_balance": {ds: next((r["class_balance"] for r in post_runs if r["dataset"] == ds and r["class_balance"]), None)
                          for ds in means},
        "check_vs_analysis_json": {
            "success_equal": all(ana["rates"]["per_arm"][a]["success"] == l1["per_arm"][a]["success"] for a in arms),
            "ranks_equal": all(abs(ana["metrics"]["f1_measured"]["avg_ranks"][a] - l2["f1_measured"]["avg_ranks"][a]) < 1e-9
                               for a in arms)},
    }


def contamination(runs_dir):
    pref = collections.Counter()
    for d in os.listdir(runs_dir):
        if not os.path.isdir(os.path.join(runs_dir, d)):
            continue
        for p in ("promptv1_main-", "automem_main-", "automem_leak-", "automem_spur-", "gitctx_leak-",
                  "automem__", "gitctx__"):
            if d.startswith(p):
                pref[p] += 1
    promptv1_arms = collections.Counter(d.split("-")[2] for d in os.listdir(runs_dir) if d.startswith("promptv1_main-"))
    automem_main = [d for d in os.listdir(runs_dir) if d.startswith("automem_main-")]
    automem_post = [d for d in automem_main if d.split("-")[1] in ("student_health", "ev_purchase")]
    gitctx = sorted(d for d in os.listdir(runs_dir) if d.startswith("gitctx_leak-"))
    gitctx_with_summary = [d for d in gitctx if os.path.exists(os.path.join(runs_dir, d, "summary.json"))]
    snap = sorted(os.listdir(os.path.join(runs_dir, "_auto_memory_snapshot_2026-10-24")))
    # benannter Vergleich (Kap. 5): archivierte Aggregate, NICHT als Ergebnis zitieren
    old = {}
    for sub in ("automem__main", "automem__post_cutoff"):
        a = jload(os.path.join(runs_dir, sub, "analysis", "analysis.json"))
        m = jload(os.path.join(runs_dir, sub, "main_runs.json"))
        fm = a["metrics"]["f1_measured"]
        old[sub] = {"success": {k: v["success"] for k, v in a["rates"]["per_arm"].items()},
                    "n": {k: v["n"] for k, v in a["rates"]["per_arm"].items()},
                    "friedman_p_chi2": fm["friedman"]["p_chi2"], "friedman_p_F": fm["friedman"]["p_F"],
                    "avg_ranks": fm["avg_ranks"], "run_sd_median": {k: v["median_sd"] for k, v in a["run_sd"].items()},
                    "cost_total": sum(c["cost_total"] for c in m["cells"]),
                    "cell_means": {ds: dict(zip(a["arms"], row))
                                   for ds, row in zip(fm["datasets"], fm["matrix"])},
                    "generated_at": m.get("generated_at")}
    sig = collections.Counter()
    nn = collections.Counter()
    for d in automem_main:
        if d in automem_post:
            continue
        s = jload(os.path.join(runs_dir, d, "summary.json"))
        ci = (s.get("baseline_comparison") or {}).get("ci_lo")
        sig[s["arm"]] += bool(ci is not None and ci > 0)
        nn[s["arm"]] += 1
    old["automem__main"]["sig_better"] = dict(sig)
    old["automem__main"]["sig_better_n"] = dict(nn)
    sb_pairs = []
    for x, y in itertools.combinations(ARMS, 2):
        sb_pairs.append({"a": x, "b": y, **fisher(sig[x], nn[x], sig[y], nn[y])})
    for p_, h in zip(sb_pairs, holm([q["p"] for q in sb_pairs])):
        p_["p_holm"] = h
    old["automem__main"]["sig_better_pairs"] = sb_pairs
    return {"archived_dir_counts": dict(pref), "promptv1_by_arm": dict(promptv1_arms),
            "automem_main_runs": len(automem_main), "automem_post_cutoff_runs": len(automem_post),
            "automem_core_runs": len(automem_main) - len(automem_post),
            "automem_total_runs": len(automem_main) + pref["automem_leak-"] + pref["automem_spur-"],
            "gitctx_dirs": len(gitctx), "gitctx_with_summary": len(gitctx_with_summary),
            "gitctx_without_summary": sorted(set(gitctx) - set(gitctx_with_summary)),
            "snapshot_files": len(snap), "snapshot_names": snap,
            "archived_comparison": old}


def expectations(core_runs, post_runs, core_l1, post, core_l2, core_rsd, leak, terc, du):
    """Praeregistrierte Erwartungs-Muster (14.09., Protokoll Punkt 8): Messwerte je Muster."""
    fm = core_l2["f1_measured"]
    ad = next(c for c in fm["cliffs"] if c["a"] == "hard_fsm" and c["b"] == "naked_react")
    ad_nem = next(p for p in fm["nemenyi"] if p["a"] == "hard_fsm" and p["b"] == "naked_react")
    cb = {}
    for ds in sorted({r["dataset"] for r in core_runs + post_runs}):
        bal = next((r["class_balance"] for r in core_runs + post_runs
                    if r["dataset"] == ds and r["class_balance"]), None)
        if bal:
            tot = sum(bal.values())
            cb[ds] = max(bal.values()) / tot
    succ_by_set = {}
    for runs in (core_runs, post_runs):
        for ds in sorted({r["dataset"] for r in runs}):
            succ_by_set[ds] = {a: sum(r["success"] for r in runs if r["arm"] == a and r["dataset"] == ds) for a in ARMS}
    pair = lambda l1, a, b: next(p for p in l1["pairs"] if p["a"] == a and p["b"] == b)  # noqa: E731
    return {
        "p1_parity_clean": {
            "core": {"friedman_p_F": fm["friedman"]["p_F"], "rank_A": fm["avg_ranks"]["hard_fsm"],
                     "rank_D": fm["avg_ranks"]["naked_react"], "rank_diff_AD": ad_nem["rank_diff"],
                     "cd": fm["cd"], "nemenyi_sig_AD": ad_nem["significant"], "cliffs_mean_AD": ad["mean_delta"],
                     "cliffs_mag_AD": ad["mean_magnitude"]},
            "post": {"f1_A": {ds: post["f1_cell_means"][ds]["hard_fsm"] for ds in post["f1_cell_means"]},
                     "f1_D": {ds: post["f1_cell_means"][ds]["naked_react"] for ds in post["f1_cell_means"]},
                     "friedman_p_F": post["friedman_f1_measured"]["friedman"]["p_F"],
                     "cd": post["friedman_f1_measured"]["cd"]}},
        "p2_traps": {"leak_perfect_D_exploited": leak["perfect"]["exploited"],
                     "leak_spurious_D_exploited": leak["spurious"]["exploited"],
                     "A_B_C_on_leakage_run": False, "poisoned_tasks_run": False,
                     "v12_bank_marketing_A_excluded": du["v12_excluded_A_B"]["hard_fsm"],
                     "bank_marketing_f1": {a: core_l2["_means"]["bank_marketing"][a] for a in ARMS},
                     "bank_marketing_success": succ_by_set["bank_marketing"]},
        "p3_runsd_A_lt_D": {"core_median_A": core_rsd["per_arm"]["hard_fsm"]["median_sd"],
                            "core_median_D": core_rsd["per_arm"]["naked_react"]["median_sd"],
                            "core_cells_A_lt_D": sum(1 for ds, row in core_rsd["per_cell"].items()
                                                     if row["hard_fsm"] < row["naked_react"]),
                            "core_cells": len(core_rsd["per_cell"]),
                            "post_median_A": post["run_sd"]["per_arm"]["hard_fsm"]["median_sd"],
                            "post_median_D": post["run_sd"]["per_arm"]["naked_react"]["median_sd"]},
        "p4_imbalanced_success_A_gt_D": {
            "majority_share": cb, "success_by_set": succ_by_set,
            "core_only_nonpass": [f["run_id"] for c in core_l1["per_cell"] for f in c["fails"]],
            "core_fisher_AD": pair(core_l1, "hard_fsm", "naked_react"),
            "post_fisher_AD": pair(post["level1"], "hard_fsm", "naked_react")},
        "p5_refinement_highdim": {
            "cnae_9": core_l2["_means"]["cnae_9"], "fashion_mnist": core_l2["_means"]["fashion_mnist"],
            "tercile_hard_ranks": {a: terc["per_tercile"]["hard"]["arms"][a]["mean_rank"] for a in ARMS}},
    }


# --------------------------------------------------------------------------- main

def run_median(runs_dir):
    out = {}
    for sub, name in (("_main", "median_core"), ("_post_cutoff", "median_postcutoff")):
        od = os.path.join(OUT, name)
        cmd = [sys.executable, os.path.join(REPO, "scripts", "analyze_main.py"),
               "--input", os.path.join(runs_dir, sub, "main_runs.json"), "--out-dir", od, "--agg", "median"]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit(f"analyze_main --agg median fehlgeschlagen ({sub})")
        a = jload(os.path.join(od, "analysis.json"))
        out[name] = {m: {"friedman": a["metrics"][m]["friedman"], "avg_ranks": a["metrics"][m].get("avg_ranks"),
                         "cd": a["metrics"][m]["cd"], "matrix": a["metrics"][m]["matrix"],
                         "datasets": a["metrics"][m]["datasets"],
                         "nemenyi_any_sig": any(p["significant"] for p in (a["metrics"][m]["nemenyi"] or []))}
                     for m in a["metrics"]}
        out[name]["_command"] = " ".join(["uv run python scripts/analyze_main.py", "--input",
                                          f"../thesis-crisp-dm-runs/{sub}/main_runs.json", "--out-dir",
                                          f"thesis/facts/derived/{name}", "--agg median"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(REPO), "thesis-crisp-dm-runs"))
    ap.add_argument("--skip-median", action="store_true", help="Median-Sensitivitaet nicht neu erzeugen")
    args = ap.parse_args()
    runs_dir = os.path.abspath(args.runs_dir)
    core_man, post_man = manifests()
    print("lade Runs ...")
    core = load_batch(runs_dir, "main", set(core_man), "core")
    post = load_batch(runs_dir, "main", set(post_man), "post")
    leak = load_batch(runs_dir, "leak", set(core_man), "leak_perfect")
    spur = load_batch(runs_dir, "spur", set(core_man), "leak_spurious")
    batches = {"leak_perfect": leak, "leak_spurious": spur, "core": core, "post_cutoff": post}
    print({k: len(v) for k, v in batches.items()})

    # --- Kap. 5
    jdump("controls.json", controls(batches))
    jdump("kgit.json", {"runs_with_git_calls": kgit(batches), "scanned_runs": sum(len(v) for v in batches.values())})
    jdump("timeline.json", timeline(batches, runs_dir))
    jdump("calibration.json", clean(calibration(runs_dir, set(core_man))))
    jdump("contamination.json", clean(contamination(runs_dir)))
    jdump("splits.json", splits(runs_dir))
    plan = jload(os.path.join(runs_dir, "_main", "plan.json"))
    agg = {}
    for sub in ("_main", "_post_cutoff", "_leakage", "_spurious"):
        m = jload(os.path.join(runs_dir, sub, "main_runs.json"))
        agg[sub] = {"generated_at": m.get("generated_at"), "replicates": m.get("replicates"),
                    "leakage": m.get("leakage"), "arms": m.get("arms"), "cells": len(m["cells"]),
                    "rescored_runs": sum(c.get("rescored", 0) for c in m["cells"]),
                    "contaminated_runs": sum(c.get("contaminated", 0) for c in m["cells"]),
                    "missing_runs": sum(c.get("missing", 0) for c in m["cells"]),
                    "n_measured": sum(c.get("n_measured", 0) for c in m["cells"])}
    n_cells = len({(j["arm"], j["dataset"]) for j in plan})
    jdump("design_checks.json", {
        "core_plan": {"jobs": len(plan), "cells": n_cells,
                      "replicate_major": all(plan[i]["replicate"] == 1 + i // n_cells for i in range(len(plan))),
                      "arm_order_within_dataset": [j["arm"] for j in plan[:4]],
                      "datasets_in_plan_order_r1": [j["dataset"] for j in plan[:n_cells:4]]},
        "aggregates": agg,
        "manifests_with_poisoned_variant": sum(1 for m in list(core_man.values()) + list(post_man.values())
                                               if m.get("poisoned_variant")),
        "manifests_total": len(core_man) + len(post_man),
        "config": code_constants(),
    })

    # --- Kap. 6 Kern
    l1 = level1(core, ARMS)
    l2, cells = level2(core, ARMS)
    means = cell_means(cells)
    l2["_means"] = means
    base = {ds: next(r["baseline_f1"] for r in core if r["dataset"] == ds) for ds in core_man}
    ana = jload(os.path.join(runs_dir, "_main", "analysis", "analysis.json"))
    checks = {
        "level1_equal_analysis_json": all(ana["rates"]["per_arm"][a]["success"] == l1["per_arm"][a]["success"] for a in ARMS),
        "matrix_equal_analysis_json": all(
            abs(x - y) < 1e-12 for m in am.METRICS
            for row_a, row_b in zip(ana["metrics"][m]["matrix"], l2[m]["matrix"].tolist() if hasattr(l2[m]["matrix"], "tolist") else l2[m]["matrix"])
            for x, y in zip(row_a, row_b)),
        "friedman_equal_analysis_json": all(
            abs(ana["metrics"][m]["friedman"]["p_F"] - l2[m]["friedman"]["p_F"]) < 1e-12 for m in am.METRICS),
        "baseline_equals_manifest_3dp": {ds: round(base[ds], 3) == round(core_man[ds]["baseline_f1"], 3) for ds in base},
        "baseline_unique_per_set": all(len({round(r["baseline_f1"], 12) for r in core if r["dataset"] == ds}) == 1 for ds in base),
    }
    core_out = {
        "level1": l1,
        "level2": {m: {k: v for k, v in res.items()} for m, res in l2.items() if m in am.METRICS},
        "cell_means_f1_measured": means,
        "cell_means_f1_combined": cell_means(cells, "f1_combined"),
        "cell_means_f1_on_success": cell_means(cells, "f1_on_success"),
        "baseline_test": base,
        "baseline_val": {ds: core_man[ds]["baseline_f1_val"] for ds in core_man},
        "baseline_cv": {ds: core_man[ds]["baseline_f1_cv"] for ds in core_man},
        "best_arm_per_set_3dp": {ds: [a for a in ARMS if round(means[ds][a], 3) == max(round(v, 3) for v in means[ds].values())]
                                 for ds in means},
        "sets_where_arm_best_3dp": {a: sum(1 for ds in means if round(means[ds][a], 3) == max(round(v, 3) for v in means[ds].values()))
                                    for a in ARMS},
        "max_cell_spread_per_set": {ds: max(means[ds].values()) - min(means[ds].values()) for ds in means},
        "median_diff_to_baseline": diff_to_baseline(core, ARMS),
        "checks": checks,
    }
    jdump("core.json", clean(json.loads(json.dumps(core_out, default=_default))))
    jdump("secondary_core.json", clean(secondary(core, ARMS)))
    rsd = run_sd_table(cells, ARMS)
    jdump("runsd_core.json", clean(rsd))
    terc = terciles(core_man, core, cells, ARMS)
    jdump("terciles.json", clean(json.loads(json.dumps(terc, default=_default))))
    pc = postcutoff(post, core, ARMS, runs_dir)
    jdump("postcutoff.json", clean(json.loads(json.dumps(pc, default=_default))))
    ph = {"core": phases(core, FSM_ARMS), "post_cutoff": phases(post, FSM_ARMS),
          "react_turns": {b: {a: {"mean": mean([r["worker_turns_events"] for r in rs if r["arm"] == a]),
                                  "median": median([r["worker_turns_events"] for r in rs if r["arm"] == a]),
                                  "max": max(r["worker_turns_events"] for r in rs if r["arm"] == a),
                                  "summary_field_zero": sum(r["worker_turns_summary"] == 0 for r in rs if r["arm"] == a),
                                  "runs": sum(1 for r in rs if r["arm"] == a)}
                              for a in REACT_ARMS}
                          for b, rs in (("core", core), ("post_cutoff", post))}}
    jdump("phases.json", clean(ph))
    du = du_exclusions(core)
    jdump("du_exclusions.json", clean(du))
    lk = leakage(leak, spur, core)
    jdump("leakage.json", clean(lk))
    jdump("costs.json", clean(costs(batches)))
    if not args.skip_median:
        jdump("median.json", clean(json.loads(json.dumps(run_median(runs_dir), default=_default))))
    ex = expectations(core, post, l1, pc, l2, rsd, lk, terc, du)
    jdump("expectations.json", clean(json.loads(json.dumps(ex, default=_default))))
    print("checks:", json.dumps(checks))


if __name__ == "__main__":
    main()
