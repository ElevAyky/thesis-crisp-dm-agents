# scripts/main_runs.py
"""Haupt-Runs der Thesis (roadmap §4, Decker 14.09.): 4 Arme x 13 Sets x n = 6
= 312 Runs, Opus 4.8, resumebar. Konfirmatorische Daten fuer H2 (Friedman +
Nemenyi ueber die Datasets, experiment-design.md §5).

Design:
- **Replikat-weise balanciert:** Replikat 1 ueber ALLE Zellen, dann Replikat 2, ...
  Ein Zwischenstand (z. B. am 05.10.) hat damit in jeder Zelle gleich viele
  Runs -> CD-Diagramm auf Teildaten moeglich, ohne Arme/Sets zu bevorzugen.
- **Resumebar:** deterministische run_ids (main-<set>-<arm>-r<k>); ein
  gemessenes summary.json wird uebersprungen. ERROR-Summaries (Harness-Crash,
  rate_limited) gelten als NICHT gemessen und werden neu gefahren.
- **Abo-Kontingent:** run.py Exit 75 -> warten (--rate-limit-wait) und denselben
  Run wiederholen (variance_scan.execute_with_rate_limit_wait).
- **Auswertung** pro Zelle: SUCCESS-Rate, F1-auf-Erfolg (final_f1), volle
  F1-Verteilung (best_f1_measured), kombinierter Score (Fail = 0),
  fail_reason-Zaehlung, Kosten. Kontaminierte Runs (holdout_intact=false)
  fliegen raus, bleiben aber gezaehlt.

Aufruf (Repo-Root):
    uv run python scripts/main_runs.py --dry-run             # Plan + Kosten
    uv run python scripts/main_runs.py                       # alles (resumebar)
    uv run python scripts/main_runs.py --replicates 2        # nur r1+r2 (Zwischenstand)
    uv run python scripts/main_runs.py --aggregate-only      # Tabelle neu bauen
    uv run python scripts/main_runs.py --arms hard_fsm naked_react --datasets cmc wine
    uv run python scripts/main_runs.py --manifests-dir manifests/post_cutoff   # -> _post_cutoff/
    # H1-Nachtrag (docs/h1-single-session.md): eigene run_ids h1-* + eigener Ausgabeordner,
    # immer BEIDE Arme angeben (main_runs.json enthaelt nur die Arme des Aufrufs)
    uv run python scripts/main_runs.py --batch h1 --arms hard_fsm_single hard_fsm       # -> _h1/
    uv run python scripts/main_runs.py --batch h1 --arms hard_fsm_single hard_fsm \
        --manifests-dir manifests/post_cutoff                                          # -> _post_cutoff_h1/

Ergebnis: <runs-dir>/_main/main_runs.json (+ plan.json, logs/). Die
Statistik (Friedman/Nemenyi/CD) liest diese Datei — eigenes Script.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_scan import (execute_with_rate_limit_wait, is_measured,  # noqa: E402
                           RATE_LIMIT_WAIT_S)

ARMS = ("hard_fsm", "soft_fsm", "crisp_react", "naked_react")
# H1-Nachtrag (13.11.): nur explizit per --arms waehlbar, nie im Default-Plan der 4 Arme
EXTRA_ARMS = ("hard_fsm_single",)
ARM_ORDER = ARMS + EXTRA_ARMS
DEFAULT_REPLICATES = 6            # experiment-design.md §5.2, Decker 14.09.
LEAKAGE_REPLICATES = 3            # docs/leakage-runs.md (Decker 09.10.)
DEFAULT_MODEL = "claude-opus-4-8"  # modellwahl.md, Decker 14.09.

# Kostenanker pro Run (Probe-Runs 18.09. mit k = 1: A auf schweren Sets ~2,2 $,
# leichte ~0,7 $; B ~ A + Pruefer; C/D 0,1-0,4 $). Nur fuer die dry-run-Schaetzung.
# hard_fsm_single: A + ~0,3-0,65 $ fuer den mitwachsenden Kontext der einen Session
# (Review 13.11., aus den A-Transkripten hochgerechnet)
COST_PER_RUN = {"hard_fsm": 1.4, "soft_fsm": 1.8, "crisp_react": 0.35, "naked_react": 0.25,
                "hard_fsm_single": 1.8}

# Zusatz-Batches mit eigenem run_id-Praefix (sonst haelt der Resume gemessene Kern-Runs
# derselben Zelle fuer erledigt) und eigenem Ausgabeordner (sonst ueberschreibt die
# Aggregation _main/ bzw. _post_cutoff/). h1 = H1-Nachtrag, docs/h1-single-session.md.
BATCHES = ("h1",)


def suite_datasets(manifests_dir: str) -> list[str]:
    """Alle Manifeste = die kuratierte Suite (13, benchmarks.md §2). Post-Cutoff-
    Sets kommen als eigener Kontroll-Arm dazu, sobald kuratiert (dann --datasets)."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(manifests_dir, "*.yaml")))


# Leakage-Batches (docs/leakage-runs.md): Variante -> (run_id-Praefix, Ausgabe-Suffix).
# Jede Batch faehrt dieselben Zellen wie der Kern — eigenes Praefix, sonst haelt der
# Resume die gemessenen Runs einer anderen Batch fuer erledigt.
LEAKAGE_VARIANTS = {"perfect": ("leak", "leakage"), "spurious": ("spur", "spurious")}


def _variant(leakage) -> str | None:
    """leakage: False/None = Kern, True = 'perfect' (Batch 1), sonst der Variantenname."""
    return "perfect" if leakage is True else (leakage or None)


def out_subdir(manifests_dir: str, leakage=False, batch: str | None = None) -> str:
    """Kern-Suite (manifests/) -> _main; jeder andere Manifest-Ordner bekommt einen
    eigenen Ausgabeordner (manifests/post_cutoff -> _post_cutoff), damit ein
    Zusatzlauf die Kern-Aggregation nicht ueberschreibt. Leakage-Batches: _leakage /
    _spurious bzw. _<ordner>_<suffix>. Zusatz-Batches (--batch h1): _h1 bzw.
    _<ordner>_h1."""
    base = os.path.basename(os.path.normpath(manifests_dir.replace("\\", "/")))
    variant = _variant(leakage)
    if batch and variant:
        raise ValueError("--batch und --leakage schliessen sich aus")
    suffix = LEAKAGE_VARIANTS[variant][1] if variant else batch
    if base == "manifests":
        return f"_{suffix}" if suffix else "_main"
    return f"_{base}_{suffix}" if suffix else f"_{base}"


def build_plan(datasets: list[str], replicates: int, arms: list[str],
               leakage=False, batch: str | None = None) -> list[dict]:
    variant = _variant(leakage)
    if batch and variant:
        raise ValueError("--batch und --leakage schliessen sich aus")
    plan = []
    for rep in range(1, replicates + 1):
        for ds in datasets:
            for arm in arms:
                job = {"arm": arm, "dataset": ds, "replicate": rep}
                if variant:
                    job["leakage"] = True
                if variant and variant != "perfect":
                    job["leakage_variant"] = variant
                if batch:
                    job["batch"] = batch
                plan.append(job)
    return plan


def run_id_for(job: dict) -> str:
    if job.get("leakage"):
        prefix = LEAKAGE_VARIANTS[job.get("leakage_variant", "perfect")][0]
    elif job.get("batch"):
        prefix = job["batch"]
    else:
        prefix = "main"
    return f"{prefix}-{job['dataset']}-{job['arm']}-r{job['replicate']}"


def summary_path(runs_dir: str, job: dict) -> str:
    return os.path.join(runs_dir, run_id_for(job), "summary.json")


def aggregate(plan: list[dict], runs_dir: str) -> dict[tuple[str, str], dict]:
    cells: dict[tuple[str, str], dict] = {}
    for job in plan:
        key = (job["arm"], job["dataset"])
        c = cells.setdefault(key, {
            "arm": job["arm"], "dataset": job["dataset"],
            "outcomes": [], "f1_on_success": [], "f1_measured": [],
            "combined": [], "fail_reasons": Counter(), "cost": [], "calls": [],
            "missing": 0, "contaminated": 0, "run_ids": [], "rescored": 0,
            "leak_exploited": [], "leak_detected": [], "leak_kappa": [],
        })
        path = summary_path(runs_dir, job)
        if not is_measured(path):
            c["missing"] += 1
            continue
        with open(path) as f:
            s = json.load(f)
        if not s.get("holdout_intact", True):
            c["contaminated"] += 1
            continue
        if s.get("rescore"):
            # Offline-Neubewertung nach der Bootstrap-Regel (scripts/rescore_baseline.py,
            # nur C/D): ersetzt die schwellen-abhaengigen Felder, Rest bleibt.
            r = s["rescore"]
            s = {**s, **{k: r[k] for k in ("outcome", "success", "fail_reason", "final_f1")}}
            c["rescored"] += 1
        c["run_ids"].append(run_id_for(job))
        c["outcomes"].append(s.get("outcome", "?"))
        if s.get("fail_reason"):
            c["fail_reasons"][s["fail_reason"]] += 1
        if s.get("final_f1") is not None:
            c["f1_on_success"].append(s["final_f1"])
        if s.get("best_f1_measured") is not None:
            c["f1_measured"].append(s["best_f1_measured"])
        # kombinierter Score (experiment-design §3.1): Fail/Abstain = 0
        c["combined"].append(s["final_f1"] if s.get("success") and s.get("final_f1") is not None else 0.0)
        cost = s.get("total_cost_usd")
        if isinstance(cost, (int, float)) and cost == cost:
            c["cost"].append(cost)
        if isinstance(s.get("total_calls"), int):
            c["calls"].append(s["total_calls"])
        # Leakage-Runs: Ausnutzung (alle Arme) und DU-Erkennung (nur A/B, sonst None)
        if s.get("leakage_exploited") is not None:
            c["leak_exploited"].append(s["leakage_exploited"])
            c["leak_kappa"].append((s.get("leakage_reliance") or {}).get("kappa"))
        if s.get("leakage_detected") is not None:
            c["leak_detected"].append(s["leakage_detected"])
    return cells


def _rate(flags):
    return sum(flags) / len(flags) if flags else None


def _mean(xs):
    return statistics.fmean(xs) if xs else None


def _sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def summarize(cells: dict) -> list[dict]:
    rows = []
    for (arm, ds), c in cells.items():
        n = len(c["outcomes"])
        rows.append({
            "arm": arm, "dataset": ds,
            "n_measured": n, "missing": c["missing"], "contaminated": c["contaminated"],
            "rescored": c["rescored"],
            "success_rate": (c["outcomes"].count("SUCCESS") / n) if n else None,
            "outcomes": dict(Counter(c["outcomes"])),
            "fail_reasons": dict(c["fail_reasons"]),
            "f1_on_success": c["f1_on_success"],
            "f1_on_success_mean": _mean(c["f1_on_success"]),
            "f1_measured": c["f1_measured"],
            "f1_measured_mean": _mean(c["f1_measured"]),
            "f1_measured_sd": _sd(c["f1_measured"]),
            "f1_combined": c["combined"],
            "f1_combined_mean": _mean(c["combined"]),
            "cost_total": sum(c["cost"]),
            "calls_mean": _mean(c["calls"]),
            "leak_exploited": c["leak_exploited"],
            "leak_exploited_rate": _rate(c["leak_exploited"]),
            "leak_detected": c["leak_detected"],
            "leak_detected_rate": _rate(c["leak_detected"]),
            "leak_kappa": c["leak_kappa"],
            "run_ids": c["run_ids"],
        })
    rows.sort(key=lambda r: (r["dataset"],
                             ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 9))
    return rows


def print_table(rows: list[dict]) -> None:
    def f(x, w=6):
        return f"{x:{w}.3f}" if isinstance(x, float) else f"{'—':>{w}}"
    print(f"\n{'dataset':15s} {'arm':12s} {'n':>2s} {'miss':>4s} {'succ':>6s} "
          f"{'F1succ':>6s} {'F1meas':>6s} {'SD':>6s} {'comb':>6s} {'cost':>6s}  fail_reasons")
    for r in rows:
        # Leakage-Runs: wie oft folgten die Predictions dem Leak / hat DU ihn ausgeschlossen
        leak = "".join(f"  {label} {sum(flags)}/{len(flags)}"
                       for label, flags in (("leak", r.get("leak_exploited")),
                                            ("DU", r.get("leak_detected"))) if flags)
        print(f"{r['dataset']:15s} {r['arm']:12s} {r['n_measured']:2d} {r['missing']:4d} "
              f"{f(r['success_rate'])} {f(r['f1_on_success_mean'])} {f(r['f1_measured_mean'])} "
              f"{f(r['f1_measured_sd'])} {f(r['f1_combined_mean'])} {r['cost_total']:6.2f}  "
              f"{r['fail_reasons'] or ''}{leak}")
    total_cost = sum(r["cost_total"] for r in rows)
    done = sum(r["n_measured"] for r in rows)
    missing = sum(r["missing"] for r in rows)
    print(f"\ngemessen {done} · offen {missing} · Kosten bisher {total_cost:.2f} $")


def run_plan(plan: list[dict], runs_dir: str, model: str, manifests_dir: str, *,
             timeout: int, log_dir: str, wait_s: int, max_failures: int) -> int:
    """Faehrt alle noch nicht gemessenen Jobs in Planreihenfolge. Rueckgabe: Fehlschlaege."""
    failures, started = 0, time.time()
    for idx, job in enumerate(plan, 1):
        rid = run_id_for(job)
        if is_measured(summary_path(runs_dir, job)):
            continue
        print(f"[{idx:3d}/{len(plan)}] {rid} ...", end=" ", flush=True)
        t0 = time.time()
        ok, note = execute_with_rate_limit_wait(job, runs_dir, model, manifests_dir,
                                                timeout, log_dir, wait_s, run_id=rid)
        print(f"{note} ({time.time() - t0:.0f}s, gesamt {(time.time() - started) / 3600:.1f} h)",
              flush=True)
        if not ok:
            failures += 1
            if max_failures and failures >= max_failures:
                print(f"\n⚠️  {failures} Fehlschlaege — abgebrochen (Logs: {log_dir}). "
                      f"Neustart faehrt nur die offenen Runs.")
                break
    return failures


def main():
    ap = argparse.ArgumentParser(description="Haupt-Runs: 4 Arme x Suite x n (resumebar)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--replicates", type=int, default=None,
                    help="n pro Zelle (Default 6, mit --leakage 3). Kleiner = Zwischenstand r1..rN")
    ap.add_argument("--leakage", nargs="?", const="perfect", default=None,
                    choices=list(LEAKAGE_VARIANTS),
                    help="Leakage-Runs (docs/leakage-runs.md): --leakage = Batch 1 (perfekt, "
                         "leak-*, _leakage/); --leakage spurious = Batch 2 (spur-*, _spurious/)")
    ap.add_argument("--arms", nargs="+", default=None, choices=ARM_ORDER,
                    help="Default: die 4 Kern-Arme (mit --batch Pflicht)")
    ap.add_argument("--batch", choices=BATCHES, default=None,
                    help="Zusatz-Batch mit eigenen run_ids (<batch>-<set>-<arm>-rK) und eigenem "
                         "Ausgabeordner (_<batch> bzw. _<ordner>_<batch>); h1 = H1-Nachtrag")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Default: alle Manifeste in --manifests-dir")
    ap.add_argument("--timeout", type=int, default=3600, help="Sekunden pro Run")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="Abbruch nach N Fehlschlaegen in Folge (0 = nie)")
    ap.add_argument("--rate-limit-wait", type=int, default=RATE_LIMIT_WAIT_S)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if args.batch and args.leakage:
        ap.error("--batch und --leakage schliessen sich aus")
    if args.batch and not args.arms:
        # Ohne --arms waeren es die 4 Kern-Arme = 312 Runs; ausserdem schreibt jeder
        # Aufruf main_runs.json nur fuer SEINE Arme -> immer alle Arme des Batches angeben.
        ap.error("--batch verlangt --arms (immer alle Arme des Batches)")
    args.arms = args.arms or list(ARMS)
    extra = [a for a in args.arms if a in EXTRA_ARMS]
    if extra and not args.batch:
        # Ohne eigenes Praefix landeten A-Single-Runs als main-* in _main/ und die
        # Aggregation ueberschriebe die Kern-Tabelle der 4 Arme.
        ap.error(f"{', '.join(extra)} nur mit --batch (z. B. --batch h1)")

    if args.replicates is None:
        args.replicates = LEAKAGE_REPLICATES if args.leakage else DEFAULT_REPLICATES
    datasets = args.datasets or suite_datasets(args.manifests_dir)
    plan = build_plan(datasets, args.replicates, args.arms, leakage=args.leakage,
                      batch=args.batch)
    out_dir = os.path.join(args.runs_dir, out_subdir(args.manifests_dir, leakage=args.leakage,
                                                     batch=args.batch))

    if args.dry_run:
        by_arm = Counter(j["arm"] for j in plan)
        kind = (f"Leakage-Runs {args.leakage}" if args.leakage
                else f"Batch {args.batch}" if args.batch else "Runs")
        print(f"Plan: {len(plan)} Runs = {len(args.arms)} Arme x {len(datasets)} Sets "
              f"x n={args.replicates} ({kind}, Ausgabe {out_dir})")
        est = 0.0
        for arm in args.arms:
            c = by_arm[arm] * COST_PER_RUN[arm]
            est += c
            print(f"  {arm:12s} {by_arm[arm]:3d} Runs  (~{c:.0f} $)")
        done = sum(1 for j in plan if is_measured(summary_path(args.runs_dir, j)))
        print(f"  Schaetzung ~{est:.0f} $ · bereits gemessen: {done} · Sets: {' '.join(datasets)}")
        return

    if not args.aggregate_only:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "plan.json"), "w") as f:
            json.dump(plan, f, indent=2)
        run_plan(plan, args.runs_dir, args.model, args.manifests_dir,
                 timeout=args.timeout, log_dir=os.path.join(out_dir, "logs"),
                 wait_s=args.rate_limit_wait, max_failures=args.max_failures)

    rows = summarize(aggregate(plan, args.runs_dir))
    print_table(rows)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "main_runs.json")
    with open(result_path, "w") as f:
        json.dump({"model": args.model, "replicates": args.replicates,
                   "leakage": args.leakage, "batch": args.batch,
                   "arms": args.arms, "datasets": datasets,
                   "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "cells": rows}, f, indent=2)
    print(f"Ergebnis geschrieben: {result_path}")


if __name__ == "__main__":
    main()
