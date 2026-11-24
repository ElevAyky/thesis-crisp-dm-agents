# scripts/variance_scan.py
"""Varianz-Scan über die kuratierte Kandidaten-Suite (roadmap §2,
experiment-design.md §5.2): misst die Run-SD des finalen F1 pro Zelle und
leitet daraus das finale **n pro Zelle** ab.

Warum: das vorläufige n=6 stammt aus dem Kalibrierungslauf über nur zwei
(prozess-leichte) Sets. Die SE-Heuristik hängt aber an der *lautesten* Zelle
der ganzen Suite — streut irgendein Set stärker als credit_g x Naked ReAct
(SD 0.0395), ist n=6 zu klein.

Scan-Plan (roadmap §2, beschlossen 08.08.):
    Arm D (naked_react) x ALLE Manifeste       x n=3   — billig + der Streuer
    Arm A (hard_fsm)    x strukturell abweichend x n=3 — teuer + fast rauschfrei,
                                                         nur Stichprobe

Die Scan-Daten sind **Kalibrierung, nicht konfirmatorisch** — sie fliessen
nicht in die Haupt-Auswertung ein (experiment-design.md §5.2).

Aufruf (aus dem Repo-Root):
    python scripts/variance_scan.py --dry-run        # Plan + Kostenschaetzung
    python scripts/variance_scan.py                  # kompletter Scan (resumebar)
    python scripts/variance_scan.py --aggregate-only # nur Tabelle neu bauen

Der Scan ist unterbrechbar: jeder Run hat eine deterministische run_id, ein
bereits geschriebenes summary.json wird uebersprungen. Einfach neu starten.
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import time

# Arm A laeuft nur auf den strukturell abweichenden Sets (roadmap §2):
# high-dim (cnae_9, fashion_mnist) + imbalanced (bank_marketing). Post-cutoff
# entfaellt vorerst — OpenML gibt nichts Ladbares her (benchmarks.md §2).
ARM_A_DATASETS = ("cnae_9", "fashion_mnist", "bank_marketing")

# Gemessene Kosten pro Run (experiment-design.md §5.2, Kalibrierung 08.08.).
COST_PER_RUN = {"naked_react": 0.35, "hard_fsm": 1.72,
                "crisp_react": 0.35, "soft_fsm": 2.10}

# SE-Ziel der n-Heuristik (experiment-design.md §5.2): SE <= 1/3 des kleinsten
# aufzuloesenden Arm-Effekts.
MIN_ARM_EFFECT = 0.05
SE_FRACTION = 3.0


def target_se() -> float:
    return MIN_ARM_EFFECT / SE_FRACTION


def required_n(sd: float) -> int:
    """n, damit SD/sqrt(n) <= target_se — die Rechnung aus §5.2."""
    if sd <= 0:
        return 1
    return max(1, math.ceil((sd / target_se()) ** 2))


def build_plan(datasets: list[str], replicates: int,
               arm_a_datasets: list[str]) -> list[dict]:
    """(arm, dataset, replicate)-Tripel in Laufreihenfolge: erst der billige
    Breiten-Arm, dann die teure Stichprobe."""
    plan = []
    for ds in datasets:
        for i in range(1, replicates + 1):
            plan.append({"arm": "naked_react", "dataset": ds, "replicate": i})
    for ds in arm_a_datasets:
        for i in range(1, replicates + 1):
            plan.append({"arm": "hard_fsm", "dataset": ds, "replicate": i})
    return plan


def run_id_for(job: dict) -> str:
    return f"scan-{job['arm']}-{job['dataset']}-r{job['replicate']}"


def summary_path(runs_dir: str, job: dict) -> str:
    return os.path.join(runs_dir, run_id_for(job), "summary.json")


def is_measured(path: str) -> bool:
    """Ein summary.json zaehlt als gemessen, wenn es existiert und KEIN ERROR ist.
    ERROR (Crash/Nicht-Terminierung des Harness, run.py) ist kein Agent-Befund:
    beim Resume neu fahren, in der Auswertung nicht in die SUCCESS-Rate mischen."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return json.load(f).get("outcome") != "ERROR"
    except (OSError, ValueError):
        return False


# Exit-Code von run.py bei Abo-Kontingent-Ende (EX_TEMPFAIL) — muss mit
# crispfsm.run.EXIT_RATE_LIMITED uebereinstimmen (hier dupliziert, damit das Script
# ohne Paket-Import laeuft).
EXIT_RATE_LIMITED = 75
RATE_LIMIT_WAIT_S = 600   # Poll-Intervall, bis wieder Kontingent da ist


def execute(job: dict, runs_dir: str, model: str, manifests_dir: str,
            timeout: int, log_dir: str, run_id: str | None = None) -> tuple[bool, str, int]:
    """Einen Run als Subprocess starten. Rueckgabe: (ok, notiz, exit_code).

    Subprocess statt Import: ein Crash oder Hänger reisst den Scan nicht mit,
    und jeder Run bekommt einen frischen Interpreter-Zustand.
    run_id: Default = Scan-Schema; andere Runner (main_runs) geben ihre eigene mit.
    """
    rid = run_id or run_id_for(job)
    cmd = [sys.executable, "-m", "crispfsm.run",
           "--dataset", job["dataset"], "--arm", job["arm"],
           "--model", model, "--runs-dir", runs_dir,
           "--manifests-dir", manifests_dir, "--run-id", rid]
    if job.get("leakage"):   # Leakage-Runs (main_runs --leakage, docs/leakage-runs.md)
        cmd.append("--leakage")
        if job.get("leakage_variant"):
            cmd += ["--leakage-variant", job["leakage_variant"]]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{rid}.log")
    with open(log_file, "w") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT nach {timeout}s", -1
    if proc.returncode != 0:
        return False, f"exit {proc.returncode} (Log: {log_file})", proc.returncode
    if not os.path.exists(os.path.join(runs_dir, rid, "summary.json")):
        return False, "kein summary.json trotz exit 0", 0
    return True, "ok", 0


def execute_with_rate_limit_wait(job: dict, runs_dir: str, model: str,
                                 manifests_dir: str, timeout: int, log_dir: str,
                                 wait_s: int = RATE_LIMIT_WAIT_S,
                                 run_id: str | None = None) -> tuple[bool, str]:
    """Wie execute(), aber Abo-Kontingent-Ende (Exit 75, ERROR/rate_limited) zaehlt
    nicht als Fehlschlag: warten, dann EXAKT denselben Run neu fahren — so lange,
    bis er ohne Rate-Limit durchlaeuft (Johannes 18.09.: Run zaehlt nicht,
    wiederholen sobald wieder Kontingent da ist). Das ERROR-summary.json des
    Fehlversuchs wird dabei ueberschrieben."""
    waits = 0
    while True:
        ok, note, rc = execute(job, runs_dir, model, manifests_dir, timeout, log_dir,
                               run_id=run_id)
        if rc != EXIT_RATE_LIMITED:
            if waits:
                note = f"{note} (nach {waits}x rate_limited-Wartezeit)"
            return ok, note
        waits += 1
        print(f"rate_limited — warte {wait_s // 60} min ({waits}. Mal) ...",
              end=" ", flush=True)
        time.sleep(wait_s)


def collect(plan: list[dict], runs_dir: str) -> dict[tuple[str, str], dict]:
    """summary.json je Job einsammeln und nach Zelle (arm, dataset) buendeln."""
    cells: dict[tuple[str, str], dict] = {}
    for job in plan:
        key = (job["arm"], job["dataset"])
        cell = cells.setdefault(key, {"arm": job["arm"], "dataset": job["dataset"],
                                      "f1": [], "outcomes": [], "cost": [],
                                      "missing": 0})
        path = summary_path(runs_dir, job)
        if not is_measured(path):
            cell["missing"] += 1        # fehlt oder ERROR -> nicht gemessen
            continue
        with open(path) as f:
            s = json.load(f)
        cell["outcomes"].append(s.get("outcome", "?"))
        cost = s.get("total_cost_usd")
        if isinstance(cost, (int, float)) and not math.isnan(cost):
            cell["cost"].append(cost)
        # holdout_intact false = kontaminiert, zaehlt nicht in die Varianz
        f1 = s.get("final_f1")
        if f1 is not None and s.get("holdout_intact", True):
            cell["f1"].append(f1)
    return cells


def summarize(cells: dict) -> list[dict]:
    rows = []
    for (arm, ds), c in cells.items():
        f1s = c["f1"]
        sd = statistics.stdev(f1s) if len(f1s) > 1 else (0.0 if f1s else None)
        rows.append({
            "arm": arm, "dataset": ds,
            "n_runs": len(c["outcomes"]), "n_f1": len(f1s), "missing": c["missing"],
            "f1_values": f1s,
            "f1_mean": statistics.fmean(f1s) if f1s else None,
            "f1_sd": sd,
            "n_required": required_n(sd) if sd is not None else None,
            "outcomes": c["outcomes"],
            "cost_total": sum(c["cost"]),
        })
    rows.sort(key=lambda r: (r["arm"], r["dataset"]))
    return rows


def print_table(rows: list[dict]) -> None:
    print("\n| Zelle | Runs | F1 (Einzelwerte) | Mittel | **Run-SD** | n nötig | $ |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        arm = {"hard_fsm": "Hard-FSM (A)", "naked_react": "Naked ReAct (D)",
               "soft_fsm": "Soft-FSM (B)", "crisp_react": "CRISP-ReAct (C)"}.get(
                   r["arm"], r["arm"])
        vals = " · ".join(f"{v:.4f}" for v in r["f1_values"]) or "—"
        mean = f"{r['f1_mean']:.4f}" if r["f1_mean"] is not None else "—"
        sd = f"**{r['f1_sd']:.4f}**" if r["f1_sd"] is not None else "—"
        need = r["n_required"] if r["n_required"] is not None else "—"
        runs = f"{r['n_f1']}/{r['n_runs']}"
        if r["missing"]:
            runs += f" (+{r['missing']} offen)"
        print(f"| {r['dataset']} × {arm} | {runs} | {vals} | {mean} | {sd} "
              f"| {need} | {r['cost_total']:.2f} |")


def print_verdict(rows: list[dict]) -> None:
    scored = [r for r in rows if r["f1_sd"] is not None and r["n_f1"] > 1]
    if not scored:
        print("\n⚠️  Keine Zelle mit >1 gemessenem F1 — kein n ableitbar.")
        return
    loudest = max(scored, key=lambda r: r["f1_sd"])
    n_final = max(r["n_required"] for r in scored)
    total = sum(r["cost_total"] for r in rows)
    incomplete = [r for r in rows if r["missing"] or r["n_f1"] < 2]

    print(f"\n=== ERGEBNIS ===")
    print(f"Lauteste Zelle : {loudest['dataset']} × {loudest['arm']} "
          f"(SD {loudest['f1_sd']:.4f}, aus {loudest['n_f1']} Runs)")
    print(f"SE-Ziel        : {target_se():.4f} "
          f"(= {MIN_ARM_EFFECT} / {SE_FRACTION:.0f}, experiment-design §5.2)")
    print(f"→ n final      : {n_final}   (SE bei n={n_final}: "
          f"{loudest['f1_sd'] / math.sqrt(n_final):.4f})")
    print(f"Scan-Kosten    : {total:.2f} $")
    prev = 6
    if n_final > prev:
        print(f"\n⚠️  n={n_final} > vorläufiges n={prev} — Haupt-Run-Budget steigt "
              f"entsprechend (experiment-design §5.2 aktualisieren).")
    elif n_final < prev:
        print(f"\nn={n_final} < vorläufiges n={prev} — Suite streut schwächer "
              f"als credit_g × D.")
    else:
        print(f"\n✅ n={prev} bestätigt.")
    if incomplete:
        print(f"\n⚠️  {len(incomplete)} Zelle(n) unvollständig (<2 verwertbare F1) — "
              f"Scan erneut starten, er setzt fort:")
        for r in incomplete:
            print(f"    {r['dataset']} × {r['arm']}: {r['n_f1']} F1, "
                  f"outcomes {r['outcomes'] or '—'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Arm-D-Sets (Default: alle Manifeste in --manifests-dir)")
    ap.add_argument("--arm-a-datasets", nargs="*", default=list(ARM_A_DATASETS),
                    help=f"Arm-A-Stichprobe (Default: {' '.join(ARM_A_DATASETS)})")
    ap.add_argument("--replicates", type=int, default=3, help="n pro Zelle im Scan")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--rate-limit-wait", type=int, default=RATE_LIMIT_WAIT_S,
                    help="Sekunden Wartezeit, wenn das Abo-Kontingent erschoepft ist "
                         "(run.py Exit 75); der Run wird danach wiederholt")
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Sekunden pro Run, dann Abbruch dieses Runs")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="Scan abbrechen nach so vielen Fehlschlaegen (0 = nie)")
    ap.add_argument("--dry-run", action="store_true",
                    help="nur Plan + Kostenschaetzung, nichts starten")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="nichts starten, nur vorhandene summary.json auswerten")
    args = ap.parse_args()

    datasets = args.datasets or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(args.manifests_dir, "*.yaml")))
    plan = build_plan(datasets, args.replicates, args.arm_a_datasets)
    out_dir = os.path.join(args.runs_dir, "_scan")

    if args.dry_run:
        est = sum(COST_PER_RUN.get(j["arm"], 0.5) for j in plan)
        by_arm: dict[str, int] = {}
        for j in plan:
            by_arm[j["arm"]] = by_arm.get(j["arm"], 0) + 1
        print(f"Scan-Plan: {len(plan)} Runs, geschätzt {est:.2f} $")
        for arm, cnt in sorted(by_arm.items()):
            print(f"  {arm:12s} {cnt:3d} Runs  "
                  f"(~{cnt * COST_PER_RUN.get(arm, 0.5):.2f} $)")
        done = sum(1 for j in plan if is_measured(summary_path(args.runs_dir, j)))
        print(f"  davon bereits gelaufen: {done}")
        print(f"\nArm D × {len(datasets)} Sets: {' '.join(datasets)}")
        print(f"Arm A × {len(args.arm_a_datasets)} Sets: "
              f"{' '.join(args.arm_a_datasets)}")
        return

    if not args.aggregate_only:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "plan.json"), "w") as f:
            json.dump(plan, f, indent=2)
        failures, started = 0, time.time()
        for idx, job in enumerate(plan, 1):
            rid = run_id_for(job)
            if is_measured(summary_path(args.runs_dir, job)):
                print(f"[{idx:3d}/{len(plan)}] {rid} — skip (liegt vor)", flush=True)
                continue
            print(f"[{idx:3d}/{len(plan)}] {rid} ...", end=" ", flush=True)
            t0 = time.time()
            ok, note = execute_with_rate_limit_wait(
                job, args.runs_dir, args.model, args.manifests_dir,
                args.timeout, os.path.join(out_dir, "logs"), wait_s=args.rate_limit_wait)
            print(f"{note} ({time.time() - t0:.0f}s)", flush=True)
            if not ok:
                failures += 1
                if args.max_failures and failures >= args.max_failures:
                    print(f"\n⚠️  {failures} Fehlschläge — Scan abgebrochen. "
                          f"Logs in {out_dir}/logs/")
                    break
        print(f"\nLaufzeit gesamt: {(time.time() - started) / 60:.1f} min")

    rows = summarize(collect(plan, args.runs_dir))
    print_table(rows)
    print_verdict(rows)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "variance_scan.json")
    with open(result_path, "w") as f:
        json.dump({"target_se": target_se(), "min_arm_effect": MIN_ARM_EFFECT,
                   "replicates": args.replicates, "model": args.model,
                   "cells": rows}, f, indent=2)
    print(f"\nErgebnis geschrieben: {result_path}")


if __name__ == "__main__":
    main()
