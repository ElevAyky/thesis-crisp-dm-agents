# scripts/rescore_baseline.py
"""Offline-Neubewertung abgeschlossener Runs der freien Arme (C/D) nach der
Erfolgsregel vom 23.09. (gepaarter Bootstrap gegen die Default-Baseline auf
demselben Holdout, scoring.compare_to_baseline).

Zulaessig NUR fuer crisp_react/naked_react: dort hing kein Retry an der Schwelle,
die abgegebenen Predictions sind also dieselben, die der Agent unter der neuen
Regel abgegeben haette -> die Neubewertung ist exakt. Arm A/B (eval->model-
Ruecksprung haengt an der Schwelle) muessen neu laufen.

Schreibt NICHTS ueber die Original-Felder: das Ergebnis landet als Block
summary["rescore"] (mit den Originalwerten), main_runs.py liest es bevorzugt.
Runs, die schon nativ mit success_rule=noninferior_paired liefen, bleiben unberuehrt.

    uv run python scripts/rescore_baseline.py --runs-dir ../thesis-crisp-dm-runs            # dry-run
    uv run python scripts/rescore_baseline.py --runs-dir ../thesis-crisp-dm-runs --write
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crispfsm.manifests import load_manifest  # noqa: E402
from crispfsm.scoring import baseline_f1_on_holdout, compare_to_baseline  # noqa: E402

RESCORABLE_ARMS = ("crisp_react", "naked_react")
RULE = "noninferior_paired"


def parse_run_id(run_id: str) -> tuple[str, str]:
    """main-<dataset>-<arm>-r<k> -> (dataset, arm). Datasets tragen '_', nie '-'."""
    _, dataset, arm, _ = run_id.split("-")
    return dataset, arm


def rescore_run(run_dir: str, manifests_dir: str) -> dict | None:
    """Neubewertung eines Runs -> rescore-Block, oder None (nicht anwendbar)."""
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    dataset, arm = parse_run_id(os.path.basename(run_dir))
    if arm not in RESCORABLE_ARMS:
        raise ValueError(f"{run_dir}: Arm {arm} hat einen schwellen-abhaengigen Retry — "
                         f"nicht offline neu bewertbar, neu fahren.")
    if s.get("outcome") == "ERROR" or s.get("success_rule") == RULE:
        return None
    original = {k: s.get(k) for k in ("outcome", "success", "fail_reason", "final_f1")}
    # Nur SUCCESS und below_threshold haengen an der Schwelle; ABSTAIN, form_error,
    # target-Fehler usw. bleiben, wie sie sind.
    if not (s.get("outcome") == "SUCCESS" or s.get("fail_reason") == "below_threshold"):
        return {**original, "rule": RULE, "original": original, "baseline_comparison": None,
                "changed": False}
    m = load_manifest(dataset, manifests_dir=manifests_dir)
    base = os.path.join(manifests_dir, "baseline_preds", f"{dataset}.parquet")
    test = os.path.join(run_dir, "holdout", "test.parquet")
    target = pd.read_parquet(test).columns[-1]          # prepare_dataset: y ist die letzte Spalte
    aligned = baseline_f1_on_holdout(base, test, target=target)
    if abs(aligned - m.baseline_f1) > 6e-4:
        raise ValueError(f"{run_dir}: Baseline-Predictions passen nicht zum Holdout "
                         f"({aligned:.4f} vs. {m.baseline_f1})")
    cmp = compare_to_baseline(os.path.join(run_dir, "work", "predictions.parquet"),
                              test, base, target=target)
    success = bool(cmp and cmp.noninferior)
    new = {"outcome": "SUCCESS" if success else "FAIL", "success": success,
           "fail_reason": None if success else "below_threshold",
           "final_f1": cmp.f1 if success else None}
    return {**new, "rule": RULE, "original": original,
            "baseline_comparison": asdict(cmp) if cmp else None,
            "changed": new["success"] != original["success"],
            "rescored_at": time.strftime("%Y-%m-%d")}


def main():
    ap = argparse.ArgumentParser(description="C/D-Runs nach der Bootstrap-Regel neu bewerten")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--pattern", default="main-*-*-r*")
    ap.add_argument("--write", action="store_true", help="rescore-Block in summary.json schreiben")
    args = ap.parse_args()

    changed = total = 0
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, args.pattern))):
        if not os.path.exists(os.path.join(run_dir, "summary.json")):
            continue
        if parse_run_id(os.path.basename(run_dir))[1] not in RESCORABLE_ARMS:
            continue
        block = rescore_run(run_dir, args.manifests_dir)
        if block is None:
            continue
        total += 1
        if block["changed"]:
            changed += 1
            c = block["baseline_comparison"]
            print(f"{os.path.basename(run_dir):40s} {block['original']['outcome']:7s} -> "
                  f"{block['outcome']:7s}  diff {c['diff']:+.4f}  KI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]")
        if args.write:
            path = os.path.join(run_dir, "summary.json")
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            s["rescore"] = block
            with open(path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
    print(f"\n{total} Runs bewertet, {changed} geaendert"
          f"{' — geschrieben' if args.write else ' (dry-run, --write zum Speichern)'}")


if __name__ == "__main__":
    main()
