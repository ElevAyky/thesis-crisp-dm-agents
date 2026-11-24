#!/usr/bin/env python
"""eucalyptus: welche Feature-Spalten nutzt das abgegebene Modell nicht? (C/D vollstaendig, A/B Kontrolle)

Schliesst Pruefbefund K6-04 (Kap. 6.6): A und B schliessen auf eucalyptus laut DU-Report
ID-artige Spalten aus (F6-633/634). Ob die freien Arme C und D dieselben Spalten nutzen,
war offen. Vorbild ist die V12-Sichtung auf bank_marketing (docs/roadmap.md:107-115).

Massgeblich ist die Pipeline, die die abgegebenen Vorhersagen erzeugt hat, nicht
verworfene Zwischenversuche. Pro Run drei Pruefungen:

1. Code-Beleg (von Hand gelesen, hier festgeschrieben): ausgeschlossene Spalten mit Datei:Zeile.
   Das Script prueft, dass die belegte Zeile die genannten Spalten bzw. das Muster enthaelt.
2. Transkript (nur C/D): die letzte Aenderung an `model.py` (Write, Edit oder `sed -i`) liegt vor
   dem letzten erfolgreichen `python model.py`, und danach schreibt kein anderer Befehl
   `predictions.parquet`.
3. Einfluss-Test (automatisch): Kopie von `work/` in ein Temp-Verzeichnis, `test_features.parquet`
   wird durch einen Stapel ersetzt: die Originalzeilen plus je Spalte zehn Bloecke, in denen nur
   diese Spalte zeilenweise permutiert ist. Dann laeuft das abgegebene Script (C/D: `model.py`,
   trainiert deterministisch neu; A/B: das Eval-Script mit gespeichertem `model.pkl` +
   Preprocessor). (a) Der Originalblock muss die abgegebenen Vorhersagen exakt reproduzieren.
   (b) Eine Spalte gilt als nicht genutzt, wenn keine der zehn Permutationen eine Vorhersage aendert.
   Der Test bestaetigt den Code-Beleg, ersetzt ihn nicht: eine genutzte, aber fast wirkungslose
   Spalte (die Ortsspalten sind untereinander redundant) koennte ihn knapp verfehlen; die
   schwaechste genutzte Spalte je Run steht deshalb mit in der Ausgabe.
   Alle Vorverarbeitungen der 14 Runs sind zeilenweise bzw. auf Train gefittet, der Stapel
   veraendert also die Behandlung der Originalzeilen nicht (durch (a) geprueft).

Python = dieselbe .venv wie bei den Runs (Traceback-Pfade in den Transkripten; uv.lock seit 23.09.
unveraendert). Ins Runs-Repo wird nichts geschrieben.

Ausgabe: thesis/facts/derived/eucalyptus_exclusions.json

Aufruf aus dem Repo-Root:
    uv run python thesis/scripts/eucalyptus_exclusions.py [--runs-dir ../thesis-crisp-dm-runs]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_actions import load_run  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RUNS = os.path.join(os.path.dirname(REPO), "thesis-crisp-dm-runs")
OUT = os.path.join(REPO, "thesis", "facts", "derived", "eucalyptus_exclusions.json")

ID_COLS = ["Abbrev", "Map_Ref", "PMCno"]      # in allen A-Runs ausgeschlossen (F6-633)
SEEDS = tuple(range(1, 11))   # 10 Permutationen je Spalte (3 reichten fuer schwache Ortsspalten nicht sicher)
L = {"hard_fsm": "A", "soft_fsm": "B", "crisp_react": "C", "naked_react": "D"}

# --------------------------------------------------------------------------- Code-Belege
# excluded = Spalten, die das abgegebene Modell laut Code nicht sieht (von Hand gelesen).
# evidence = (Datei relativ zum Run-Ordner, Zeile, Teilstrings, die in der Zeile stehen muessen)
# transcript_must = Teilstring, der in der Ausgabe des letzten model.py-Laufs stehen muss
STATIC = {
    "main-eucalyptus-crisp_react-r1": dict(
        excluded=[], evidence=[("work/model.py", 12, ["train.drop(columns=[TARGET])"])],
        note="alle 19 Spalten; kategoriale Spalten ordinal kodiert, numerische Median-imputiert; RF"),
    "main-eucalyptus-crisp_react-r2": dict(
        excluded=[], evidence=[("work/model.py", 15, ["train.drop(columns=[TARGET])"]),
                               ("work/model.py", 19, ["DBH", "clip"])],
        note="alle 19 Spalten (DBH nur gekappt, nicht entfernt); HGB"),
    "main-eucalyptus-crisp_react-r3": dict(
        excluded=["Abbrev", "Map_Ref", "PMCno"],
        evidence=[("work/model.py", 20, ["Map_Ref", "PMCno", "Abbrev"]), ("work/model.py", 21, ["drop(columns=drop_cols)"])],
        note="zuerst nur Map_Ref, nach CV-Vergleich im Transkript (Aktion 6) per Edit auf drei Spalten erweitert, "
             "dann neu ausgefuehrt; HGB"),
    "main-eucalyptus-crisp_react-r4": dict(
        excluded=["Map_Ref"],
        evidence=[("work/model.py", 14, ["DROP", "Map_Ref"]), ("work/model.py", 29, ["CAT + NUM"])],
        note="Spaltenauswahl ueber die Listen CAT/NUM (Z. 15-17), Map_Ref fehlt darin; HGB"),
    "main-eucalyptus-crisp_react-r5": dict(
        excluded=["Rep"],
        evidence=[("work/model.py", 22, ["Rep"]), ("work/model.py", 23, ["drop(columns=drop_cols)"])],
        note="nur Rep (Wiederholungsindex) entfernt; HGB"),
    "main-eucalyptus-crisp_react-r6": dict(
        excluded=["Map_Ref", "PMCno"],
        evidence=[("work/model.py", 34, ["PMCno", "Map_Ref"]), ("work/model.py", 39, ["num_cols + cat_cols"])],
        note="Auswahl ueber num_cols/cat_cols (Z. 35-37), drop_cols selbst wird nicht benutzt; RF vs. HGB per CV, HGB gewaehlt"),
    "main-eucalyptus-naked_react-r1": dict(
        excluded=["PMCno"],
        evidence=[("work/model.py", 10, ["DROP", "PMCno"]), ("work/model.py", 16, ["[TARGET] + DROP"])],
        note="RF"),
    "main-eucalyptus-naked_react-r2": dict(
        excluded=[], evidence=[("work/model.py", 14, ["tr.drop(columns=[TARGET])"]),
                               ("work/model.py", 20, ["nunique(dropna=False) <= 1"])],
        transcript_must="dropped constants: []",
        note="entfernt nur konstante Spalten, es gibt keine (Transkript: 'dropped constants: []'); "
             "Kommentar Z. 16 nennt Abbrev ID-artig, behaelt es; RF"),
    "main-eucalyptus-naked_react-r3": dict(
        excluded=[], evidence=[("work/model.py", 13, ["drop = [TARGET]"]),
                               ("work/model.py", 12, ["Map_Ref", "Latitude", "keep"])],
        note="Kommentar: Map_Ref/Latitude quasi-Identifikatoren, aber geografisch informativ -> behalten; RF"),
    "main-eucalyptus-naked_react-r4": dict(
        excluded=[], evidence=[("work/model.py", 17, ["tr.drop(columns=[TARGET])"]),
                               ("work/model.py", 19, ["legitimate"])],
        note="Kommentar: alle Nicht-Ziel-Spalten legitim; HGB"),
    "main-eucalyptus-naked_react-r5": dict(
        excluded=["Abbrev", "Latitude", "Map_Ref", "PMCno"],
        evidence=[("work/model.py", 20, ["Abbrev", "Map_Ref", "Latitude", "PMCno"]),
                  ("work/model.py", 21, ["drop_cols + [TARGET]"])],
        note="nach dem einzigen model.py-Lauf CV-Varianten per python -c (Aktion 6: drop [] 0.639, nur PMCno 0.641, "
             "drei Spalten 0.637), model.py aber nicht mehr geaendert -> abgegeben mit vier ausgeschlossenen Spalten; RF"),
    "main-eucalyptus-naked_react-r6": dict(
        excluded=["Abbrev", "Map_Ref"],
        evidence=[("work/model.py", 15, ["Map_Ref", "Abbrev"]), ("work/model.py", 16, ["not in drop_cols"])],
        note="RF"),
    # Kontrolle A/B (F6-633/634): gleiche Methode, Eval-Script mit gespeicherten Artefakten
    "main-eucalyptus-hard_fsm-r3": dict(
        excluded=["Abbrev", "Latitude", "Locality", "Map_Ref", "PMCno", "Rep"],
        evidence=[("work/prep/preprocess.py", 21, ["PMCno", "Abbrev", "Map_Ref", "Latitude", "Locality", "Rep"]),
                  ("work/prep/preprocess.py", 32, ["NUMERIC_SITE + NUMERIC_GROWTH"])],
        script="eval/predict.py",
        note="Features = NUMERIC_SITE + NUMERIC_GROWTH + Missing-Flags + Sp one-hot (Z. 24-30, 55-68); Kontrolle"),
    "main-eucalyptus-soft_fsm-r1": dict(
        excluded=["Abbrev", "Map_Ref"],
        evidence=[("work/prep/preprocess.py", 24, ["Abbrev", "Map_Ref"]), ("work/prep/preprocess.py", 92, ["EXCLUDE"])],
        script="eval/predict.py",
        note="PMCno bleibt (numerisch + Missing-Indikator, Z. 30); Kontrolle"),
}
CD_RUNS = [f"main-eucalyptus-{a}-r{k}" for a in ("crisp_react", "naked_react") for k in range(1, 7)]
CONTROL_RUNS = ["main-eucalyptus-hard_fsm-r3", "main-eucalyptus-soft_fsm-r1"]

RUN_RE = re.compile(r"python\s+model\.py")
SED_RE = re.compile(r"sed\s+-i[^\n]*?model\.py")


def check_evidence(run_dir: str, spec: dict) -> list:
    out = []
    for rel, line, needles in spec["evidence"]:
        with open(os.path.join(run_dir, rel), encoding="utf-8") as f:
            lines = f.read().split("\n")
        text = lines[line - 1]
        missing = [s for s in needles if s not in text]
        if missing:
            raise AssertionError(f"{run_dir} {rel}:{line} enthaelt {missing} nicht: {text!r}")
        out.append({"file": rel, "line": line, "text": text.strip()})
    return out


def transcript_check(runs_dir: str, run_id: str, must: str | None) -> dict:
    """Letzte model.py-Aenderung vor letztem erfolgreichem Lauf; danach kein anderer Schreiber."""
    ra = load_run(runs_dir, run_id)
    last_edit, last_run, last_run_out, writers = None, None, "", []
    for e in ra.actions:
        pos = float(e.n)
        if e.tool in ("Write", "Edit") and os.path.basename(e.path.replace("\\", "/")) == "model.py":
            last_edit = pos
        elif e.tool == "Bash":
            cmd = e.inp.get("command", "")
            sed, run = SED_RE.search(cmd), RUN_RE.search(cmd)
            if sed:
                last_edit = pos
            if run and not e.is_error and not e.result.startswith("Exit code"):
                # sed vor dem Lauf im selben Befehl -> Lauf zaehlt als danach
                last_run, last_run_out = (pos + 0.5 if sed and sed.start() < run.start() else pos), e.result
            elif not run and "to_parquet" in cmd and "predictions" in cmd:
                writers.append(e.n)
    writers_after = [n for n in writers if last_run is not None and n > last_run]
    ok = last_run is not None and last_edit is not None and last_run > last_edit and not writers_after
    if must is not None:
        ok = ok and must in last_run_out
    return {"last_model_py_edit_action": last_edit, "last_successful_model_py_run_action": last_run,
            "other_prediction_writers_after": writers_after, "n_actions": len(ra.actions), "ok": bool(ok)}


def influence_test(run_dir: str, script: str, pred_rel: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="euca_excl_") as tmp:
        work = os.path.join(tmp, "work")
        shutil.copytree(os.path.join(run_dir, "work"), work,
                        ignore=shutil.ignore_patterns("predictions.parquet", "__pycache__"))
        tf = os.path.join(work, "data", "test_features.parquet")
        test = pd.read_parquet(tf).reset_index(drop=True)
        n = len(test)
        blocks, keys = [test], [(None, None)]
        for c in test.columns:
            for s in SEEDS:
                b = test.copy()
                b[c] = test[c].iloc[np.random.default_rng(s).permutation(n)].reset_index(drop=True)
                blocks.append(b)
                keys.append((c, s))
        stacked = pd.concat(blocks, ignore_index=True)
        assert (stacked.dtypes == test.dtypes).all(), "Stapel aendert dtypes"
        stacked.to_parquet(tf, index=False)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, script], cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, timeout=3600)
        if p.returncode != 0:
            raise RuntimeError(f"{run_dir}: {script} exit {p.returncode}\n{p.stderr[-2000:]}")
        pred = pd.read_parquet(os.path.join(work, pred_rel))["prediction"].astype(str).to_numpy()
    assert len(pred) == n * len(keys), f"{run_dir}: {len(pred)} Vorhersagen statt {n * len(keys)}"
    orig = pred[:n]
    delivered = pd.read_parquet(os.path.join(run_dir, "work", pred_rel))["prediction"].astype(str).to_numpy()
    changed = {}
    for i, (c, _s) in enumerate(keys[1:], start=1):
        changed.setdefault(c, []).append(int((pred[i * n:(i + 1) * n] != orig).sum()))
    used = {c: v for c, v in changed.items() if max(v) > 0}
    weakest = min(used, key=lambda c: (sum(x > 0 for x in used[c]), max(used[c])))
    return {
        "test_rows": n, "features": list(test.columns),
        "reproduced_rows": int((orig == delivered).sum()), "reproduced": bool((orig == delivered).all()),
        "changed_predictions_per_permutation": changed,
        "excluded_dynamic": sorted(c for c in test.columns if max(changed[c]) == 0),
        # Empfindlichkeit: genutzte Spalte mit den wenigsten wirksamen Permutationen
        "weakest_used_column": {"column": weakest, "permutations_with_change": sum(x > 0 for x in used[weakest]),
                                "max_changed_rows": max(used[weakest])},
    }


def mean(xs):
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    runs_dir = ap.parse_args().runs_dir

    runs = {}
    for rid in CD_RUNS + CONTROL_RUNS:
        rdir = os.path.join(runs_dir, rid)
        spec = STATIC[rid]
        with open(os.path.join(rdir, "summary.json"), encoding="utf-8") as f:
            summ = json.load(f)
        arm = summ["arm"]
        is_cd = arm in ("crisp_react", "naked_react")
        ev = check_evidence(rdir, spec)
        script = spec.get("script", "model.py")
        pred_rel = "predictions.parquet" if is_cd else os.path.join("eval", "predictions.parquet")
        dyn = influence_test(rdir, script, pred_rel)
        static = sorted(spec["excluded"])
        rec = {
            "arm": L[arm], "rep": int(rid.rsplit("-r", 1)[1]), "outcome": summ["outcome"],
            "best_f1_measured": summ["best_f1_measured"],
            "excluded_columns": static, "excluded_id_columns": [c for c in static if c in ID_COLS],
            "all_features_used": not static,
            "static_equals_dynamic": static == dyn["excluded_dynamic"],
            "evidence": ev, "note": spec["note"], "influence_test": dyn,
        }
        if is_cd:
            rec["transcript"] = transcript_check(runs_dir, rid, spec.get("transcript_must"))
        else:
            du = [c.get("column") for c in ((summ.get("du_report") or {}).get("excluded_columns") or [])]
            rec["du_report_excluded"] = sorted(du)
            rec["du_report_equals_dynamic"] = sorted(du) == dyn["excluded_dynamic"]
        runs[rid] = rec
        flag = "" if rec["static_equals_dynamic"] and dyn["reproduced"] else "  <-- PRUEFEN"
        print(f"{rid}: excl {static} | dyn {dyn['excluded_dynamic']} | repro {dyn['reproduced_rows']}/{dyn['test_rows']} "
              f"| F1 {summ['best_f1_measured']:.3f} {summ['outcome']}"
              + (f" | transcript ok {rec['transcript']['ok']}" if is_cd else f" | du_report {rec['du_report_excluded']}")
              + flag)

    by_arm = {}
    for a in ("C", "D"):
        rs = [r for rid, r in runs.items() if r["arm"] == a]
        cols = {}
        for r in rs:
            for c in r["excluded_columns"]:
                cols[c] = cols.get(c, 0) + 1
        excl = [r["best_f1_measured"] for r in rs if r["excluded_id_columns"]]
        keep = [r["best_f1_measured"] for r in rs if not r["excluded_id_columns"]]
        by_arm[a] = {
            "runs": len(rs),
            "excluded_column_counts": dict(sorted(cols.items(), key=lambda kv: (-kv[1], kv[0]))),
            "runs_excluding_any_id_column": sorted(r["rep"] for r in rs if r["excluded_id_columns"]),
            "runs_all_features": sorted(r["rep"] for r in rs if r["all_features_used"]),
            "f1_mean_excluding_id": mean(excl), "f1_range_excluding_id": [min(excl), max(excl)] if excl else None,
            "f1_mean_keeping_id": mean(keep), "f1_range_keeping_id": [min(keep), max(keep)] if keep else None,
            "f1_mean_all": mean([r["best_f1_measured"] for r in rs]),
            "outcomes": sorted({r["outcome"] for r in rs}),
        }
    cd = [runs[r] for r in CD_RUNS]
    checks = {
        "cd_runs": len(cd),
        "cd_reproduced": sum(r["influence_test"]["reproduced"] for r in cd),
        "cd_static_equals_dynamic": sum(r["static_equals_dynamic"] for r in cd),
        "cd_transcript_ok": sum(r["transcript"]["ok"] for r in cd),
        "min_permutations_with_change_used_columns": min(
            r["influence_test"]["weakest_used_column"]["permutations_with_change"] for r in runs.values()),
        "control_runs": CONTROL_RUNS,
        "control_reproduced": sum(runs[r]["influence_test"]["reproduced"] for r in CONTROL_RUNS),
        "control_du_report_equals_dynamic": sum(runs[r]["du_report_equals_dynamic"] for r in CONTROL_RUNS),
        "control_static_equals_dynamic": sum(runs[r]["static_equals_dynamic"] for r in CONTROL_RUNS),
    }
    out = {
        "method": "Code-Beleg (Datei:Zeile) + Transkript-Reihenfolge (C/D) + Einfluss-Test (Spalte je 3x permutiert, "
                  "abgegebenes Script neu ausgefuehrt; nicht genutzt = keine Vorhersage aendert sich; Originalblock "
                  "muss die abgegebenen Vorhersagen exakt reproduzieren)",
        "id_columns": ID_COLS, "permutation_seeds": list(SEEDS),
        "n_features": len(cd[0]["influence_test"]["features"]),
        "checks": checks, "by_arm": by_arm, "runs": runs,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"checks": checks, "by_arm": by_arm}, indent=1, ensure_ascii=False))
    print(f"-> {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
