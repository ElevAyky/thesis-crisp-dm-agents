# scripts/memorization_check.py
"""Memorisierungs- und Aussenzugriffs-Check ueber die Haupt-Runs (Kap. 6, explorative
These 23.09. in Kap. 7.2: "Prozessanleitung kann memorisiertes Vorwissen verdraengen").

Pro Run werden durchsucht: die Abschlussnachrichten aller Worker-Calls (events.jsonl)
und alle Python-Dateien, die der Worker im Arbeitsverzeichnis hinterlassen hat.

Drei Signale:
- **named**    — der Agent benennt das Dataset mit einem Namen/Detail, das NICHT im
                 Task-Prompt steht (z. B. "Statlog", "Wisconsin", Rebsorten bei wine).
                 Muster, die den eigenen Task-Prompt treffen, werden verworfen (Test).
- **generic**  — allgemeiner Wiedererkennungs-Hinweis ("UCI", "well-known dataset", ...).
- **external** — Code greift auf Daten ausserhalb des Arbeitsverzeichnisses zu
                 (fetch_openml, sklearn-Loader, HTTP, Kaggle, pip). Das waere ein Leck
                 um den Holdout herum und ist unabhaengig von der These zu pruefen.

Grenze: Nur Abschlussnachrichten und hinterlassene Dateien sind sichtbar, nicht jeder
einzelne Shell-Befehl (inline `python -c`). Ein Nicht-Treffer beweist kein Nicht-Wissen.

    uv run python scripts/memorization_check.py --runs-dir ../thesis-crisp-dm-runs
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crispfsm.manifests import load_manifest  # noqa: E402

# Dataset-spezifische Namen/Details (Original-Quelle, Autoren, Klassen-Semantik), die
# der Task-Prompt NICHT verraet. Jedes Muster wird zusaetzlich gegen den eigenen
# Prompt geprueft (specific_patterns) — trifft es dort, faellt es raus.
SPECIFIC = {
    "adult": [r"census[- ]income", r"uci[- ]adult", r"adult[- ](data ?set|census)", r"adult\.data"],
    "bank_marketing": [r"bank[- ]marketing", r"portug", r"moro et al"],
    "breast_cancer": [r"wisconsin", r"\bwdbc\b", r"load_breast_cancer"],
    "cmc": [r"contraceptive method choice", r"indonesia", r"\bcmc\b"],
    "cnae_9": [r"\bcnae"],
    "connect_4": [r"\btromp\b", r"connect-4 (data ?set|opening)"],
    "covertype": [r"covertype", r"\bcovtype\b", r"roosevelt", r"colorado"],
    "credit_g": [r"german credit", r"statlog", r"credit-g", r"hofmann"],
    "eucalyptus": [r"bulluch", r"new zealand"],
    "fashion_mnist": [r"fashion[- ]?mnist", r"zalando", r"\bmnist\b"],
    "phoneme": [r"\belena\b", r"\broarr\b"],
    "vehicle": [r"statlog", r"turing institute", r"siebert"],
    "wine": [r"load_wine", r"uci wine", r"wine recognition", r"forina", r"barolo",
             r"grignolino", r"barbera", r"piemont"],
}
GENERIC = [r"\buci\b", r"\bopenml\b", r"(well[- ]known|classic|famous|standard|bekannte\w*|"
           r"klassische\w*|beruehmte\w*|berühmte\w*) (ml[- ])?(data ?set|datensatz|benchmark)",
           r"this is (the|a) [\w\- ]{0,40}data ?set"]
EXTERNAL = [r"fetch_openml", r"sklearn\.datasets", r"\bload_(wine|breast_cancer|digits|iris)\b",
            r"urllib", r"requests\.(get|post)", r"\bwget\b", r"\bcurl\b", r"\bkaggle\b",
            r"\bopenml\.", r"pip install"]


def specific_patterns(dataset: str, task_prompt: str) -> list[str]:
    """Nur Muster, die der eigene Task-Prompt nicht schon enthaelt."""
    return [p for p in SPECIFIC.get(dataset, []) if not re.search(p, task_prompt, re.I)]


def run_texts(run_dir: str) -> tuple[str, str]:
    """(Abschlussnachrichten aller Worker-Calls, Inhalt aller hinterlassenen .py-Dateien)."""
    messages = []
    events = os.path.join(run_dir, "events.jsonl")
    if os.path.exists(events):
        with open(events, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line).get("worker_text") or "")
    code = []
    for path in glob.glob(os.path.join(run_dir, "work", "**", "*.py"), recursive=True):
        with open(path, encoding="utf-8", errors="replace") as f:
            code.append(f.read())
    return "\n".join(messages), "\n".join(code)


def scan_run(run_dir: str, patterns: list[str]) -> dict:
    messages, code = run_texts(run_dir)
    both = messages + "\n" + code
    hit = lambda pats, text: sorted({m.group(0).lower() for p in pats
                                     for m in re.finditer(p, text, re.I)})
    return {"named": hit(patterns, both), "generic": hit(GENERIC, both),
            "external": hit(EXTERNAL, code)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--pattern", default="main-*-*-r*")
    ap.add_argument("--json", default=None, help="Rohdaten pro Run hierhin schreiben")
    args = ap.parse_args()

    prompts = {}
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, args.pattern))):
        summary = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary):
            continue
        _, dataset, arm, _ = os.path.basename(run_dir).split("-")
        if dataset not in prompts:
            try:
                prompts[dataset] = load_manifest(dataset, args.manifests_dir).task_prompt
            except FileNotFoundError:  # main-* enthaelt auch die Post-Cutoff-Runs; ihre Manifeste liegen eine Ebene tiefer
                prompts[dataset] = load_manifest(dataset, os.path.join(args.manifests_dir, "post_cutoff")).task_prompt
        with open(summary, encoding="utf-8") as f:
            s = json.load(f)
        if s.get("outcome") == "ERROR":
            continue
        rows.append({"run": os.path.basename(run_dir), "dataset": dataset, "arm": arm,
                     "f1": s.get("best_f1_measured"),
                     **scan_run(run_dir, specific_patterns(dataset, prompts[dataset]))})

    cells = defaultdict(list)
    for r in rows:
        cells[(r["dataset"], r["arm"])].append(r)
    arms = sorted({r["arm"] for r in rows})
    print(f"{'dataset':15s} " + " ".join(f"{a:>22s}" for a in arms))
    print(f"{'':15s} " + " ".join(f"{'named/gen/ext of n':>22s}" for _ in arms))
    for ds in sorted({r["dataset"] for r in rows}):
        line = f"{ds:15s} "
        for a in arms:
            c = cells.get((ds, a), [])
            line += (f"{sum(bool(r['named']) for r in c):>9d}/{sum(bool(r['generic']) for r in c)}"
                     f"/{sum(bool(r['external']) for r in c)} of {len(c):<3d}   ")
        print(line)
    print()
    for a in arms:
        ar = [r for r in rows if r["arm"] == a]
        print(f"{a:12s} named {sum(bool(r['named']) for r in ar):3d}/{len(ar)}  "
              f"generic {sum(bool(r['generic']) for r in ar):3d}/{len(ar)}  "
              f"external {sum(bool(r['external']) for r in ar):3d}/{len(ar)}")
    # F1: benannt vs. nicht benannt, nur innerhalb derselben Zelle (Dataset-Effekt raus)
    diffs = []
    for c in cells.values():
        yes = [r["f1"] for r in c if r["named"] and r["f1"] is not None]
        no = [r["f1"] for r in c if not r["named"] and r["f1"] is not None]
        if yes and no:
            diffs.append(statistics.fmean(yes) - statistics.fmean(no))
    if diffs:
        print(f"\nF1 benannt minus nicht benannt, innerhalb gemischter Zellen: "
              f"Mittel {statistics.fmean(diffs):+.4f} ueber {len(diffs)} Zellen")
    ext = [r for r in rows if r["external"]]
    if ext:
        print("\nAUSSENZUGRIFF im Code:")
        for r in ext:
            print(f"  {r['run']:40s} {r['external']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
