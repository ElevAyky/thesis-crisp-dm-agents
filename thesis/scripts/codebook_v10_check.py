#!/usr/bin/env python
"""Pruefung zu Codebuch v1.1 (Faktenblatt kap7.md F7-930): Entscheiden die Muster, die
code_process.py ueber die Listen von Codebuch v1.0 hinaus enthaelt, irgendeinen Code-Wert?

Der zweite Kodierer (process_coding_manual.json) hat die Listen von v1.0 als abgeschlossen
gelesen. Dieses Script kodiert alle C/D-Runs erneut mit den auf v1.0 verengten Mustern und
vergleicht die Werte mit thesis/facts/derived/process_coding.json (28 Hauptcodes + Hilfscodes
bu_target, bu_metric, leakage_mention_any, prep_select_rule). Schreibt nichts.

Aufruf aus dem Repo-Root:
    uv run python thesis/scripts/codebook_v10_check.py [--runs-dir ../thesis-crisp-dm-runs]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_process as cp  # noqa: E402
import extract_actions as ea  # noqa: E402

CODING = os.path.join(cp.REPO, "thesis", "facts", "derived", "process_coding.json")
SAMPLE = os.path.join(cp.REPO, "thesis", "facts", "derived", "validation_sample.json")


def _ci(p):
    return re.compile(p, re.I)


# Muster mit Zusaetzen im Script, verengt auf die Listen von Codebuch v1.0 (codebook.md Paragraf 4)
V10_RX = {
    "du_schema": re.compile(r"\.shape\b|\.dtypes\b|\.head\(|\.info\(|print\([^)]*\.columns"),
    "du_missing": re.compile(r"\.(isna|isnull|notna)\(|\.info\(|hasnans"),
    "du_relations": re.compile(r"\.corr\(|corrwith\(|crosstab\(|mutual_info|\bchi2\(|f_classif|\.groupby\("),
    "prep_select_tool": re.compile(r"\b(VarianceThreshold|SelectKBest|SelectFromModel|RFE)\s*\("),
}
V10_TX = {
    "rationale": _ci(r"imbalanc|\bbalanced\b|minority|majority|skew|class distribution|dominat"),
    "success": _ci(r"success criteri|success (is|means)|\b(beat|exceed|outperform)s?\b[^.\n]{0,40}\bbaseline"
                   r"|must (beat|exceed|outperform)"),
    "eval_review_next": _ci(r"next steps?|further (iteration|improvement|work|tuning)|could (be )?(further )?improv"
                            r"|future work|\brecommend|process review|overlook|possible (improvement|action)s?"
                            r"|would (try|explore)|if (more )?time"),
    "leak": re.compile(r"(?i:leak)|\bIDs?\b|(?i:\bid[- ]?(like|column)\b|identifier|row[- ]?(id|index))"),
    "eval_business_ref": _ci(r"success criteri|business (objective|goal|need)|\bobjectives?\b|\bgoal\b"),
    "eval_business_verb": _ci(r"\b(is|was) (met|achieved|satisfied|fulfilled)\b|\bmeets\b|\bsatisfies\b"
                              r"|\bachieves\b|\bfulfils\b"),
    "baseline_num": _ci(r"(baseline|most[- ]frequent|majority)[^.\n]{0,60}\d\.\d"
                        r"|\d\.\d[^.\n]{0,40}(baseline|most[- ]frequent|majority)"),
}
V10_TGT_REF = r"\b(y|TARGET|target|label)\b"
CODES = [c for c, _, _ in cp.BIN_REACT] + ["bu_target", "bu_metric", "leakage_mention_any", "prep_select_rule"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=ea.DEFAULT_RUNS)
    args = ap.parse_args()
    cp.ARGS.runs_dir = args.runs_dir
    with open(CODING, encoding="utf-8") as f:
        coded = json.load(f)["runs"]
    with open(SAMPLE, encoding="utf-8") as f:
        sample = json.load(f)
    sample_ids = {r["run"] for r in sample["runs"]}
    cp.RX.update(V10_RX)
    cp.TX.update(V10_TX)
    cp.TGT_REF = V10_TGT_REF
    changed = []
    n = 0
    for rid, r in coded.items():
        if r["arm"] not in cp.REACT:
            continue
        n += 1
        v = cp.code_react(ea.load_run(args.runs_dir, rid))["values"]
        changed += [(rid, c, r["values"].get(c), v.get(c)) for c in CODES if v.get(c) != r["values"].get(c)]
    print(f"{n} C/D-Runs neu kodiert mit den Listen von Codebuch v1.0, {len(CODES)} Codes je Run")
    print(f"geaenderte Zellen: {len(changed)} (davon in der Validierungs-Stichprobe: "
          f"{sum(1 for x in changed if x[0] in sample_ids)})")
    for x in changed:
        print("  ", *x)


if __name__ == "__main__":
    main()
