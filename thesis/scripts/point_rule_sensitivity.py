"""Post-hoc sensitivity: pass rates of the core suite under a point comparison.

The reported success rule (paired bootstrap, ``noninferior_paired``) replaced a
point threshold on 23 September. This script asks what level 1 would show if a
run passed only when its test-row F1 is at least that of the default baseline on
the same rows (Delta >= 0), using the unrounded difference that every run stores
in ``summary.baseline_comparison.diff``. Formal requirements (no abstention,
valid predictions, correct target) are kept as in the reported rule.

Reads only the run repository; writes thesis/facts/derived/point_rule.json.

    uv run python thesis/scripts/point_rule_sensitivity.py --runs-dir ../thesis-crisp-dm-runs
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

from scipy.stats import fisher_exact

ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "..", "facts", "derived")


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "..", "..", "..", "thesis-crisp-dm-runs"))
    args = ap.parse_args()

    core = json.load(open(os.path.join(DERIVED, "core.json"), encoding="utf-8"))
    sets = sorted(core["cell_means_f1_measured"])

    per_arm = {a: {"point": 0, "reported": 0, "n": 0} for a in ARMS}
    per_set = {s: {a: {"point": 0, "n": 0} for a in ARMS} for s in sets}
    for path in glob.glob(os.path.join(args.runs_dir, "main-*", "summary.json")):
        name = os.path.basename(os.path.dirname(path))
        ds = next((s for s in sets if name.startswith(f"main-{s}-")), None)
        if ds is None:
            continue  # post-cutoff and other sets are not part of the core suite
        s = json.load(open(path, encoding="utf-8"))
        arm = s["arm"]
        comp = s.get("baseline_comparison") or {}
        formal_ok = (not s.get("abstained")) and s.get("target_ok") is True and comp.get("diff") is not None
        point_pass = bool(formal_ok and comp["diff"] >= 0)
        per_arm[arm]["n"] += 1
        per_arm[arm]["point"] += point_pass
        per_arm[arm]["reported"] += bool(s.get("success"))
        per_set[ds][arm]["n"] += 1
        per_set[ds][arm]["point"] += point_pass

    pairs = []
    for a, b in itertools.combinations(ARMS, 2):
        ka, na = per_arm[a]["point"], per_arm[a]["n"]
        kb, nb = per_arm[b]["point"], per_arm[b]["n"]
        p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        pairs.append({"pair": f"{a}-{b}", "p": p})
    for p, h in zip(pairs, holm([p["p"] for p in pairs])):
        p["p_holm"] = h

    out = {"rule": "point comparison on test rows: baseline_comparison.diff >= 0 (unrounded), formal checks as reported",
           "per_arm": per_arm, "per_set": per_set, "fisher_holm": pairs}
    with open(os.path.join(DERIVED, "point_rule.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for a in ARMS:
        print(a, per_arm[a])
    for p in pairs:
        print(p["pair"], round(p["p"], 4), round(p["p_holm"], 3))
    for s in sets:
        print(s, {a[:4]: f'{per_set[s][a]["point"]}/{per_set[s][a]["n"]}' for a in ARMS})


if __name__ == "__main__":
    main()
