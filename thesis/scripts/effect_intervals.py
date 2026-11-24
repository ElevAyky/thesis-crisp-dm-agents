"""Post-hoc effect sizes: mean difference of the cell means against Arm D.

For each of Arms A, B and C, the difference to Arm D is taken per core dataset
(primary view, ``cell_means_f1_measured`` in derived/core.json), averaged over
the 13 datasets and given a two-sided 95 % t-interval (df = N - 1). The
intervals are descriptive and were computed after the results (external
feedback, 24 November); they are not a planned equivalence test.

Reads derived/core.json; writes derived/effect_intervals.json.

    uv run python thesis/scripts/effect_intervals.py
"""
from __future__ import annotations

import json
import math
import os
import statistics

from scipy.stats import t

HERE = os.path.dirname(os.path.abspath(__file__))
DERIVED = os.path.join(HERE, "..", "facts", "derived")
REF = "naked_react"
ARMS = ["hard_fsm", "soft_fsm", "crisp_react"]


def main() -> None:
    core = json.load(open(os.path.join(DERIVED, "core.json"), encoding="utf-8"))
    cm = core["cell_means_f1_measured"]
    sets = sorted(cm)
    n = len(sets)
    crit = t.ppf(0.975, n - 1)
    out = {"reference": REF, "n_datasets": n, "t_crit": crit, "view": "cell_means_f1_measured", "diffs": {}}
    for arm in ARMS:
        d = [cm[s][arm] - cm[s][REF] for s in sets]
        m, sd = statistics.mean(d), statistics.stdev(d)
        half = crit * sd / math.sqrt(n)
        out["diffs"][arm] = {"mean": m, "ci_lo": m - half, "ci_hi": m + half, "min": min(d), "max": max(d)}
        print(f"{arm} - {REF}: {m:+.4f} [{m - half:+.4f}, {m + half:+.4f}]")
    with open(os.path.join(DERIVED, "effect_intervals.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
