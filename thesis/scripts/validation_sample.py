"""Deterministic validation sample for the blind re-coding of the process coding (Kap. 7, F7-9xx).

24 runs: 12 per arm C (crisp_react) and D (naked_react); per arm 3 easy, 3 medium, 4 hard (core
terciles, F6-401) and 2 post-cutoff (student_health, ev_purchase).

Rule (reproducible):
  * Universe = run folders `main-<set>-<arm>-r<k>` in ../thesis-crisp-dm-runs that contain a
    summary.json (archived prefixes automem_/gitctx_/promptv1_ never match `main-`).
  * One `random.Random(SEED)`; iteration order fixed: arms sorted, strata in the order
    easy, medium, hard, post_cutoff.
  * Per stratum: draw k *distinct* data sets (rng.sample over the sorted set names), then per
    drawn set (sorted) one replicate (rng.choice over the sorted replicate numbers).
    Distinct sets maximise coverage of the data sets within the small sample.

Output: thesis/facts/derived/validation_sample.json (+ stdout).
Run: uv run python thesis/scripts/validation_sample.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

SEED = 20261109
ARMS = ["crisp_react", "naked_react"]
QUOTA = {"easy": 3, "medium": 3, "hard": 4, "post_cutoff": 2}

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO.parent / "thesis-crisp-dm-runs"
OUT = REPO / "thesis" / "facts" / "derived" / "validation_sample.json"

# Core terciles (F6-401, thesis/facts/kap6.md; = derived/terciles.json "assignment").
TERCILE = {
    "wine": "easy", "breast_cancer": "easy", "cnae_9": "easy", "covertype": "easy",
    "phoneme": "medium", "fashion_mnist": "medium", "adult": "medium", "vehicle": "medium",
    "bank_marketing": "hard", "credit_g": "hard", "connect_4": "hard", "eucalyptus": "hard",
    "cmc": "hard",
}
POST_CUTOFF = sorted(p.stem for p in (REPO / "manifests" / "post_cutoff").glob("*.yaml"))

RX_RUN = re.compile(r"^main-(?P<set>[a-z0-9_]+)-(?P<arm>crisp_react|naked_react)-r(?P<rep>\d+)$")


def stratum(ds: str) -> str:
    if ds in POST_CUTOFF:
        return "post_cutoff"
    return TERCILE[ds]


def main() -> None:
    # cross-check the hard-coded terciles against the derived file, if present
    terc_file = REPO / "thesis" / "facts" / "derived" / "terciles.json"
    if terc_file.exists():
        assignment = json.loads(terc_file.read_text(encoding="utf-8"))["assignment"]
        assert assignment == TERCILE, "tercile assignment differs from derived/terciles.json"
    assert POST_CUTOFF == ["ev_purchase", "student_health"], POST_CUTOFF

    universe: dict[tuple[str, str], dict[str, list[int]]] = {}
    for d in sorted(RUNS.iterdir()):
        m = RX_RUN.match(d.name)
        if not m or not (d / "summary.json").exists():
            continue
        key = (m["arm"], stratum(m["set"]))
        universe.setdefault(key, {}).setdefault(m["set"], []).append(int(m["rep"]))

    rng = random.Random(SEED)
    sample = []
    for arm in ARMS:
        for st, k in QUOTA.items():
            sets = universe[(arm, st)]
            names = sorted(sets)
            chosen = sorted(rng.sample(names, k))
            for ds in chosen:
                rep = rng.choice(sorted(sets[ds]))
                sample.append({
                    "run": f"main-{ds}-{arm}-r{rep}",
                    "arm": arm, "set": ds, "rep": rep, "stratum": st,
                })

    out = {
        "seed": SEED,
        "rule": "per arm (sorted) x stratum (easy, medium, hard, post_cutoff): rng.sample of k distinct "
                "sets (sorted names), then rng.choice of one replicate per set (sorted)",
        "quota": QUOTA,
        "universe_size": {f"{a}/{s}": sum(len(v) for v in sets.values())
                          for (a, s), sets in sorted(universe.items())},
        "n": len(sample),
        "runs": sample,
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in sample:
        print(f"{r['arm']:12s} {r['stratum']:12s} {r['run']}")
    print("n =", len(sample), "->", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
