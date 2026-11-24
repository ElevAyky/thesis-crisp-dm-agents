# scripts/analyze_h1.py
"""Auswertung des H1-Nachtrags (docs/h1-single-session.md §4, festgelegt 13.11. vor den Runs).

Kern, PRIMAER: Arm A gegen Arm A-Single, beide aus dem H1-Batch (gleichzeitig gefahren,
<runs>/_h1/main_runs.json), gepaart ueber die 13 Datasets — wie H2 in zwei Stufen:
- H1a (Bestanden): SUCCESS gepoolt, Wilson-KIs, Fisher's Exact (zweiseitig).
- H1b (Qualitaet): Vorzeichen-Rang-Test (Wilcoxon) auf den Zellmitteln von f1_measured,
  zweiseitig, EXAKT per Vorzeichen-Permutation der Nicht-Null-Differenzen, Null-Differenzen
  nach Demšar 2006 ('zsplit': Raenge halb auf R+ und R-). Richtung aus R+ gegen R-.
  Sensitivitaet f1_combined, ergaenzend f1_on_success; Cliff's Delta pro Dataset.
- Urteil nur bei vollstaendigen Daten (13 Datasets x n = 6 je Arm, nichts fehlend oder
  kontaminiert), sonst 'vorlaeufig'.
Kern, ergaenzend: A(Okt., <runs>/_main) gegen A-Single und A(Okt.) gegen A(neu) =
Replikation/Drift ueber die Zeit.
Post-Cutoff (explorativ, N = 2): A(Okt.), A(neu), A-Single, B, C, D aus <runs>/_post_cutoff/
und <runs>/_post_cutoff_h1/.

Schreibt <runs>/_h1/analysis/report.md + analysis.json.

Aufruf (Repo-Root; vorher main_runs.py --batch h1 --arms hard_fsm_single hard_fsm
--aggregate-only fuer beide Manifest-Ordner):
    uv run python scripts/analyze_h1.py --runs-dir ../thesis-crisp-dm-runs
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import statistics
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_main import (ALPHA, METRICS, _success_counts, cell_matrix,  # noqa: E402
                          cell_values, cliffs_delta, cliffs_magnitude, fisher_success,
                          run_sd, wilson)

A_OLD = "hard_fsm_old"     # Arm A aus den Kern-/Post-Cutoff-Runs vom 26./27.10.
A = "hard_fsm"             # Arm A aus dem H1-Batch, gleichzeitig mit A-Single
S = "hard_fsm_single"
EXPECTED_N = 6
PC_ORDER = (A_OLD, A, S, "soft_fsm", "crisp_react", "naked_react")
LABEL = {A_OLD: "A Hard-FSM (Okt.)", A: "A Hard-FSM (neu)", S: "A-Single",
         "soft_fsm": "B Soft-FSM", "crisp_react": "C CRISP-ReAct", "naked_react": "D Naked ReAct"}


def load_cells(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["cells"]


def relabel(cells: list[dict], src: str, dst: str) -> list[dict]:
    """Kopien; nur Zellen von src, umbenannt in dst."""
    return [{**c, "arm": dst} for c in cells if c["arm"] == src]


def signed_rank_test(x: list[float], y: list[float]) -> dict:
    """Zweiseitiger Vorzeichen-Rang-Test auf x - y, exakt: alle 2^k Vorzeichen der k
    Nicht-Null-Differenzen (bedingt auf die Null-Differenzen). Statistik T = min(R+, R-),
    Null-Differenzen tragen ihre (gemittelten) Raenge je halb zu R+ und R- bei (zsplit)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    out = {"n": int(len(d)), "diffs": d.tolist(),
           "median_diff": float(np.median(d)) if len(d) else None,
           "x_ahead": int((d > 0).sum()), "y_ahead": int((d < 0).sum()),
           "ties": int((d == 0).sum()),
           "method": "exakte Vorzeichen-Permutation, Null-Differenzen zsplit"}
    if not len(d):
        return {**out, "r_plus": None, "r_minus": None, "T": None, "p": None}
    ranks = rankdata(np.abs(d))
    zero = d == 0
    half = ranks[zero].sum() / 2
    r_plus = float(ranks[d > 0].sum() + half)
    r_minus = float(ranks[d < 0].sum() + half)
    T = min(r_plus, r_minus)
    nz = ranks[~zero]
    if len(nz) == 0:
        p = 1.0
    else:
        if len(nz) > 20:
            raise ValueError("exakte Permutation nur bis 20 Nicht-Null-Differenzen")
        signs = np.array(list(itertools.product((0, 1), repeat=len(nz))), dtype=float)
        rp = signs @ nz + half
        rm = nz.sum() - signs @ nz + half
        p = float(np.mean(np.minimum(rp, rm) <= T + 1e-9))
    return {**out, "r_plus": r_plus, "r_minus": r_minus, "T": T, "p": p}


def completeness(cells: list[dict], arms: list[str], datasets: list[str]) -> dict:
    by = {(c["arm"], c["dataset"]): c for c in cells}
    problems = []
    for arm in arms:
        for ds in datasets:
            c = by.get((arm, ds))
            if c is None:
                problems.append(f"{arm}/{ds}: fehlt")
                continue
            if c.get("n_measured", 0) != EXPECTED_N:
                problems.append(f"{arm}/{ds}: n = {c.get('n_measured', 0)}")
            if c.get("missing", 0) or c.get("contaminated", 0):
                problems.append(f"{arm}/{ds}: fehlend {c.get('missing', 0)}, "
                                f"kontaminiert {c.get('contaminated', 0)}")
    return {"complete": not problems, "problems": problems}


def _direction(p, x_ahead, alpha=ALPHA) -> str:
    if p is None:
        return "kein Test"
    if p >= alpha:
        return "kein Unterschied"
    return "x vorn" if x_ahead else "y vorn"


def compare(cells: list[dict], x: str, y: str, datasets: list[str], agg: str = "mean") -> dict:
    """Gepaarter Vergleich zweier Arme ueber die Datasets (beide Stufen)."""
    cells = [c for c in cells if c["arm"] in (x, y)]
    by = {(c["arm"], c["dataset"]): c for c in cells}
    metrics = {}
    for metric in METRICS:
        kept, M, dropped = cell_matrix(cells, [x, y], metric, agg)
        test = signed_rank_test(M[:, 0].tolist(), M[:, 1].tolist()) if kept else None
        per_ds = {ds: cliffs_delta(cell_values(by[(x, ds)], metric), cell_values(by[(y, ds)], metric))
                  for ds in kept}
        mean_d = statistics.fmean(per_ds.values()) if per_ds else None
        metrics[metric] = {
            "datasets": kept, "dropped": dropped, "matrix": M.tolist(), "test": test,
            "direction": _direction(test["p"], test["r_plus"] > test["r_minus"]) if test else "kein Test",
            "cliffs_per_dataset": per_ds, "cliffs_mean": mean_d,
            "cliffs_mean_magnitude": cliffs_magnitude(mean_d) if mean_d is not None else None,
            "cliffs_large": {"x": sum(1 for d in per_ds.values() if d > 0 and cliffs_magnitude(d) == "large"),
                             "y": sum(1 for d in per_ds.values() if d < 0 and cliffs_magnitude(d) == "large")},
        }
    rates = {}
    for arm in (x, y):
        k, n = _success_counts(cells, arm)
        cs = [c for c in cells if c["arm"] == arm]
        runs = sum(c.get("n_measured", 0) for c in cs)
        calls = [c["calls_mean"] for c in cs if c.get("calls_mean") is not None]
        rates[arm] = {"success": k, "n": n, "rate": k / n if n else None, "wilson": list(wilson(k, n)),
                      "cost_per_run": (sum(c.get("cost_total", 0.0) for c in cs) / runs) if runs else None,
                      "calls_mean": statistics.fmean(calls) if calls else None}
    fisher = fisher_success(cells, x, y)
    rx, ry = rates[x]["rate"], rates[y]["rate"]
    comp = completeness(cells, [x, y], datasets)
    return {"x": x, "y": y, "metrics": metrics, "rates": rates, "run_sd": run_sd(cells, [x, y]),
            "fisher": fisher, "completeness": comp,
            "pass_direction": _direction(fisher["p"], (rx or 0) > (ry or 0)),
            "f1_direction": metrics["f1_measured"]["direction"]}


H1_WORDS = {"x vorn": "gestuetzt", "y vorn": "umgekehrt", "kein Unterschied": "nicht gestuetzt",
            "kein Test": "kein Test"}


def h1_verdicts(primary: dict) -> dict:
    """H1a (Bestanden) und H1b (F1) getrennt, wie H2a/H2b, ohne Korrektur. x = A, y = A-Single:
    'gestuetzt' = A (ein Agent pro Phase) signifikant vorn."""
    if not primary["completeness"]["complete"]:
        return {"H1a": "vorlaeufig (unvollstaendig)", "H1b": "vorlaeufig (unvollstaendig)"}
    return {"H1a": H1_WORDS[primary["pass_direction"]], "H1b": H1_WORDS[primary["f1_direction"]]}


def core_analysis(main_cells: list[dict], h1_cells: list[dict], agg: str = "mean") -> dict:
    a_old = relabel(main_cells, "hard_fsm", A_OLD)
    a_new = [c for c in h1_cells if c["arm"] == A]
    single = [c for c in h1_cells if c["arm"] == S]
    datasets = sorted({c["dataset"] for c in a_old}) or sorted({c["dataset"] for c in single})
    res = {"datasets": datasets}
    if a_new and single:
        res["primary"] = compare(a_new + single, A, S, datasets, agg)
        res["verdicts"] = h1_verdicts(res["primary"])
    if a_old and single:
        res["old_vs_single"] = compare(a_old + single, A_OLD, S, datasets, agg)
    if a_old and a_new:
        res["replication"] = compare(a_old + a_new, A_OLD, A, datasets, agg)
    return res


def post_cutoff(old_cells: list[dict], new_cells: list[dict]) -> dict:
    """Altes A heisst A_OLD; A aus dem H1-Batch bleibt 'hard_fsm' (= A neu)."""
    cells = relabel(old_cells, "hard_fsm", A_OLD) + [dict(c) for c in old_cells if c["arm"] != "hard_fsm"]
    cells += [dict(c) for c in new_cells if c["arm"] in (A, S)]
    missing_arms = [arm for arm in (A, S) if not any(c["arm"] == arm for c in new_cells)]
    rows = []
    for c in sorted(cells, key=lambda c: (c["dataset"], PC_ORDER.index(c["arm"])
                                          if c["arm"] in PC_ORDER else 9)):
        f1 = cell_values(c, "f1_measured")
        rows.append({"arm": c["arm"], "dataset": c["dataset"],
                     "success": (c.get("outcomes") or {}).get("SUCCESS", 0),
                     "n": c.get("n_measured", 0), "missing": c.get("missing", 0),
                     "f1_measured_mean": statistics.fmean(f1) if f1 else None,
                     "f1_measured_sd": statistics.stdev(f1) if len(f1) > 1 else None,
                     "fail_reasons": c.get("fail_reasons") or {},
                     "cost_total": c.get("cost_total"), "calls_mean": c.get("calls_mean")})
    pooled = {}
    for arm in PC_ORDER:
        k, n = _success_counts(cells, arm)
        if n:
            pooled[arm] = {"success": k, "n": n, "wilson": list(wilson(k, n))}
    comps = {}
    for x, y in ((A, S), (A_OLD, A), (A_OLD, S)):
        if x in pooled and y in pooled:
            comps[f"{x}_vs_{y}"] = fisher_success(cells, x, y)
    by = {(r["arm"], r["dataset"]): r for r in rows}
    drift = {ds: by[(A, ds)]["f1_measured_mean"] - by[(A_OLD, ds)]["f1_measured_mean"]
             for (arm, ds) in by if arm == A and (A_OLD, ds) in by
             and by[(A, ds)]["f1_measured_mean"] is not None
             and by[(A_OLD, ds)]["f1_measured_mean"] is not None}
    return {"rows": rows, "pooled": pooled, "fisher": comps, "f1_drift_new_minus_old": drift,
            "missing_arms": missing_arms}


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _render_compare(cmp: dict, title: str) -> list[str]:
    x, y = cmp["x"], cmp["y"]
    lines = [f"## {title}", ""]
    if not cmp["completeness"]["complete"]:
        lines += ["**Unvollstaendig:** " + "; ".join(cmp["completeness"]["problems"][:12]), ""]
    lines += ["| Arm | SUCCESS | Wilson-KI | Kosten/Run | Calls/Run | Run-SD (Median) |",
              "|---|---|---|---|---|---|"]
    for arm in (x, y):
        r, sd = cmp["rates"][arm], cmp["run_sd"][arm]
        lines.append(f"| {LABEL[arm]} | {r['success']}/{r['n']} | [{_f(r['wilson'][0])}, "
                     f"{_f(r['wilson'][1])}] | {_f(r['cost_per_run'], 2)} $ | "
                     f"{_f(r['calls_mean'], 2)} | {_f(sd['median_sd'], 4)} |")
    fi = cmp["fisher"]
    lines += ["", f"Fisher's Exact (SUCCESS): {fi['table']}, p = {_f(fi['p'])} → {cmp['pass_direction']}", ""]
    for metric, label in METRICS.items():
        m, t = cmp["metrics"][metric], cmp["metrics"][metric]["test"]
        lines += [f"### {metric}", "", label, ""]
        if not t:
            lines += ["keine vollstaendigen Paare", ""]
            continue
        lines += [f"- Vorzeichen-Rang-Test ({t['method']}): R+ = {_f(t['r_plus'], 1)}, "
                  f"R− = {_f(t['r_minus'], 1)}, p = {_f(t['p'], 4)} → {m['direction']}",
                  f"- {t['n']} Datasets: {LABEL[x]} vorn {t['x_ahead']}, {LABEL[y]} vorn "
                  f"{t['y_ahead']}, gleich {t['ties']}; Median-Differenz {_f(t['median_diff'], 4)}",
                  f"- Cliff's Delta im Mittel {_f(m['cliffs_mean'])} ({m['cliffs_mean_magnitude']}); "
                  f"large: {LABEL[x]} {m['cliffs_large']['x']}, {LABEL[y]} {m['cliffs_large']['y']}"]
        if m["dropped"]:
            lines.append(f"- ohne vollstaendiges Paar: {', '.join(m['dropped'])}")
        lines += ["", f"| Dataset | {LABEL[x]} | {LABEL[y]} | Diff | Cliff |", "|---|---|---|---|---|"]
        for ds, (vx, vy) in zip(m["datasets"], m["matrix"]):
            lines.append(f"| {ds} | {_f(vx, 4)} | {_f(vy, 4)} | {_f(vx - vy, 4)} | "
                         f"{_f(m['cliffs_per_dataset'][ds], 2)} |")
        lines.append("")
    return lines


def render_report(res: dict) -> str:
    lines = ["# H1-Nachtrag: A gegen A-Single", "",
             "Vorab festgelegt in `docs/h1-single-session.md` (13.11.2026).", ""]
    core = res.get("core") or {}
    if "verdicts" in core:
        v = core["verdicts"]
        lines += [f"**H1a (Bestanden): {v['H1a']} · H1b (F1, primaer f1_measured): {v['H1b']}**", ""]
    if "primary" in core:
        lines += _render_compare(core["primary"], "Kern, primaer: A (neu) gegen A-Single (gleichzeitig)")
    if "old_vs_single" in core:
        lines += _render_compare(core["old_vs_single"], "Kern, ergaenzend: A (Okt.) gegen A-Single")
    if "replication" in core:
        lines += _render_compare(core["replication"], "Kern, Replikation/Drift: A (Okt.) gegen A (neu)")
    pc = res.get("post_cutoff")
    if pc:
        lines += ["## Post-Cutoff (explorativ, N = 2)", ""]
        if pc["missing_arms"]:
            lines += [f"**Fehlende Arme im H1-Batch:** {', '.join(pc['missing_arms'])}", ""]
        lines += ["| Dataset | Arm | SUCCESS | F1 (Mittel) | F1-SD | Fail-Gruende | Kosten | Calls |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in pc["rows"]:
            lines.append(f"| {row['dataset']} | {LABEL.get(row['arm'], row['arm'])} | "
                         f"{row['success']}/{row['n']} | {_f(row['f1_measured_mean'], 4)} | "
                         f"{_f(row['f1_measured_sd'], 4)} | {row['fail_reasons'] or ''} | "
                         f"{_f(row['cost_total'], 2)} | {_f(row['calls_mean'], 2)} |")
        lines += ["", "Gepoolt: " + " · ".join(f"{LABEL[a]} {v['success']}/{v['n']}"
                                              for a, v in pc["pooled"].items()), ""]
        for name, fi in pc["fisher"].items():
            lines.append(f"- Fisher {name}: {fi['table']}, p = {_f(fi['p'])} (beschreibend)")
        if pc["f1_drift_new_minus_old"]:
            lines.append("- F1-Drift A (neu) − A (Okt.): " + ", ".join(
                f"{ds} {_f(dv, 4)}" for ds, dv in sorted(pc["f1_drift_new_minus_old"].items())))
        lines.append("")
    return "\n".join(lines)


def run_analysis(runs_dir: str, agg: str = "mean") -> dict:
    res = {"agg": agg}
    main_p = os.path.join(runs_dir, "_main", "main_runs.json")
    h1_p = os.path.join(runs_dir, "_h1", "main_runs.json")
    if os.path.exists(main_p) and os.path.exists(h1_p):
        res["core"] = core_analysis(load_cells(main_p), load_cells(h1_p), agg)
    pc_old = os.path.join(runs_dir, "_post_cutoff", "main_runs.json")
    pc_new = os.path.join(runs_dir, "_post_cutoff_h1", "main_runs.json")
    if os.path.exists(pc_old) and os.path.exists(pc_new):
        res["post_cutoff"] = post_cutoff(load_cells(pc_old), load_cells(pc_new))
    out = os.path.join(runs_dir, "_h1", "analysis")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "analysis.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(res))
    return res


def main():
    ap = argparse.ArgumentParser(description="H1-Nachtrag: A vs. A-Single")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    args = ap.parse_args()
    res = run_analysis(args.runs_dir, args.agg)
    try:                                    # Windows-Konsole (cp1252) kennt − und — nicht
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    print(render_report(res))


if __name__ == "__main__":
    main()
