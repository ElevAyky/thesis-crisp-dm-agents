# scripts/analyze_main.py
"""Auswertung der Haupt-Runs nach experiment-design.md §5 (festgelegt 05.08.).

Liest <runs-dir>/_main/main_runs.json (scripts/main_runs.py) und schreibt nach
<runs-dir>/_main/analysis/:
- report.md      — Tabellen fuers Methoden-/Ergebniskapitel + Decker-Meeting
- analysis.json  — alle Zahlen maschinenlesbar
- cd_<metrik>.png  — Critical-Difference-Diagramme (Demšar 2006)
- box_<metrik>.png — Boxplots pro Arm x Dataset (Deskriptiv, §5.1 Punkt 5)

Verfahren:
- **Friedman** ueber die Arme, Datasets als Bloecke (Zellwert = Mittel, Sensitivitaet
  Median ueber die n Runs) + **Iman-Davenport-F** (Demšar Gl. 2, weniger konservativ).
- **Nemenyi** post-hoc: CD = q_alpha * sqrt(k(k+1)/(6N)), q_alpha aus der
  studentisierten Spannweite / sqrt(2) (= Demšar Tab. 5a).
- **Cliff's Delta** pro Arm-Paar: pro Dataset ueber die Runs (unabhaengige
  Stichproben innerhalb eines Blocks), berichtet als Mittel + Verteilung der
  Magnituden ueber die Datasets. Magnituden nach Romano et al. 2006.
- **Wilson-KIs** auf SUCCESS-Raten (pro Arm gepoolt + pro Zelle).
- **Fisher's Exact** paarweise auf gepoolten SUCCESS-Zaehlungen (Enforcement-
  Messpunkt Kontraktbruch, §5.2 Befund 3). Fuer das BU-Robustheits-Set dieselbe
  Funktion auf den vergifteten Paaren.

Zwei Stufen (festgelegt 23.09., experiment-design §3):
- **Stufe 1 — Bestanden:** SUCCESS-Rate (durchgelaufen, gueltige Predictions, nicht
  signifikant schlechter als die Default-Baseline) -> Wilson-KIs + Fisher.
- **Stufe 2 — Qualitaet:** Friedman/Nemenyi auf F1. Drei Berichte:
  - f1_measured   — PRIMAER: F1 jedes Runs mit gueltigen Predictions (best_f1_measured),
    unabhaengig von der Erfolgsregel — vermischt die Stufen nicht
  - f1_combined   — Sensitivitaet: Fail/Abstain = 0 (zaehlt Scheitern doppelt: Stufe 1 + 2)
  - f1_on_success — ergaenzend: nur SUCCESS-Runs (Zellen ohne SUCCESS -> Block faellt raus)

Aufruf (Repo-Root):
    uv run python scripts/analyze_main.py --runs-dir ../thesis-crisp-dm-runs
    uv run python scripts/analyze_main.py --runs-dir ../thesis-crisp-dm-runs --agg median
Vorher `main_runs.py --aggregate-only`, damit main_runs.json aktuell ist.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import os
import statistics

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist
from scipy.stats import fisher_exact, rankdata, studentized_range

ARM_ORDER = ("hard_fsm", "soft_fsm", "crisp_react", "naked_react")
ARM_LABEL = {"hard_fsm": "A Hard-FSM", "soft_fsm": "B Soft-FSM",
             "crisp_react": "C CRISP-ReAct", "naked_react": "D Naked ReAct"}
# Reihenfolge = Reihenfolge im Report: primaer zuerst (23.09.)
METRICS = {
    "f1_measured": "Primaer: F1 aller Runs mit gueltigen Predictions (best_f1_measured)",
    "f1_combined": "Sensitivitaet: kombinierter Score (Fail/Abstain = 0)",
    "f1_on_success": "Ergaenzend: F1 nur auf SUCCESS-Runs",
}
ALPHA = 0.05


# --- Zellen -----------------------------------------------------------------

def cell_values(cell: dict, metric: str) -> list[float]:
    """Run-Werte einer Zelle. f1_combined fehlt in main_runs.json vor 23.09. ->
    aus F1-auf-Erfolg + Nullen fuer alle Nicht-SUCCESS-Runs rekonstruieren."""
    if metric == "f1_combined" and "f1_combined" not in cell:
        succ = cell.get("f1_on_success") or []
        return list(succ) + [0.0] * (cell.get("n_measured", 0) - len(succ))
    return list(cell.get(metric) or [])


def _agg(xs: list[float], agg: str) -> float:
    return statistics.median(xs) if agg == "median" else statistics.fmean(xs)


def cell_matrix(cells: list[dict], arms: list[str], metric: str,
                agg: str = "mean") -> tuple[list[str], np.ndarray, list[str]]:
    """N x k-Matrix (Datasets x Arme). Datasets, bei denen eine Zelle keinen Wert
    hat, fallen raus (Friedman braucht vollstaendige Bloecke) -> `dropped`."""
    by = {(c["arm"], c["dataset"]): c for c in cells}
    datasets = sorted({c["dataset"] for c in cells})
    kept, dropped, rows = [], [], []
    for ds in datasets:
        vals = [cell_values(by[(a, ds)], metric) if (a, ds) in by else [] for a in arms]
        if all(vals):
            kept.append(ds)
            rows.append([_agg(v, agg) for v in vals])
        else:
            dropped.append(ds)
    return kept, np.array(rows, dtype=float).reshape(len(rows), len(arms)), dropped


# --- Friedman / Nemenyi -----------------------------------------------------

def rank_matrix(M: np.ndarray) -> np.ndarray:
    """Pro Block ranken, hoechster F1 = Rang 1, Ties gemittelt (Demšar §3.2.2)."""
    return np.vstack([rankdata(-row) for row in M])


def friedman(M: np.ndarray) -> dict:
    """Friedman-chi2 + Iman-Davenport-F nach Demšar 2006 Gl. 1/2."""
    N, k = M.shape
    R = rank_matrix(M).mean(axis=0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    denom = N * (k - 1) - chi2
    F = (N - 1) * chi2 / denom if denom > 0 else math.inf
    return {
        "N": N, "k": k, "avg_ranks": R.tolist(),
        "chi2": float(chi2), "p_chi2": float(chi2_dist.sf(chi2, k - 1)),
        "F": float(F), "df": [k - 1, (k - 1) * (N - 1)],
        "p_F": float(f_dist.sf(F, k - 1, (k - 1) * (N - 1))) if math.isfinite(F) else 0.0,
    }


def nemenyi_cd(k: int, n_datasets: int, alpha: float = ALPHA) -> float:
    q = studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * n_datasets)))


def nemenyi_pairs(avg_ranks: dict[str, float], cd: float) -> list[dict]:
    return [{"a": a, "b": b, "rank_diff": abs(avg_ranks[a] - avg_ranks[b]),
             "significant": abs(avg_ranks[a] - avg_ranks[b]) > cd}
            for a, b in itertools.combinations(avg_ranks, 2)]


def cd_cliques(avg_ranks: dict[str, float], cd: float) -> list[list[str]]:
    """Maximale Gruppen nicht signifikant verschiedener Arme (die dicken Linien im
    CD-Diagramm). Einzelgaenger bekommen keine Linie."""
    order = sorted(avg_ranks, key=avg_ranks.get)
    groups = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and avg_ranks[order[j + 1]] - avg_ranks[order[i]] <= cd:
            j += 1
        if j > i:
            groups.append(order[i:j + 1])
    return [g for g in groups
            if not any(set(g) < set(h) for h in groups)]


# --- Effektgroessen / Raten -------------------------------------------------

def cliffs_delta(x: list[float], y: list[float]) -> float:
    """P(X > Y) - P(X < Y) ueber alle Paare."""
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    return (gt - lt) / (len(x) * len(y))


def cliffs_magnitude(d: float) -> str:
    """Romano et al. 2006: 0.147 / 0.33 / 0.474."""
    d = abs(d)
    return ("negligible" if d < 0.147 else "small" if d < 0.33
            else "medium" if d < 0.474 else "large")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _success_counts(cells: list[dict], arm: str) -> tuple[int, int]:
    cs = [c for c in cells if c["arm"] == arm]
    k = sum((c.get("outcomes") or {}).get("SUCCESS", 0) for c in cs)
    n = sum(c.get("n_measured", 0) for c in cs)
    return k, n


def fisher_success(cells: list[dict], arm_a: str, arm_b: str) -> dict:
    ka, na = _success_counts(cells, arm_a)
    kb, nb = _success_counts(cells, arm_b)
    table = [[ka, na - ka], [kb, nb - kb]]
    odds, p = fisher_exact(table)
    return {"a": arm_a, "b": arm_b, "table": table, "odds_ratio": float(odds), "p": float(p)}


def pairwise_cliffs(cells: list[dict], arms: list[str], metric: str,
                    datasets: list[str]) -> list[dict]:
    by = {(c["arm"], c["dataset"]): c for c in cells}
    out = []
    for a, b in itertools.combinations(arms, 2):
        per_ds = {}
        for ds in datasets:
            x, y = cell_values(by[(a, ds)], metric), cell_values(by[(b, ds)], metric)
            if x and y:
                per_ds[ds] = cliffs_delta(x, y)
        ds_vals = list(per_ds.values())
        mags = [cliffs_magnitude(d) for d in ds_vals]
        mean_d = statistics.fmean(ds_vals) if ds_vals else None
        out.append({"a": a, "b": b, "per_dataset": per_ds, "mean_delta": mean_d,
                    "mean_magnitude": cliffs_magnitude(mean_d) if mean_d is not None else None,
                    "a_better_large": sum(1 for d, m in zip(ds_vals, mags) if d > 0 and m == "large"),
                    "b_better_large": sum(1 for d, m in zip(ds_vals, mags) if d < 0 and m == "large")})
    return out


# --- Plots ------------------------------------------------------------------

def plot_cd(avg_ranks: dict[str, float], cd: float, path: str, title: str) -> None:
    """CD-Diagramm im Demšar-Stil: Rang-Achse (1 links = bester), CD-Balken oben,
    dicke Linien verbinden nicht signifikant verschiedene Arme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(avg_ranks)
    order = sorted(avg_ranks, key=avg_ranks.get)
    cliques = cd_cliques(avg_ranks, cd)
    half = math.ceil(k / 2)
    # y in Datenkoordinaten: Achse 0, darueber Ticks + CD-Balken, darunter
    # Clique-Balken, dann die Arm-Beschriftungen.
    clique_y = [-0.25 - 0.18 * j for j in range(len(cliques))]
    label_top = min(clique_y, default=-0.1) - 0.35
    levels = [label_top - 0.4 * i for i in range(half)]
    fig, ax = plt.subplots(figsize=(7, 0.9 + 0.45 * (half + len(cliques))))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(levels[-1] - 0.3, 1.2)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.1, color="black", lw=1)
        ax.text(r, 0.15, str(r), ha="center", va="bottom", fontsize=9)
    ax.hlines(0.75, 1, 1 + cd, color="black", lw=1.5)
    ax.vlines([1, 1 + cd], 0.68, 0.82, color="black", lw=1.5)
    ax.text(1 + cd / 2, 0.86, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    for i, arm in enumerate(order):
        r = avg_ranks[arm]
        left = i < half
        level = levels[i] if left else levels[k - 1 - i]
        x_text = 0.55 if left else k + 0.45
        ax.plot([r, r, x_text], [0, level, level], color="black", lw=0.8)
        ax.text(x_text + (-0.02 if left else 0.02), level,
                f"{ARM_LABEL.get(arm, arm)} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)
    for y, g in zip(clique_y, cliques):
        ax.hlines(y, avg_ranks[g[0]] - 0.03, avg_ranks[g[-1]] + 0.03, color="black", lw=3)
    ax.set_title(title, fontsize=10, pad=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_boxes(cells: list[dict], arms: list[str], metric: str, path: str,
               title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {(c["arm"], c["dataset"]): c for c in cells}
    datasets = sorted({c["dataset"] for c in cells})
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(datasets) * len(arms) / 2), 4.5))
    width = 0.8 / len(arms)
    colors = ["#1f77b4", "#6baed6", "#d62728", "#fc9272"]
    for i, arm in enumerate(arms):
        data = [cell_values(by[(arm, ds)], metric) if (arm, ds) in by else [] for ds in datasets]
        pos = [d + (i - (len(arms) - 1) / 2) * width for d in range(len(datasets))]
        bp = ax.boxplot([x if x else [np.nan] for x in data], positions=pos, widths=width * 0.9,
                        patch_artist=True, showfliers=True, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[ARM_ORDER.index(arm) % 4] if arm in ARM_ORDER else "grey")
        ax.plot([], [], color=colors[ARM_ORDER.index(arm) % 4] if arm in ARM_ORDER else "grey",
                lw=6, label=ARM_LABEL.get(arm, arm))
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, ncol=len(arms), loc="lower left")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Analyse + Report -------------------------------------------------------

def analyze_metric(cells: list[dict], arms: list[str], metric: str, agg: str,
                   alpha: float) -> dict:
    datasets, M, dropped = cell_matrix(cells, arms, metric, agg)
    res = {"metric": metric, "agg": agg, "datasets": datasets, "dropped": dropped,
           "matrix": M.tolist(), "friedman": None, "cd": None, "nemenyi": None,
           "cliques": None, "cliffs": pairwise_cliffs(cells, arms, metric, datasets),
           "skipped": None}
    if len(arms) < 3:
        res["skipped"] = f"Friedman/Nemenyi braucht mindestens 3 Arme (vorhanden: {len(arms)})"
        return res
    if len(datasets) < 2:
        res["skipped"] = f"zu wenige vollstaendige Datasets ({len(datasets)})"
        return res
    fr = friedman(M)
    ranks = dict(zip(arms, fr["avg_ranks"]))
    cd = nemenyi_cd(len(arms), len(datasets), alpha)
    res.update(friedman=fr, cd=cd, avg_ranks=ranks,
               nemenyi=nemenyi_pairs(ranks, cd), cliques=cd_cliques(ranks, cd))
    return res


def rates(cells: list[dict], arms: list[str]) -> dict:
    per_arm = {}
    for arm in arms:
        k, n = _success_counts(cells, arm)
        per_arm[arm] = {"success": k, "n": n, "rate": k / n if n else None,
                        "wilson": wilson(k, n)}
    per_cell = []
    for c in sorted(cells, key=lambda c: (c["dataset"], arms.index(c["arm"]))):
        k = (c.get("outcomes") or {}).get("SUCCESS", 0)
        per_cell.append({"arm": c["arm"], "dataset": c["dataset"], "success": k,
                         "n": c.get("n_measured", 0), "wilson": wilson(k, c.get("n_measured", 0)),
                         "fail_reasons": c.get("fail_reasons") or {}})
    fisher = [fisher_success(cells, a, b) for a, b in itertools.combinations(arms, 2)]
    return {"per_arm": per_arm, "per_cell": per_cell, "fisher": fisher}


def run_sd(cells: list[dict], arms: list[str]) -> dict:
    """Run-SD pro Zelle auf f1_measured als eigene AV (§5.2 Befund 2)."""
    out = {}
    for arm in arms:
        sds = [statistics.stdev(v) for c in cells if c["arm"] == arm
               for v in [cell_values(c, "f1_measured")] if len(v) > 1]
        out[arm] = {"median_sd": statistics.median(sds) if sds else None,
                    "max_sd": max(sds) if sds else None, "cells": len(sds)}
    return out


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "—"


def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _p_txt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def render_report(data: dict, result: dict) -> str:
    arms = result["arms"]
    L = [f"# Auswertung Haupt-Runs",
         "",
         f"Quelle `main_runs.json` · Modell {data.get('model')} · n-Soll {data.get('replicates')} · "
         f"Zellwert = {result['agg']} · alpha = {result['alpha']}",
         "",
         "## Stand der Daten", "",
         "| Arm | gemessen | offen | kontaminiert | Kosten $ |", "|---|---|---|---|---|"]
    for arm in arms:
        cs = [c for c in data["cells"] if c["arm"] == arm]
        L.append(f"| {ARM_LABEL.get(arm, arm)} | {sum(c.get('n_measured', 0) for c in cs)} | "
                 f"{sum(c.get('missing', 0) for c in cs)} | {sum(c.get('contaminated', 0) for c in cs)} | "
                 f"{sum(c.get('cost_total', 0) or 0 for c in cs):.2f} |")
    r = result["rates"]
    L += ["", "## Stufe 1 — Bestanden (SUCCESS-Rate, Wilson 95 %)", "",
          "SUCCESS = vollstaendig durchgelaufen, gueltige Predictions, auf denselben Test-Zeilen "
          "nicht signifikant schlechter als die Default-Baseline (gepaarter Bootstrap, 95 %).", "",
          "| Arm | SUCCESS / n | Rate | Wilson-KI |", "|---|---|---|---|"]
    for arm, v in r["per_arm"].items():
        lo, hi = v["wilson"]
        L.append(f"| {ARM_LABEL.get(arm, arm)} | {v['success']} / {v['n']} | {_fmt(v['rate'])} | "
                 f"[{_fmt(lo)}, {_fmt(hi)}] |")
    L += ["", "| Paar (Fisher exact, gepoolt) | Tabelle [[S,F],[S,F]] | OR | p |", "|---|---|---|---|"]
    for f in r["fisher"]:
        L.append(f"| {ARM_LABEL.get(f['a'], f['a'])} vs {ARM_LABEL.get(f['b'], f['b'])} | "
                 f"{f['table']} | {_fmt(f['odds_ratio'], 2)} | {_p(f['p'])} |")
    L += ["", "### fail_reason pro Zelle (nur Zellen mit Nicht-SUCCESS)", "",
          "| Dataset | Arm | SUCCESS / n | fail_reasons |", "|---|---|---|---|"]
    for c in r["per_cell"]:
        if c["success"] < c["n"]:
            L.append(f"| {c['dataset']} | {ARM_LABEL.get(c['arm'], c['arm'])} | {c['success']} / {c['n']} | "
                     f"{', '.join(f'{k} {v}' for k, v in c['fail_reasons'].items())} |")
    L += ["", "# Stufe 2 — Qualitaet (Friedman + Nemenyi auf F1)"]
    for metric, label in METRICS.items():
        m = result["metrics"][metric]
        L += ["", f"## {label}", ""]
        if m["dropped"]:
            L.append(f"Ohne vollstaendigen Block (raus): {', '.join(m['dropped'])}  ")
        if m["skipped"]:
            L += [f"*Test ausgelassen: {m['skipped']}.*", ""]
        else:
            fr = m["friedman"]
            L += [f"**Friedman:** chi2 = {fr['chi2']:.2f} (df {fr['k'] - 1}), {_p_txt(fr['p_chi2'])} · "
                  f"**Iman-Davenport:** F({fr['df'][0]}, {fr['df'][1]}) = {fr['F']:.2f}, {_p_txt(fr['p_F'])} · "
                  f"N = {fr['N']}, k = {fr['k']}, **CD = {m['cd']:.2f}**", "",
                  f"![CD {metric}](cd_{metric}.png)", "",
                  "| Arm | mittlerer Rang |", "|---|---|"]
            for arm, r in sorted(m["avg_ranks"].items(), key=lambda x: x[1]):
                L.append(f"| {ARM_LABEL.get(arm, arm)} | {r:.2f} |")
            L += ["", "| Paar | Rang-Diff | Nemenyi sig. | Cliff's δ (Mittel) | Magnitude | large a>b / b>a |",
                  "|---|---|---|---|---|---|"]
            cl = {(c["a"], c["b"]): c for c in m["cliffs"]}
            for p in m["nemenyi"]:
                c = cl[(p["a"], p["b"])]
                L.append(f"| {ARM_LABEL.get(p['a'], p['a'])} vs {ARM_LABEL.get(p['b'], p['b'])} | "
                         f"{p['rank_diff']:.2f} | {'ja' if p['significant'] else 'nein'} | "
                         f"{_fmt(c['mean_delta'])} | {c['mean_magnitude'] or '—'} | "
                         f"{c['a_better_large']} / {c['b_better_large']} |")
        # Zellen-Matrix
        L += ["", "| Dataset | " + " | ".join(ARM_LABEL.get(a, a) for a in arms) + " |",
              "|---|" + "---|" * len(arms)]
        for ds, row in zip(m["datasets"], m["matrix"]):
            best = max(row)
            L.append(f"| {ds} | " + " | ".join(
                f"**{_fmt(v)}**" if v == best else _fmt(v) for v in row) + " |")
        L += ["", f"![Boxplot {metric}](box_{metric}.png)"]
    L += ["", "## Run-SD (f1_measured) — Stabilitaet als AV", "",
          "| Arm | Median-SD | Max-SD | Zellen |", "|---|---|---|---|"]
    for arm, v in result["run_sd"].items():
        L.append(f"| {ARM_LABEL.get(arm, arm)} | {_fmt(v['median_sd'], 4)} | {_fmt(v['max_sd'], 4)} | {v['cells']} |")
    return "\n".join(L) + "\n"


def run_analysis(input_path: str, out_dir: str, agg: str = "mean",
                 alpha: float = ALPHA) -> dict:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    cells = [c for c in data["cells"] if c.get("n_measured", 0) > 0]
    present = {c["arm"] for c in cells}
    arms = [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))
    os.makedirs(out_dir, exist_ok=True)
    result = {"input": os.path.abspath(input_path), "agg": agg, "alpha": alpha,
              "arms": arms, "metrics": {}, "rates": rates(cells, arms),
              "run_sd": run_sd(cells, arms)}
    for metric, label in METRICS.items():
        m = analyze_metric(cells, arms, metric, agg, alpha)
        result["metrics"][metric] = m
        if m["friedman"]:
            plot_cd(m["avg_ranks"], m["cd"], os.path.join(out_dir, f"cd_{metric}.png"),
                    f"{label} — N = {len(m['datasets'])}, alpha = {alpha}")
        if cells:
            plot_boxes(cells, arms, metric, os.path.join(out_dir, f"box_{metric}.png"), label)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(data, result))
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main():
    ap = argparse.ArgumentParser(description="Friedman/Nemenyi/CD + Raten auf main_runs.json")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--input", default=None, help="Default: <runs-dir>/_main/main_runs.json")
    ap.add_argument("--out-dir", default=None, help="Default: <runs-dir>/_main/analysis")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean",
                    help="Zellwert ueber die n Runs (median = Sensitivitaet)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()
    src = args.input or os.path.join(args.runs_dir, "_main", "main_runs.json")
    out = args.out_dir or os.path.join(args.runs_dir, "_main",
                                       "analysis" if args.agg == "mean" else f"analysis_{args.agg}")
    result = run_analysis(src, out, args.agg, args.alpha)
    for metric in METRICS:
        m = result["metrics"][metric]
        if m["friedman"]:
            fr = m["friedman"]
            ranks = ", ".join(f"{a} {r:.2f}" for a, r in sorted(m["avg_ranks"].items(), key=lambda x: x[1]))
            print(f"{metric:14s} N={fr['N']:2d}  {_p_txt(fr['p_F'])}  CD={m['cd']:.2f}  Raenge: {ranks}")
        else:
            print(f"{metric:14s} {m['skipped']}")
    print(f"Report: {os.path.join(out, 'report.md')}")


if __name__ == "__main__":
    main()
