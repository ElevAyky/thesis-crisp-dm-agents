# thesis/scripts/make_tables.py
"""LaTeX-Tabellen fuer Kap. 5 und 6 -> thesis/tables/*.tex (+ thesis/tables/TABLES.md).

Liest ausschliesslich thesis/facts/derived/*.json (collect_facts.py), also dieselben Werte wie
die Faktenblaetter. Jede Datei ist eine vollstaendige table-Umgebung mit \\caption und \\label;
im Kapitel genuegt \\input{tables/<name>}. booktabs, \\small, auf 15 cm Textbreite ausgelegt.

Aufruf (Repo-Root, nach collect_facts.py):
    uv run python thesis/scripts/make_tables.py
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DER = os.path.join(REPO, "thesis", "facts", "derived")
OUT = os.path.join(REPO, "thesis", "tables")
ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
L = {"hard_fsm": "A", "soft_fsm": "B", "crisp_react": "C", "naked_react": "D"}
NAME = {"wine": "wine", "breast_cancer": "breast cancer", "cnae_9": "cnae-9", "covertype": "covertype",
        "phoneme": "phoneme", "fashion_mnist": "fashion-mnist", "adult": "adult", "vehicle": "vehicle",
        "bank_marketing": "bank-marketing", "credit_g": "credit-g", "connect_4": "connect-4",
        "eucalyptus": "eucalyptus", "cmc": "cmc", "student_health": "student health", "ev_purchase": "EV purchase",
        "jm1": "jm1", "synth_machine": "synth machine", "uber_rides": "uber rides"}
CORE_ORDER = ["wine", "breast_cancer", "cnae_9", "covertype", "phoneme", "fashion_mnist", "adult", "vehicle",
              "bank_marketing", "credit_g", "connect_4", "eucalyptus", "cmc"]
POST_ORDER = ["student_health", "ev_purchase"]
ARMS_REF = r"Arms as in \autoref{tab:3:arms}."
REGISTRY = []   # (file, label, caption, section, source)


def j(name):
    with open(os.path.join(DER, name), encoding="utf-8") as f:
        return json.load(f)


def n(x, nd=3, sign=False):
    """Zahl fuer LaTeX: echtes Minus im Mathemodus, optional Vorzeichen."""
    if x is None:
        return "--"
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    if s.startswith("-"):
        return f"$-${s[1:]}"
    return f"$+${s}" if sign else s


def thou(x):
    return f"{x:,.0f}"


def p(x):
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def wil(w):
    return f"[{w[0]:.3f}, {w[1]:.3f}]"


def bold_best(vals: dict, nd=3):
    top = max(round(v, nd) for v in vals.values())
    return {a: (rf"\textbf{{{n(v, nd)}}}" if round(v, nd) == top else n(v, nd)) for a, v in vals.items()}


# Kurztitel fuer das Tabellenverzeichnis (externes Feedback 24.11.2026); die Caption ueber der Tabelle bleibt.
SHORT_TITLES = {
    "tab5_calibration": "Calibration and variance scan",
    "tab5_batches": "Clean experiment batches",
    "tab6_passrates": "Level 1 on the core suite",
    "tab6_f1cells": "Mean macro-F1 per cell",
    "tab6_ranktests": "Friedman test over the core datasets",
    "tab6_pairs": "Pairwise comparisons",
    "tab6_runsd": "Run-to-run standard deviation per cell",
    "tab6_terciles": "Difficulty terciles as moderator",
    "tab6_postcutoff": "Post-cutoff control",
    "tab6_postcutoff_tests": "Fisher's exact test on the post-cutoff pass counts",
    "tab6_phases": "Phase-level diagnostics of the state-machine arms",
    "tab6_leakage": "Injected leakage, Arm D",
    "tab6_cost": "Cost and token usage",
    "tab6_expectations": "Expected pattern and outcome",
}


def write(name, label, caption, section, source, colspec, header, rows, *, env="tabular", width=None, note=None,
          small=True, short=None):
    short = short or SHORT_TITLES.get(name)
    lines = [r"\begin{table}[htbp]", r"  \centering"]
    if small:
        lines.append(r"  \small")
    lines += [rf"  \caption[{short}]{{{caption}}}" if short else rf"  \caption{{{caption}}}", rf"  \label{{{label}}}"]
    if env == "tabularx":
        lines.append(rf"  \begin{{tabularx}}{{{width or r'\textwidth'}}}{{{colspec}}}")
    else:
        lines.append(rf"  \begin{{tabular}}{{{colspec}}}")
    lines.append(r"    \toprule")
    for h in header:
        lines.append("    " + h + r" \\")
    lines.append(r"    \midrule")
    for r in rows:
        if r in (r"\midrule", r"\addlinespace"):
            lines.append("    " + r)
        else:
            lines.append("    " + r + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(rf"  \end{{{env}}}")
    if note:
        lines.append(r"  \par\smallskip\footnotesize " + note)
    lines.append(r"\end{table}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("%!TEX root = ../thesis.tex\n% generiert von thesis/scripts/make_tables.py — nicht von Hand aendern\n")
        f.write("\n".join(lines) + "\n")
    REGISTRY.append((name + ".tex", label, caption, section, source))
    print(f"  -> thesis/tables/{name}.tex")


def row(*cells):
    return " & ".join(str(c) for c in cells)


# =========================================================================== Kap. 5

def tab_batches():
    tl, co = j("timeline.json"), j("costs.json")
    order = [("leak_perfect", "Leakage, perfect"), ("leak_spurious", "Leakage, spurious"),
             ("core", "Core suite"), ("post_cutoff", "Post-cutoff control")]

    def period(t):
        a, b = t["start_local"], t["end_local"]
        da, ta = a.split(" ")[0], a.split(" ")[1]
        db, tb = b.split(" ")[0], b.split(" ")[1]
        fmt = lambda d: f"{int(d[8:10])} Oct"  # noqa: E731
        return (rf"\mbox{{{fmt(da)} {ta}--{tb}}}" if da == db
                else rf"\mbox{{{fmt(da)} {ta}}} -- \mbox{{{fmt(db)} {tb}}}")
    rows = []
    for key, name in order:
        t = tl[key]
        arms = "D" if t["arms"] == ["naked_react"] else "A--D"
        rows.append(row(name, arms, t["n_runs"], t["harness_commits"][0], period(t), n(t["wall_clock_h"], 2),
                        len(t["gaps_over_60min"]), n(t["cost_usd"], 2)))
    rows.append(r"\midrule")
    rows.append(row("Total", "", sum(tl[k]["n_runs"] for k, _ in order), "", "", "", "", n(co["_all_clean_batches_total"], 2)))
    write("tab5_batches", "tab:5:batches",
          "Clean experiment batches in execution order. All runs used Claude Code 2.1.280 and the same prompt "
          "fingerprint; harness \\texttt{9bb6a91} differs from \\texttt{14e99e4} in documentation only. \\emph{Period}: "
          "local time, CEST until 25 Oct 03:00, CET afterwards; \\emph{Hours}: elapsed time from the first to the last "
          "transcript time stamp; \\emph{Pauses}: interruptions "
          "longer than 60 minutes, resumed by the runner; \\emph{USD}: cost at list prices.",
          "5.2", "derived/timeline.json, derived/costs.json",
          r"llrl>{\RaggedRight\arraybackslash}p{2.9cm}rrr",
          [row("Batch", "Arms", "Runs", "Harness", "Period (2026)", "Hours", "Pauses", "USD")], rows)


def tab_calibration():
    cal = j("calibration.json")
    cells = sorted(cal["cells"], key=lambda c: (c["phase"] != "calib", not c["in_core_suite"], c["arm"] != "hard_fsm",
                                                CORE_ORDER.index(c["dataset"]) if c["dataset"] in CORE_ORDER else 99, c["dataset"]))
    rows, prev = [], None
    for c in cells:
        grp = ("Calibration" if c["phase"] == "calib" else "Variance scan" if c["in_core_suite"] else "Scan, outside suite")
        if prev is not None and grp != prev:
            rows.append(r"\midrule")
        rows.append(row(grp if grp != prev else "", NAME[c["dataset"]], L[c["arm"]],
                        f"{c['outcomes'].get('SUCCESS', 0)}/{c['n']}", n(c["mean"], 3),
                        n(c["sd"], 4) if c["sd"] is not None else "--", n(c["cost_per_run"], 2)))
        prev = grp
    h = cal["heuristic"]
    write("tab5_calibration", "tab:5:calibration",
          "Calibration (8 Aug) and variance scan (8--9 Aug) with $n = 3$ per cell. These pre-study runs measured "
          "\\emph{weighted} F1, passed at a fixed threshold of 0.5 and ran without the isolated worker context; they "
          "served as design input only. The governing cell for the number of replicates was covertype under Arm D "
          f"(pooled SD {n(h['worst_sd'], 4)} over calibration and scan, \\mbox{{$n \\geq {h['n_required_worst']:.2f}$}}, hence $n = 6$). "
          "Mean F1 over runs with valid predictions. " + ARMS_REF,
          "5.1", "derived/calibration.json",
          "lllcrrr", [row("Stage", "Dataset", "Arm", "Passed", "Mean F1", "Run SD", "USD/run")], rows)


def tab_contamination():
    """Erster (archivierter) Durchlauf gegen die saubere Wiederholung -- einziger zitierbarer Einsatz der
    archivierten Werte (benannter Vergleich in Kap. 5, F5-417 bis F5-425). Status-Spalte = Einordnung aus
    docs/roadmap.md "haelt / faellt weg" (F5-426, F5-427), keine Rechnung."""
    con, core, rsd, pc, co = j("contamination.json"), j("core.json"), j("runsd_core.json"), j("postcutoff.json"), j("costs.json")
    o, op = con["archived_comparison"]["automem__main"], con["archived_comparison"]["automem__post_cutoff"]
    l2 = core["level2"]["f1_measured"]

    def pair(label, first, clean, status, fmt, status2=""):
        if isinstance(first, dict):
            a = row(label, "first", *[fmt(first[x]) for x in ARMS], status)
            b = row("", "second", *[fmt(clean[x]) for x in ARMS], status2)
        else:
            a = row(label, "first", rf"\multicolumn{{4}}{{c}}{{{fmt(first)}}}", status)
            b = row("", "second", rf"\multicolumn{{4}}{{c}}{{{fmt(clean)}}}", status2)
        return [a, b, r"\addlinespace"]

    rows = []
    rows += pair("Friedman test, $p$ (Iman--Davenport)", o["friedman_p_F"], l2["friedman"]["p_F"], "holds", n)
    # B/C tauschen zwischen den Durchlaeufen, nur A vorn und D hinten halten (Review 25.11.)
    rows += pair("Mean rank, core suite", o["avg_ranks"], l2["avg_ranks"], "holds", lambda v: f"{v:.2f}",
                 "(A first, D last)")
    rows += pair("Passed runs, core suite (of 78)", o["success"],
                 {a: core["level1"]["per_arm"][a]["success"] for a in ARMS}, "--", str)
    rows += pair("Median run SD, core suite", o["run_sd_median"], {a: rsd["per_arm"][a]["median_sd"] for a in ARMS},
                 "falls", lambda v: n(v, 4))
    rows += pair("Mean F1 on covertype", o["cell_means"]["covertype"], core["cell_means_f1_measured"]["covertype"],
                 "falls", n)
    rows += pair("Passed runs, post-cutoff (of 12)", op["success"],
                 {a: pc["level1"]["per_arm"][a]["success"] for a in ARMS}, "falls", str)
    rows += pair("Cost in USD, core suite", o["cost_total"], co["core"]["total"], "--", lambda v: n(v, 2))
    rows += pair("Cost in USD, post-cutoff", op["cost_total"], co["post_cutoff"]["total"], "--", lambda v: n(v, 2))
    rows = rows[:-1]
    write("tab5_contamination", "tab:5:contamination",
          "Findings of the archived first pass, which ran with the shared client memory, against the second pass. "
          "The values of the first pass are reported for this comparison only. \\emph{Status}: whether the finding "
          "suggested by the first pass holds in the second pass. Mean F1 is the cell mean of the primary F1 measure "
          "(\\autoref{sec:3:Variables}). " + ARMS_REF,
          "5.2.1", "derived/contamination.json archived_comparison (F5-417 bis F5-425); derived/core.json, "
          "runsd_core.json, postcutoff.json, costs.json; Status: F5-426, F5-427",
          "llrrrrl", [row("Measure", "Pass", "A", "B", "C", "D", "Status")], rows,
          short="First pass against second pass")


# =========================================================================== Kap. 6

def tab_passrates():
    core, sec = j("core.json"), j("secondary_core.json")
    rows = []
    for a in ARMS:
        x, s = core["level1"]["per_arm"][a], sec["per_arm"][a]
        fr = ", ".join(f"{v}$\\times$ below baseline" for k, v in x["fail_reasons"].items()) or "--"
        rows.append(row(L[a], f"{x['success']}/{x['n']}", n(x["rate"]), wil(x["wilson"]), fr,
                        f"{s['sig_better']}/{s['n']}", n(s["rate"]), wil(s["wilson"])))
    pair = {(q["a"], q["b"]): q for q in sec["pairs"]}
    ac = pair[("hard_fsm", "crisp_react")]
    write("tab6_passrates", "tab:6:passrates",
          "Level~1 on the core suite (78 runs per arm). \\emph{Passed}: not significantly worse than the default "
          "baseline (paired bootstrap); \\emph{Sig.\\ better}: lower bound of the same interval above zero (secondary, "
          "descriptive). Wilson 95\\,\\% intervals. Fisher's exact test: all pairs $p = 1.000$ for passing; for "
          f"\\emph{{sig.\\ better}} the smallest $p$ is A vs.\\ C with {p(ac['p'])} (Holm {p(ac['p_holm'])}). " + ARMS_REF,
          "6.1", "derived/core.json level1, derived/secondary_core.json",
          "lrrlcrrl", [row("", r"\multicolumn{4}{c}{Passed}", r"\multicolumn{3}{c}{Sig.\ better than baseline}"),
                       r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}" + "\n    "
                       + row("Arm", "$k/n$", "Rate", "Wilson CI", "Failures", "$k/n$", "Rate", "Wilson CI")], rows)


def tab_f1cells():
    core, ter = j("core.json"), j("terciles.json")
    maj = j("expectations.json")["p4_imbalanced_success_A_gt_D"]["majority_share"]  # F6-909
    m, base = core["cell_means_f1_measured"], core["baseline_test"]
    rows, prev = [], None
    tname = {"easy": "easy", "medium": "medium", "hard": "hard"}
    for ds in CORE_ORDER:
        t = ter["assignment"][ds]
        if prev is not None and t != prev:
            rows.append(r"\midrule")
        b = bold_best(m[ds])
        rows.append(row(NAME[ds], tname[t], f"{100 * maj[ds]:.0f}\\,\\%", n(base[ds]), *[b[a] for a in ARMS]))
        prev = t
    rows.append(r"\midrule")
    rk = core["level2"]["f1_measured"]["avg_ranks"]
    rows.append(row("Mean rank", "", "", "", *[f"{rk[a]:.2f}" for a in ARMS]))
    write("tab6_f1cells", "tab:6:f1cells",
          "Mean macro-F1 per cell (six runs, primary view) on the test split, grouped by difficulty tercile. "
          "\\emph{Majority}: share of the majority class in the training rows. \\emph{Baseline}: default reference "
          "model on the same test rows. The best value per "
          "dataset at three decimals is set in bold (ties in bold for all). " + ARMS_REF,
          "6.2", "derived/core.json cell_means_f1_measured, baseline_test, level2; derived/terciles.json; "
          "derived/expectations.json p4_imbalanced_success_A_gt_D.majority_share",
          "llrrrrrr", [row("Dataset", "Tercile", "Majority", "Baseline", "A", "B", "C", "D")], rows)


def tab_ranktests():
    core, med = j("core.json"), j("median.json")
    views = [("f1_measured", "mean", "F1 of all valid runs (primary)"), ("f1_combined", "mean", "Combined score (fail $= 0$)"),
             ("f1_on_success", "mean", "F1 of passed runs only"), ("f1_measured", "median", "Primary view, cell median")]
    rows = []
    for key, agg, label in views:
        if agg == "mean":
            r = core["level2"][key]
            fr, rk = r["friedman"], r["avg_ranks"]
        else:
            r = med["median_core"][key]
            fr, rk = r["friedman"], r["avg_ranks"]
        rows.append(row(label, n(fr["chi2"], 2), p(fr["p_chi2"]), n(fr["F"], 2), p(fr["p_F"]), *[f"{rk[a]:.2f}" for a in ARMS]))
    write("tab6_ranktests", "tab:6:ranktests",
          "Friedman test over the 13 core datasets ($k = 4$, $N = 13$, df $= 3$) with the Iman--Davenport correction "
          "($F(3, 36)$) and mean ranks (1 = best). The Nemenyi critical difference is $\\mathit{CD} = 1.30$; no pair of arms "
          "differs significantly in any view. " + ARMS_REF,
          "6.2", "derived/core.json level2, derived/median.json",
          "lrrrrrrrr", [row("", r"\multicolumn{2}{c}{Friedman}", r"\multicolumn{2}{c}{Iman--Davenport}",
                            r"\multicolumn{4}{c}{Mean rank}"),
                        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}" + "\n    "
                        + row("View", r"$\chi^2$", "$p$", "$F$", "$p$", "A", "B", "C", "D")], rows)


def tab_pairs():
    core = j("core.json")
    r = core["level2"]["f1_measured"]
    nem = {(q["a"], q["b"]): q for q in r["nemenyi"]}
    rows = []
    for c in r["cliffs"]:
        q = nem[(c["a"], c["b"])]
        rows.append(row(f"{L[c['a']]} vs.\\ {L[c['b']]}", f"{q['rank_diff']:.2f}", "no" if not q["significant"] else "yes",
                        n(c["mean_delta"], 3, sign=True), c["mean_magnitude"], f"{c['a_better_large']} / {c['b_better_large']}"))
    write("tab6_pairs", "tab:6:pairs",
          "Pairwise comparisons in the primary view. Rank difference against $\\mathit{CD} = 1.30$ (Nemenyi); Cliff's "
          "$\\delta$ computed per dataset over the six runs of each arm and averaged over the 13 datasets (magnitude after "
          "Romano et al.~\\cite{romano2006appropriate}); \\emph{Large}: number of the 13 datasets on which $\\delta$ is large "
          "($|\\delta| \\geq 0.474$) in favor of the first arm / of the second arm; for example, 4 / 3 for A vs.\\ D means "
          "that A is clearly ahead on four datasets and D on three. "
          + ARMS_REF,
          "6.2", "derived/core.json level2.f1_measured.nemenyi, cliffs",
          "lrcrlc", [row("Pair", "Rank diff.", "Nemenyi sig.", r"Mean $\delta$", "Magnitude", "Large")], rows)


def tab_runsd():
    rsd = j("runsd_core.json")
    rows = []
    for ds in CORE_ORDER:
        rows.append(row(NAME[ds], *[n(rsd["per_cell"][ds][a], 4) for a in ARMS]))
    rows.append(r"\midrule")
    rows.append(row("Median", *[n(rsd["per_arm"][a]["median_sd"], 4) for a in ARMS]))
    rows.append(row("Maximum", *[n(rsd["per_arm"][a]["max_sd"], 4) for a in ARMS]))
    write("tab6_runsd", "tab:6:runsd",
          "Run-to-run standard deviation of macro-F1 per cell (six runs, primary view), core suite. Descriptive only. "
          + ARMS_REF,
          "6.3", "derived/runsd_core.json", "lrrrr", [row("Dataset", "A", "B", "C", "D")], rows)


def tab_terciles():
    ter = j("terciles.json")
    rows = []
    for i, t in enumerate(["easy", "medium", "hard"]):
        pt = ter["per_tercile"][t]
        dss = ", ".join(NAME[d] for d in pt["datasets"])
        if i:
            rows.append(r"\midrule")
        rows.append(row(rf"\multicolumn{{5}}{{l}}{{\emph{{{t.capitalize()}}} ({dss})}}"))
        rows.append(row(r"\quad Mean rank", *[f"{pt['arms'][a]['mean_rank']:.2f}" for a in ARMS]))
        rows.append(row(r"\quad Mean F1 $-$ baseline", *[n(pt['arms'][a]['mean_f1_minus_baseline'], 3, sign=True) for a in ARMS]))
        rows.append(row(r"\quad Passed", *[f"{pt['arms'][a]['success']}/{pt['arms'][a]['n']}" for a in ARMS]))
    write("tab6_terciles", "tab:6:terciles",
          "Difficulty terciles as moderator (descriptive). Mean rank: ranks of the cell means within each dataset, "
          "averaged over the datasets of the tercile; mean F1 $-$ baseline: average distance of the cell means to the "
          "default baseline on the test split. " + ARMS_REF,
          "6.4", "derived/terciles.json", "lrrrr", [row("", "A", "B", "C", "D")], rows)


def tab_postcutoff():
    pc = j("postcutoff.json")
    rows = []
    for i, ds in enumerate(POST_ORDER):
        if i:
            rows.append(r"\midrule")
        cells = {c["arm"]: c for c in pc["level1"]["per_cell"] if c["dataset"] == ds}
        rows.append(row(rf"\multicolumn{{5}}{{l}}{{\emph{{{NAME[ds]}}} (baseline {n(pc['baseline_test'][ds])})}}"))
        rows.append(row(r"\quad Passed (of 6)", *[cells[a]["success"] for a in ARMS]))
        b = bold_best(pc["f1_cell_means"][ds])
        rows.append(row(r"\quad Mean F1", *[b[a] for a in ARMS]))
        rows.append(row(r"\quad Mean F1 $-$ baseline", *[n(pc["f1_cell_minus_baseline"][ds][a], 3, sign=True) for a in ARMS]))
        rows.append(row(r"\quad Run SD", *[n(pc["run_sd"]["per_cell"][ds][a], 4) for a in ARMS]))
    rows.append(r"\midrule")
    pa = pc["level1"]["per_arm"]
    rows.append(row("Passed, both sets (of 12)", *[pa[a]["success"] for a in ARMS]))
    rows.append(row(r"\quad Wilson 95\,\% CI", *[wil(pa[a]["wilson"]) for a in ARMS]))
    md = pc["median_diff_to_baseline_post"]
    rows.append(row(r"Median F1 $-$ baseline, all runs", *[n(md[a]["median_diff"], 3, sign=True) for a in ARMS]))
    mc = pc["median_diff_to_baseline_core"]
    rows.append(row(r"\quad same, core suite", *[n(mc[a]["median_diff"], 3, sign=True) for a in ARMS]))
    write("tab6_postcutoff", "tab:6:postcutoff",
          "Post-cutoff control: two synthetic Kaggle Playground datasets released after the model's training cutoff, "
          "six runs per cell. All failures are \\emph{below baseline}. With $N = 2$ the results are descriptive; the "
          "Friedman test has no power ($\\mathit{CD} = 3.32$). Best mean F1 per dataset at three decimals in bold. "
          "The last two rows compare the median distance to the "
          "baseline with the core suite (memorization check). " + ARMS_REF,
          "6.5", "derived/postcutoff.json", "lrrrr", [row("", "A", "B", "C", "D")], rows)


def tab_postcutoff_tests():
    pc = j("postcutoff.json")
    rows = []
    for q in pc["level1"]["pairs"]:
        t = q["table"]
        rows.append(row(f"{L[q['a']]} vs.\\ {L[q['b']]}", f"{t[0][0]}/{sum(t[0])} vs.\\ {t[1][0]}/{sum(t[1])}", p(q["p"]), p(q["p_holm"])))
    rows.append(r"\midrule")
    po = pc["pooled_fsm_vs_react"]
    rows.append(row(r"A+B vs.\ C+D (post hoc)", f"{po['AB'][0]}/{po['AB'][1]} vs.\\ {po['CD'][0]}/{po['CD'][1]}", p(po["p"]), "--"))
    write("tab6_postcutoff_tests", "tab:6:postcutoff-tests",
          "Fisher's exact test on the post-cutoff pass counts, raw and Holm-adjusted over the six pairs. The pooled "
          "comparison of the state-machine arms (A+B) with the free arms (C+D) was chosen after seeing the data and is "
          "exploratory only; it treats the runs as independent although they cluster on two datasets. " + ARMS_REF,
          "6.5", "derived/postcutoff.json level1.pairs, pooled_fsm_vs_react", "llrr",
          [row("Pair", "Passed", "$p$", "$p$ (Holm)")], rows)


def tab_phases():
    ph = j("phases.json")
    cols = [("core", "hard_fsm"), ("core", "soft_fsm"), ("post_cutoff", "hard_fsm"), ("post_cutoff", "soft_fsm")]
    X = [ph[b][a] for b, a in cols]
    rows = [row("Runs", *[x["runs"] for x in X])]
    rows.append(row("Phase attempts BU / DU / Prep", *[f"{x['attempts_total']['bu']} / {x['attempts_total']['du']} / "
                                                       f"{x['attempts_total']['prep']}" for x in X]))
    rows.append(row("Phase attempts Model / Eval", *[f"{x['attempts_total']['model']} / {x['attempts_total']['eval']}" for x in X]))
    rows.append(row("Runs with a repeated phase", *[x["runs_with"].get("any_repeat", 0) for x in X]))
    rows.append(row(r"\quad DU retries (target listed as excluded)", *[f"{x['loops'].get('retry_du', 0)} "
                                                                        f"({x['du_gate_fail_feedback'].get('target_excluded', 0)})" for x in X]))
    rows.append(row(r"\quad Model retries", *[x["loops"].get("retry_model", 0) for x in X]))
    rows.append(row(r"\quad Eval $\rightarrow$ Model jumps", *[x["loops"].get("back_eval_to_model", 0) for x in X]))
    rows.append(row(r"\quad DU $\rightarrow$ BU jumps (spec.\ mismatch)", *[x["loops"].get("back_du_to_bu", 0) for x in X]))
    rows.append(row("Worker calls, mean (max)", *[f"{x['total_calls_mean']:.2f} ({x['total_calls_max']})" for x in X]))
    rows.append(row("Judge calls, total", *[x["gate_calls_total"] if x["gate_calls_total"] else "--" for x in X]))
    rows.append(row("Worker turns, mean (max)", *[f"{x['worker_turns_mean']:.1f} ({x['worker_turns_max']})" for x in X]))
    rows.append(row("Budget exhausted", *[x["budget_exhausted"] for x in X]))
    rows.append(row("Target correct in first BU attempt", *[f"{x['bu_correct_first']}/{x['runs']}" for x in X]))
    rows.append(r"\midrule")
    rt = ph["react_turns"]
    rows.append(row("Free arms: worker turns C / D, mean (max)",
                    rf"\multicolumn{{2}}{{c}}{{{rt['core']['crisp_react']['mean']:.1f} ({rt['core']['crisp_react']['max']}) / "
                    rf"{rt['core']['naked_react']['mean']:.1f} ({rt['core']['naked_react']['max']})}}",
                    rf"\multicolumn{{2}}{{c}}{{{rt['post_cutoff']['crisp_react']['mean']:.1f} ({rt['post_cutoff']['crisp_react']['max']}) / "
                    rf"{rt['post_cutoff']['naked_react']['mean']:.1f} ({rt['post_cutoff']['naked_react']['max']})}}"))
    write("tab6_phases", "tab:6:phases",
          "Phase-level diagnostics of the state-machine arms. Retries repeat a phase after a failed gate; jumps return "
          "to an earlier phase. All DU gate failures were caused by the worker listing the target column among the "
          "excluded columns. The free arms run a single session, so only their turn counts are comparable. "
          + ARMS_REF,
          "6.6", "derived/phases.json", "lrrrr",
          [row("", r"\multicolumn{2}{c}{Core suite}", r"\multicolumn{2}{c}{Post-cutoff}"),
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n    " + row("Measure", "A", "B", "A", "B")], rows)


def tab_leakage():
    lk = j("leakage.json")
    rows = []
    for ds in CORE_ORDER:
        a, b = lk["perfect"]["per_set"][ds], lk["spurious"]["per_set"][ds]
        rows.append(row(NAME[ds], f"{a['exploited']}/3", n(a["kappa_mean"], 2), f"{a['success']}/3",
                        f"{b['exploited']}/3", n(b["kappa_mean"], 2), f"{b['success']}/3"))
    rows.append(r"\midrule")
    P, S = lk["perfect"], lk["spurious"]
    rows.append(row("Total", f"{P['exploited']}/{P['runs']}", "", f"{P['success']}/{P['runs']}",
                    f"{S['exploited']}/{S['runs']}", "", f"{S['success']}/{S['runs']}"))
    rows.append(row("Cost (USD)", rf"\multicolumn{{3}}{{c}}{{{n(P['cost'], 2)}}}", rf"\multicolumn{{3}}{{c}}{{{n(S['cost'], 2)}}}"))
    write("tab6_leakage", "tab:6:leakage",
          "Injected leakage, Arm D only, three runs per dataset. \\emph{Perfect}: a disguised column perfectly correlated "
          "with the target in training; \\emph{spurious}: an absurdly named column with about 85\\,\\% separation. "
          "\\emph{Exploited}: Cohen's $\\kappa$ between predictions and leak label with $z \\geq 3.09$ and $\\kappa \\geq 0.2$; "
          "$\\kappa$ is the mean over the three runs. The rule fixed before the runs (exploitation in at least 8 of 39 runs) "
          "was not met in either batch, so Arms A--C were not run.",
          "6.7", "derived/leakage.json", "lrrrrrr",
          [row("", r"\multicolumn{3}{c}{Perfect leak}", r"\multicolumn{3}{c}{Spurious leak}"),
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" + "\n    "
           + row("Dataset", "Exploited", r"$\kappa$", "Passed", "Exploited", r"$\kappa$", "Passed")], rows)


def tab_cost():
    co = j("costs.json")
    rows = []
    for i, (b, lab) in enumerate((("core", "Core suite (78 runs per arm)"), ("post_cutoff", "Post-cutoff (12 runs per arm)"))):
        if i:
            rows.append(r"\midrule")
        rows.append(row(rf"\multicolumn{{8}}{{l}}{{\emph{{{lab}}}}}"))
        c = co[b]
        pa, rr = c["per_arm"], c["ratios_cost_per_run"]
        for a in ARMS:
            x = pa[a]
            ratio = {"hard_fsm": f"{rr['A/C']:.2f} / {rr['A/D']:.2f}", "soft_fsm": f"{rr['B/C']:.2f} / {rr['B/D']:.2f}",
                     "crisp_react": f"-- / {rr['C/D']:.2f}", "naked_react": "--"}[a]
            t = x["tokens_per_run"]
            rows.append(row(rf"\quad {L[a]}", n(x["cost_total"], 2), n(x["cost_per_run_mean"], 3), ratio,
                            thou(t["input_tokens"]), thou(t["output_tokens"]), thou(t["cache_read_tokens"]),
                            thou(t["cache_creation_tokens"])))
        rows.append(row(r"\quad Total", n(c["total"], 2), "", "", "", "", "", ""))
    bc, bp = co["core"]["per_arm"]["soft_fsm"], co["post_cutoff"]["per_arm"]["soft_fsm"]
    write("tab6_cost", "tab:6:cost",
          "Cost and token usage, computed from the recorded token counts at the provider's list prices "
          "(USD 5 / 25 per million input / output tokens, 0.50 for cache reads, 6.25 for cache creation). Because all "
          "workers run inside the same command-line scaffold, costs are comparable between arms but not with published "
          f"API figures. The judge calls account for {100 * bc['gate_cost_share']:.1f}\\,\\% of the cost of Arm B in the "
          f"core suite and {100 * bp['gate_cost_share']:.1f}\\,\\% post-cutoff. " + ARMS_REF,
          "6.8", "derived/costs.json", "lrrlrrrr",
          [row("", r"\multicolumn{3}{c}{Cost (USD)}", r"\multicolumn{4}{c}{Tokens per run (mean)}"),
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}" + "\n    "
           + row("Arm", "Total", "Per run", "Ratio to C / D", "Input", "Output", "Cache read", "Cache creation")], rows)


def tab_expectations():
    ex, pc = j("expectations.json"), j("postcutoff.json")
    p1, p2, p3, p4, p5 = (ex[k] for k in ("p1_parity_clean", "p2_traps", "p3_runsd_A_lt_D",
                                           "p4_imbalanced_success_A_gt_D", "p5_refinement_highdim"))
    pf = p4["post_fisher_AD"]
    bm = p2["bank_marketing_f1"]
    rows = [
        row("1. Clean datasets: parity of A and D",
            f"Consistent (n.\\,s.): Friedman (Iman--Davenport) $p = {p1['core']['friedman_p_F']:.3f}$, rank difference A--D {p1['core']['rank_diff_AD']:.2f} "
            f"$<$ CD, $\\delta = {n(p1['core']['cliffs_mean_AD'], 3, sign=True).replace('$', '')}$",
            f"Not testable ($N = 2$); A above D on both sets"),
        row("2. Traps: A $>$ D",
            f"Not tested for injected leaks (D exploited {p2['leak_perfect_D_exploited']}/39 and "
            # D 1/6 = r5, der einzige Kern-Fail (Code-Sichtung work/model.py, F6-631/F6-906; nicht in derived/)
            f"{p2['leak_spurious_D_exploited']}/39) and unsolvable tasks; natural leak in bank-marketing: A excludes it "
            f"in {p2['v12_bank_marketing_A_excluded']}/6 runs and scores lower (F1 {n(bm['hard_fsm'])} vs.\\ "
            f"{n(bm['naked_react'])}); D excludes it in 1/6, its only failed run",
            "Not applicable"),
        row("3. Run SD of A $<$ D",
            f"No: median {n(p3['core_median_A'], 4)} vs.\\ {n(p3['core_median_D'], 4)}; A lower on "
            f"{p3['core_cells_A_lt_D']} of {p3['core_cells']} datasets",
            f"Yes (descriptive): {n(p3['post_median_A'], 4)} vs.\\ {n(p3['post_median_D'], 4)}"),
        row("4. Imbalanced datasets: pass rate A $>$ D",
            "No: direction only by a single run (bank-marketing, 88\\,\\% majority class: 6/6 vs.\\ 5/6), "
            "equal on all other datasets; $p = 1.000$ pooled over all 13 datasets",
            f"Direction yes: 11/12 vs.\\ 6/12, \\mbox{{$p = {pf['p']:.3f}$}} (Holm {pf['p_holm']:.3f})"),
        row("5. Refinement: D ahead on high-dimensional datasets",
            f"No: cnae-9 {n(p5['cnae_9']['hard_fsm'])} vs.\\ {n(p5['cnae_9']['naked_react'])}, "
            f"fashion-mnist {n(p5['fashion_mnist']['hard_fsm'])} vs.\\ {n(p5['fashion_mnist']['naked_react'])} (A vs.\\ D)",
            "Not applicable"),
    ]
    rows = [x for r_ in rows for x in (r_, r"\addlinespace")][:-1]
    write("tab6_expectations", "tab:6:expectations",
          "The expected pattern fixed before the main runs (\\autoref{sec:3:GoalAndHypotheses}) "
          "against the observations in the core suite and the post-cutoff control. " + ARMS_REF,
          "6 (Zusammenfassung) / 7.1", "derived/expectations.json", r">{\RaggedRight\arraybackslash}p{3.2cm}LL",
          [row("Expected pattern", "Core suite", "Post-cutoff control")], rows, env="tabularx", width=r"\textwidth")


def tab_h1():
    h = j("h1.json")
    cm = h["cell_means_f1_measured"]
    d, dr = h["diff_A_new_minus_single"], h["diff_A_new_minus_A_old"]
    old = {ds: v[0] for ds, v in cm["A_old_vs_A_new"].items()}
    fails = {}
    for x in h["failed_runs"]:
        _, ds, arm, _ = x["run_id"].split("-")
        fails[(arm, ds)] = fails.get((arm, ds), 0) + 1

    def mark(arm, ds):
        k = fails.get((arm, ds))
        return "" if not k else rf"$^{{{6 - k}/6}}$"

    rows = []
    for ds in CORE_ORDER:
        a, s = cm["A_new_vs_single"][ds]
        rows.append(row(NAME[ds], n(a) + mark("hard_fsm", ds), n(s) + mark("hard_fsm_single", ds), n(d[ds], sign=True),
                        n(old[ds]), n(dr[ds], sign=True)))
    pr, rp = h["analysis"]["core"]["primary"], h["analysis"]["core"]["replication"]
    ra = pr["rates"]
    rows.append(r"\midrule")
    rows.append(row("Passed", f"{ra['hard_fsm']['success']}/{ra['hard_fsm']['n']}",
                    f"{ra['hard_fsm_single']['success']}/{ra['hard_fsm_single']['n']}", "",
                    f"{rp['rates']['hard_fsm_old']['success']}/{rp['rates']['hard_fsm_old']['n']}", ""))
    rows.append(row("Signed-rank $p$", "", "", p(pr["metrics"]["f1_measured"]["test"]["p"]), "",
                    p(rp["metrics"]["f1_measured"]["test"]["p"])))
    write("tab6_h1", "tab:6:h1",
          "Second stage, core suite: mean macro-F1 per cell (six runs, primary view) of Arm~A and Arm A-Single, run "
          "interleaved in one batch, and of the main runs of Arm~A from October. $\\Delta$: A minus A-Single, and A minus "
          "A (Oct.). A superscript gives the passed runs of a cell with fewer than six. \\emph{Signed-rank $p$}: exact "
          "two-sided Wilcoxon signed-rank test over the 13 datasets. "
          # Review 23.11. (R9): Rohwert breast_cancer A neu - A Okt. = +0.00486 (derived/h1.json diff_A_new_minus_A_old);
          # Fussnote statt vier Nachkommastellen, damit check_tables.py ohne neue Zahl im Faktenblatt durchlaeuft.
          + (f"On breast cancer, the second $\\Delta$ is shown rounded to {n(dr['breast_cancer'], sign=True)}; "
             "unrounded, it lies below 0.005. " if 0.0045 <= abs(dr["breast_cancer"]) < 0.005 else "") + ARMS_REF,
          "6.10", "derived/h1.json cell_means_f1_measured, diff_A_new_minus_single, diff_A_new_minus_A_old, failed_runs, "
          "analysis.core", "lrrrrr", [row("Dataset", "A", "A-Single", r"$\Delta$", "A (Oct.)", r"$\Delta$")], rows,
          short="Second stage: F1 per cell, A and A-Single")


def write_registry():
    lines = ["# Tabellen für Kap. 5 und 6", "",
             "> Generiert von `thesis/scripts/make_tables.py` aus `thesis/facts/derived/*.json` (dieselben Werte wie",
             "> `thesis/facts/kap5.md` / `kap6.md`). Einbinden mit `\\input{tables/<datei ohne .tex>}`. Jede Datei",
             "> enthält die vollständige `table`-Umgebung mit `\\caption` und `\\label`. Captions sind Entwürfe.", "",
             "| Datei | Label | Captionentwurf | Abschnitt | Datenquelle |", "|---|---|---|---|---|"]
    for f, lab, cap, sec, src in REGISTRY:
        c = cap.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{f}` | `{lab}` | {c} | {sec} | `{src}` |")
    with open(os.path.join(OUT, "TABLES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  -> thesis/tables/TABLES.md")


def main():
    tab_batches()
    tab_calibration()
    tab_contamination()
    tab_passrates()
    tab_f1cells()
    tab_ranktests()
    tab_pairs()
    tab_runsd()
    tab_terciles()
    tab_postcutoff()
    tab_postcutoff_tests()
    tab_phases()
    tab_leakage()
    tab_cost()
    tab_expectations()
    tab_h1()
    write_registry()


if __name__ == "__main__":
    main()
