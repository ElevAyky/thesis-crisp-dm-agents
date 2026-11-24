# thesis/scripts/make_appendix_tables.py
"""Anhang D (Full Result Tables) -> thesis/appendix/generated/results_*.tex

Liest die Rohdaten (summary.json aller 360 sauberen main-*-Runs: 312 Kern + 48 Post-Cutoff) ueber
collect_facts.load_run (dieselben Felder wie die Faktenbasis von Kap. 5/6) und schreibt:
  results_runs.tex      longtable, alle 360 Runs (tab:D:runs)
  results_primary.tex   Mittel + SD je Zelle, Primaersicht (tab:D:primary)
  results_combined.tex  dasselbe, kombinierter Score (tab:D:combined)
  results_passed.tex    dasselbe, nur bestandene Runs (tab:D:passed)
  results_median.tex    Zellmediane, Median-Sensitivitaet (tab:D:median)
  results_leakage.tex   longtable, alle 78 Leakage-Runs (leak-* 39, spur-* 39; tab:D:leakage)
  results_check.json    Protokoll der Abgleiche

Abgleich (assert, Abbruch bei Abweichung) gegen thesis/facts/derived/*.json (Kap. 6 / kap6.md):
Zellmittel aller drei F1-Sichten (core.json; Post-Cutoff: postcutoff.json + _post_cutoff/analysis),
Run-SD (runsd_core.json, postcutoff.json), Zellmediane (median_core/, median_postcutoff/),
Bestehenszahlen und Fails (core.json, postcutoff.json), Sekundaermass (secondary_core.json,
postcutoff.json), Baselines, Kosten je Arm (costs.json), Leakage je Dataset (leakage.json:
exploited, kappas, success, F1-Mittel, Kosten); zusaetzlich die gerundeten Werte der
Kap.-6-Tabellen tab6_f1cells, tab6_runsd, tab6_postcutoff, tab6_leakage.

Aufruf (Repo-Root):
    uv run python thesis/scripts/make_appendix_tables.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect_facts as cf  # noqa: E402
from make_tables import ARMS, CORE_ORDER, L, NAME, POST_ORDER, n  # noqa: E402

REPO = cf.REPO
DER = os.path.join(REPO, "thesis", "facts", "derived")
TAB = os.path.join(REPO, "thesis", "tables")
OUT = os.path.join(REPO, "thesis", "appendix", "generated")
EPS = 1e-12
ARMS_REF = r"Arms as in \autoref{tab:3:arms}."
VIEWS = {"primary": "f1_measured", "combined": "f1_combined", "passed": "f1_on_success"}
REASON = {"below_threshold": "below baseline", "form_error": "form error", "abstain": "abstain",
          "wrong_target": "wrong target", "spec_mismatch": "spec.\\ mismatch", "gate_failed": "gate failed",
          "budget_exhausted": "budget exhausted", "safety_max_calls": "safety limit"}
CHECKS: dict[str, object] = {}


def j(name):
    return cf.jload(os.path.join(DER, name))


def close(a, b, what, tol=EPS):
    if a is None or b is None:
        assert a is None and b is None, f"{what}: {a!r} != {b!r}"
        return
    assert abs(a - b) <= tol, f"{what}: {a!r} != {b!r}"


def ok(name, value=True):
    CHECKS[name] = value
    print(f"  ok  {name}" + ("" if value is True else f": {value}"))


# =========================================================================== Rohdaten

def load(runs_dir):
    core_man, post_man = cf.manifests()
    runs = {"core": [], "post": []}
    for batch, man in (("core", core_man), ("post", post_man)):
        for d in sorted(os.listdir(runs_dir)):
            p = d.split("-")
            if d.startswith("main-") and len(p) == 4 and p[1] in man:
                runs[batch].append(cf.load_run(runs_dir, d, batch, with_transcripts=False))
    assert len(runs["core"]) == 312 and len(runs["post"]) == 48, {k: len(v) for k, v in runs.items()}
    for batch, order in (("core", CORE_ORDER), ("post", POST_ORDER)):
        for ds in order:
            for a in ARMS:
                reps = sorted(r["rep"] for r in runs[batch] if r["dataset"] == ds and r["arm"] == a)
                assert reps == [1, 2, 3, 4, 5, 6], (ds, a, reps)
    for r in runs["core"] + runs["post"]:
        assert r["outcome"] in ("SUCCESS", "FAIL"), r["run_id"]
        assert r["success"] == (r["outcome"] == "SUCCESS"), r["run_id"]
        assert r["success"] == (r["fail_reason"] is None), r["run_id"]
        assert r["f1"] is not None, f"{r['run_id']}: keine gueltigen Predictions"   # Primaersicht n = 6
        assert r["harness"]["commit"].startswith("14e99e4"), r["run_id"]
    ok("runs_loaded", {"core": 312, "post_cutoff": 48})
    return runs, core_man, post_man


LEAK = {"perfect": ("leak", "9bb6a91"), "spurious": ("spur", "14e99e4")}   # Praefix, Harness-Commit


def load_leakage(runs_dir, core_man):
    """Die 78 sauberen Leakage-Runs (nur Arm D, n = 3 je Dataset und Variante)."""
    leak = {}
    for variant, (prefix, commit) in LEAK.items():
        rs = cf.load_batch(runs_dir, prefix, set(core_man), "leak_" + variant)
        assert len(rs) == 39, (variant, len(rs))
        for ds in CORE_ORDER:
            assert sorted(r["rep"] for r in rs if r["dataset"] == ds) == [1, 2, 3], (variant, ds)
        for r in rs:
            assert r["arm"] == "naked_react" and r["leak_variant"] == variant, r["run_id"]
            assert r["harness"]["commit"].startswith(commit) and r["harness"].get("dirty") is False, r["run_id"]
            assert r["outcome"] in ("SUCCESS", "FAIL") and r["success"] == (r["fail_reason"] is None), r["run_id"]
            assert r["f1"] is not None and r["leak_kappa"] is not None and r["leak_z"] is not None, r["run_id"]
        leak[variant] = rs
    ok("leakage_runs_loaded", {v: len(rs) for v, rs in leak.items()})
    return leak


def cells(runs):
    return {(c["dataset"], c["arm"]): c for c in cf.to_cells(runs)}


def agg(cell, metric, how):
    v = cell[metric]
    if not v:
        return None
    if how == "mean":
        return statistics.fmean(v)
    if how == "median":
        return statistics.median(v)
    return statistics.stdev(v) if len(v) > 1 else None


# =========================================================================== Abgleich

def check_against_derived(runs, runs_dir):
    core_c, post_c = cells(runs["core"]), cells(runs["post"])
    core, post = j("core.json"), j("postcutoff.json")
    rsd = j("runsd_core.json")
    # Zellmittel, drei Sichten — Kern
    for view, metric in VIEWS.items():
        ref = core[f"cell_means_{metric}"]
        for ds in CORE_ORDER:
            for a in ARMS:
                close(agg(core_c[(ds, a)], metric, "mean"), ref[ds][a], f"core {view} {ds} {a}")
    ok("core_cell_means_3_views = derived/core.json")
    # Zellmittel Post-Cutoff: primaer gegen postcutoff.json, alle Sichten gegen _post_cutoff/analysis
    for ds in POST_ORDER:
        for a in ARMS:
            close(agg(post_c[(ds, a)], "f1_measured", "mean"), post["f1_cell_means"][ds][a], f"post primary {ds} {a}")
    ok("post_cell_means_primary = derived/postcutoff.json")
    for sub, cc, order in (("_main", core_c, CORE_ORDER), ("_post_cutoff", post_c, POST_ORDER)):
        ana = cf.jload(os.path.join(runs_dir, sub, "analysis", "analysis.json"))
        for metric in VIEWS.values():
            m = ana["metrics"][metric]
            for i, ds in enumerate(m["datasets"]):
                for k, a in enumerate(ana["arms"]):
                    close(agg(cc[(ds, a)], metric, "mean"), m["matrix"][i][k], f"{sub} {metric} {ds} {a}")
            assert sorted(m["datasets"]) == sorted(order) and not m["dropped"]
    ok("cell_means_3_views = runs/_main + _post_cutoff/analysis/analysis.json")
    # Run-SD (Primaersicht)
    for ds in CORE_ORDER:
        for a in ARMS:
            close(agg(core_c[(ds, a)], "f1_measured", "sd"), rsd["per_cell"][ds][a], f"core sd {ds} {a}")
    for ds in POST_ORDER:
        for a in ARMS:
            close(agg(post_c[(ds, a)], "f1_measured", "sd"), post["run_sd"]["per_cell"][ds][a], f"post sd {ds} {a}")
    ok("run_sd = derived/runsd_core.json + postcutoff.json")
    # Zellmediane, drei Sichten
    for sub, cc in (("median_core", core_c), ("median_postcutoff", post_c)):
        ana = j(os.path.join(sub, "analysis.json"))
        assert ana["agg"] == "median"
        for metric in VIEWS.values():
            m = ana["metrics"][metric]
            for i, ds in enumerate(m["datasets"]):
                for k, a in enumerate(ana["arms"]):
                    close(agg(cc[(ds, a)], metric, "median"), m["matrix"][i][k], f"{sub} {metric} {ds} {a}")
    ok("cell_medians_3_views = derived/median_core + median_postcutoff")
    # Bestehen, Fails, Baselines
    for ref, cc in ((core["level1"]["per_cell"], core_c), (post["level1"]["per_cell"], post_c)):
        for row in ref:
            c = cc[(row["dataset"], row["arm"])]
            assert c["outcomes"].get("SUCCESS", 0) == row["success"] and c["n_measured"] == row["n"], row
    ok("passes_per_cell = derived level1.per_cell")
    fails = {r["run_id"]: r for r in runs["core"] + runs["post"] if not r["success"]}
    ref_fails = {f["run_id"]: f for row in core["level1"]["per_cell"] for f in row["fails"]}
    ref_fails.update({f["run_id"]: f for f in post["fails"]})
    assert set(fails) == set(ref_fails), (set(fails) ^ set(ref_fails))
    for rid, f in ref_fails.items():
        for k in ("f1", "diff", "ci_hi"):
            close(fails[rid][k], f[k], f"fail {rid} {k}")
        assert fails[rid]["fail_reason"] == f["fail_reason"]
    ok("fails = derived (core 1, post 11)", {"core": sum(r["batch"] == "core" for r in fails.values()),
                                              "post_cutoff": sum(r["batch"] == "post" for r in fails.values())})
    for batch, ref, order in (("core", core["baseline_test"], CORE_ORDER), ("post", post["baseline_test"], POST_ORDER)):
        for ds in order:
            vals = {r["baseline_f1"] for r in runs[batch] if r["dataset"] == ds}
            assert len(vals) == 1, (ds, vals)
            close(vals.pop(), ref[ds], f"baseline {ds}")
    ok("baseline_test = derived (one value per dataset)")
    # Sekundaermass (ci_lo > 0)
    for batch, ref in (("core", j("secondary_core.json")["per_arm"]), ("post", post["secondary"]["per_arm"])):
        for a in ARMS:
            k = sum(r["sig_better"] for r in runs[batch] if r["arm"] == a)
            assert k == ref[a]["sig_better"], (batch, a, k, ref[a]["sig_better"])
    ok("significantly_better_counts = derived secondary")
    # Kosten
    costs = j("costs.json")
    for batch, key in (("core", "core"), ("post", "post_cutoff")):
        for a in ARMS:
            close(sum(r["cost"] for r in runs[batch] if r["arm"] == a), costs[key]["per_arm"][a]["cost_total"],
                  f"cost {batch} {a}", tol=1e-6)
        close(sum(r["cost"] for r in runs[batch]), costs[key]["total"], f"cost {batch}", tol=1e-6)
    ok("cost_per_arm = derived/costs.json",
       {k: round(costs[k]["total"], 2) for k in ("core", "post_cutoff")})
    return core_c, post_c


def num_cells(tex_row: str):
    """Zahlen einer Tabellenzeile aus Kap. 6 (ohne \\textbf, $-$ -> -, $+$ -> +)."""
    s = tex_row.replace(r"\textbf{", "").replace("}", "").replace("$-$", "-").replace("$+$", "+")
    return [c.strip() for c in s.rstrip("\\ ").split("&")]


def check_against_kap6_tables(core_c, post_c):
    """Gerundete Werte, wie sie in Kap. 6 gedruckt sind."""
    with open(os.path.join(TAB, "tab6_f1cells.tex"), encoding="utf-8") as f:
        f1 = f.read()
    with open(os.path.join(TAB, "tab6_runsd.tex"), encoding="utf-8") as f:
        rsd = f.read()
    for ds in CORE_ORDER:
        row = next(line for line in f1.splitlines() if line.strip().startswith(NAME[ds] + " &"))
        got = num_cells(row)[-4:]
        exp = [f"{agg(core_c[(ds, a)], 'f1_measured', 'mean'):.3f}" for a in ARMS]
        assert got == exp, (ds, got, exp)
        row = next(line for line in rsd.splitlines() if line.strip().startswith(NAME[ds] + " &"))
        got = num_cells(row)[-4:]
        exp = [f"{agg(core_c[(ds, a)], 'f1_measured', 'sd'):.4f}" for a in ARMS]
        assert got == exp, (ds, got, exp)
    with open(os.path.join(TAB, "tab6_postcutoff.tex"), encoding="utf-8") as f:
        pc = f.read().splitlines()
    for ds in POST_ORDER:
        i = next(k for k, line in enumerate(pc) if "\\emph{" + NAME[ds] + "}" in line)
        mean_row = next(line for line in pc[i:] if "Mean F1 &" in line)
        sd_row = next(line for line in pc[i:] if "Run SD &" in line)
        assert num_cells(mean_row)[-4:] == [f"{agg(post_c[(ds, a)], 'f1_measured', 'mean'):.3f}" for a in ARMS], ds
        assert num_cells(sd_row)[-4:] == [f"{agg(post_c[(ds, a)], 'f1_measured', 'sd'):.4f}" for a in ARMS], ds
    ok("rounded values = thesis/tables/tab6_f1cells, tab6_runsd, tab6_postcutoff")


def check_leakage(leak):
    """Leakage-Runs gegen derived/leakage.json und die gerundeten Werte von tab6_leakage."""
    ref = j("leakage.json")
    for variant, rs in leak.items():
        R = ref[variant]
        for ds in CORE_ORDER:
            x = sorted((r for r in rs if r["dataset"] == ds), key=lambda r: r["rep"])
            P = R["per_set"][ds]
            assert sum(bool(r["leak_exploited"]) for r in x) == P["exploited"], (variant, ds)
            assert sum(r["success"] for r in x) == P["success"], (variant, ds)
            assert all(r["leak_column"] == P["column"] for r in x), (variant, ds)
            for r, k in zip(x, P["kappas"]):
                close(r["leak_kappa"], k, f"leak {variant} {ds} kappa r{r['rep']}")
            close(statistics.fmean(r["f1"] for r in x), P["f1_mean"], f"leak {variant} {ds} f1", tol=1e-12)
            close(sum(r["cost"] for r in x), P["cost"], f"leak {variant} {ds} cost", tol=1e-9)
        assert sum(bool(r["leak_exploited"]) for r in rs) == R["exploited"]
        assert sum(r["success"] for r in rs) == R["success"]
        close(sum(r["cost"] for r in rs), R["cost"], f"leak {variant} cost", tol=1e-9)
        assert {r["run_id"] for r in rs if r["leak_exploited"]} == {e["run_id"] for e in R["exploited_detail"]}
    ok("leakage = derived/leakage.json", {v: {"exploited": ref[v]["exploited"], "passed": ref[v]["success"],
                                               "cost": round(ref[v]["cost"], 2)} for v in leak})
    with open(os.path.join(TAB, "tab6_leakage.tex"), encoding="utf-8") as f:
        tab = f.read().splitlines()
    for ds in CORE_ORDER:
        row = next(line for line in tab if line.strip().startswith(NAME[ds] + " &"))
        exp = []
        for variant in ("perfect", "spurious"):
            x = [r for r in leak[variant] if r["dataset"] == ds]
            exp += [f"{sum(bool(r['leak_exploited']) for r in x)}/3",
                    n(statistics.fmean(r["leak_kappa"] for r in x), 2).replace("$-$", "-"),
                    f"{sum(r['success'] for r in x)}/3"]
        assert num_cells(row)[1:] == exp, (ds, num_cells(row)[1:], exp)
    ok("rounded values = thesis/tables/tab6_leakage")


# =========================================================================== LaTeX

HEADER = ("%!TEX root = ../../thesis.tex\n"
          "% generiert von thesis/scripts/make_appendix_tables.py — nicht von Hand aendern\n")


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".tex")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + body)
    print(f"  -> {os.path.relpath(path, REPO)}")


def ci_bound(x):
    """Wie n(x, sign=True), aber das Vorzeichen bleibt sichtbar, wenn |x| auf 0.000 rundet: Bestehen
    (obere Grenze >= 0) und 'signifikant besser' (untere Grenze > 0) haengen am Vorzeichen
    (z. B. leak-cnae_9-naked_react-r3: ci_hi = -0.00028, sonst als 0.000 gedruckt)."""
    s = n(x, sign=True)
    if x is not None and x != 0 and s == "0.000":
        return ("$-$" if x < 0 else "$+$") + s
    return s


def ci(lo, hi):
    return f"[{ci_bound(lo)}, {ci_bound(hi)}]"


def run_table(runs):
    cols = 9
    head = (r"Dataset & Arm & Rep. & Outcome & F1 & Baseline & $\Delta$ & 95\,\% CI of $\Delta$ & Cost (\$) \\")
    cap = ("All 360 clean runs: 312 of the core suite (datasets in tercile order) and 48 of the post-cutoff "
           "control. \\emph{Outcome}: \\emph{pass} or the failure reason (\\autoref{sec:3:Variables}). "
           "\\emph{F1}: macro-F1 of the finally delivered predictions on the test rows (primary measure). "
           "\\emph{Baseline}: F1 of the reference baseline on the same rows. $\\Delta$ and \\emph{95\\,\\% CI}: "
           "difference to the baseline and its paired bootstrap interval (\\autoref{sec:3:SuccessCriterion}); "
           "a run passes if the upper bound is not negative and counts as significantly better if the lower "
           "bound is positive. \\emph{Cost}: list-price cost of the run, including the judges of Arm~B "
           "(\\autoref{sec:6:Cost}). All values rounded independently; a sign before a bound of 0.000 marks a "
           "bound that is negative or positive but rounds to zero. " + ARMS_REF)
    lines = [r"{\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{longtable}{lccl rrrcr}",
             rf"\caption[All clean runs]{{{cap}}}\label{{tab:D:runs}}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             rf"\multicolumn{{{cols}}}{{l}}{{\emph{{\tablename~\thetable{{}} continued}}}}\\",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule", rf"\multicolumn{{{cols}}}{{r}}{{\emph{{continued on next page}}}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    n_rows = 0
    for batch, order, title in (("core", CORE_ORDER, "Core suite"), ("post", POST_ORDER, "Post-cutoff control")):
        if batch == "post":
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{cols}}}{{l}}{{\textbf{{{title}}}}}\\*")  # kein Seitenumbruch nach der Gruppenzeile
        for k, ds in enumerate(order):
            lines.append(r"\midrule" if k else r"\addlinespace[2pt]")
            for ai, a in enumerate(ARMS):
                if ai:
                    lines.append(r"\addlinespace[2pt]")
                for r in sorted((r for r in runs[batch] if r["dataset"] == ds and r["arm"] == a),
                                key=lambda r: r["rep"]):
                    outcome = "pass" if r["success"] else r"\emph{" + REASON[r["fail_reason"]] + "}"
                    lines.append(" & ".join([NAME[ds], L[a], str(r["rep"]), outcome, n(r["f1"]), n(r["baseline_f1"]),
                                             n(r["diff"], sign=True), ci(r["ci_lo"], r["ci_hi"]),
                                             f"{r['cost']:.3f}"]) + r" \\")
                    n_rows += 1
    lines += [r"\end{longtable}", r"}"]
    assert n_rows == 360
    write("results_runs", "\n".join(lines) + "\n")
    return n_rows


def leakage_table(leak):
    cols = 10
    head = (r"Dataset & Rep. & Outcome & F1 & $\Delta$ & 95\,\% CI of $\Delta$ & $\kappa$ & $z$ & Exploited & "
            r"Cost (\$) \\")
    cap = ("All 78 runs of the leakage experiment (\\autoref{sec:6:RobustnessUnsolvableAndLeakyTa}), Arm~D only, "
           "three runs per dataset and variant; datasets in tercile order. The perfect batch ran with harness "
           "commit \\texttt{9bb6a91}, the spurious batch with \\texttt{14e99e4} (\\autoref{tab:B:versions}). "
           "\\emph{Outcome}, \\emph{F1}, $\\Delta$, \\emph{95\\,\\% CI} and \\emph{Cost} as in \\autoref{tab:D:runs}; "
           "the baseline is that of the core suite (\\autoref{tab:D:primary}). $\\kappa$: Cohen's $\\kappa$ "
           "between the predictions on the test rows and the class that the leak column indicates in these rows; "
           "$z$: the excess of this agreement over chance in standard errors. \\emph{Exploited}: $z \\geq 3.09$ and "
           "$\\kappa \\geq 0.2$ (\\autoref{sec:3:LeakageExperiments}). All values rounded independently; signs "
           "of bounds that round to 0.000 as in \\autoref{tab:D:runs}.")
    lines = [r"{\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
             r"\begin{longtable}{lcl rrc rrcr}",
             rf"\caption[All leakage runs]{{{cap}}}\label{{tab:D:leakage}}\\",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             rf"\multicolumn{{{cols}}}{{l}}{{\emph{{\tablename~\thetable{{}} continued}}}}\\",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule", rf"\multicolumn{{{cols}}}{{r}}{{\emph{{continued on next page}}}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    n_rows = 0
    for variant, title in (("perfect", "Perfect leak"), ("spurious", "Spurious leak")):
        if variant == "spurious":
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{cols}}}{{l}}{{\textbf{{{title}}}}}\\*")  # kein Seitenumbruch nach der Gruppenzeile (Sichtpruefung 24.11.)
        for k, ds in enumerate(CORE_ORDER):
            lines.append(r"\midrule" if k else r"\addlinespace[2pt]")
            for r in sorted((r for r in leak[variant] if r["dataset"] == ds), key=lambda r: r["rep"]):
                outcome = "pass" if r["success"] else r"\emph{" + REASON[r["fail_reason"]] + "}"
                lines.append(" & ".join([NAME[ds], str(r["rep"]), outcome, n(r["f1"]), n(r["diff"], sign=True),
                                         ci(r["ci_lo"], r["ci_hi"]), n(r["leak_kappa"], 2), n(r["leak_z"], 1),
                                         "yes" if r["leak_exploited"] else "no", f"{r['cost']:.3f}"]) + r" \\")
                n_rows += 1
    lines += [r"\end{longtable}", r"}"]
    assert n_rows == 78
    write("results_leakage", "\n".join(lines) + "\n")
    return n_rows


def cell_table(name, label, short, caption, core_c, post_c, metric, *, show_n=False, base_core=None,
               base_post=None):
    head1 = (r"Dataset & Base. & \multicolumn{2}{c}{A} & \multicolumn{2}{c}{B} & \multicolumn{2}{c}{C} & "
             r"\multicolumn{2}{c}{D} \\")
    cmid = r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}"
    head2 = r" & & Mean & SD & Mean & SD & Mean & SD & Mean & SD \\"
    lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
             rf"  \caption[{short}]{{{caption}}}", rf"  \label{{{label}}}",
             r"  \begin{tabular}{lrrrrrrrrr}", r"    \toprule", "    " + head1, "    " + cmid, "    " + head2,
             r"    \midrule"]
    terc = j("terciles.json")["assignment"]
    rows = 0
    for batch, cc, order, base in (("core", core_c, CORE_ORDER, base_core), ("post", post_c, POST_ORDER, base_post)):
        if batch == "post":
            lines += [r"    \midrule", r"    \multicolumn{10}{l}{\emph{Post-cutoff control}} \\"]
        prev = None
        for ds in order:
            if batch == "core" and prev is not None and terc[ds] != terc[prev]:
                lines.append(r"    \addlinespace")
            prev = ds
            cells_txt = []
            for a in ARMS:
                c = cc[(ds, a)]
                m, s = agg(c, metric, "mean"), agg(c, metric, "sd")
                k = len(c[metric])
                mt = n(m)
                if show_n and k < 6:
                    mt += rf"$^{{{k}}}$"
                cells_txt += [mt, n(s, 4)]
            lines.append("    " + " & ".join([NAME[ds], n(base[ds])] + cells_txt) + r" \\")
            rows += 1
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write(name, "\n".join(lines) + "\n")
    return rows


def median_table(core_c, post_c, base_core, base_post):
    lines = [r"\begin{table}[htbp]", r"  \centering", r"  \small"]
    diffs = []
    body = []
    terc = j("terciles.json")["assignment"]
    for batch, cc, order, base in (("core", core_c, CORE_ORDER, base_core), ("post", post_c, POST_ORDER, base_post)):
        if batch == "post":
            body += [r"    \midrule", r"    \multicolumn{6}{l}{\emph{Post-cutoff control}} \\"]
        prev = None
        for ds in order:
            if batch == "core" and prev is not None and terc[ds] != terc[prev]:
                body.append(r"    \addlinespace")
            prev = ds
            vals = []
            for a in ARMS:
                med = {v: agg(cc[(ds, a)], m, "median") for v, m in VIEWS.items()}
                vals.append(n(med["primary"]))
                for v in ("combined", "passed"):
                    # auf Druckgenauigkeit vergleichen: ungerundet weicht z. B. student health x A,
                    # nur bestandene Runs, um 1e-6 ab und erschiene sonst mit demselben Wert 0.903
                    if med[v] is None or n(med[v]) != n(med["primary"]):
                        diffs.append((NAME[ds], L[a], v, med[v]))
            body.append("    " + " & ".join([NAME[ds], n(base[ds])] + vals) + r" \\")
    view_name = {"combined": "combined score", "passed": "passed runs only"}
    if diffs:
        listing = "; ".join(f"{ds}, {a}, {view_name[v]}: {n(x)}" for ds, a, v, x in diffs)
        note = (" At the printed precision, the medians of the two other views differ from these only in the "
                "following cells: " + listing + ".")
    else:
        note = " The two other views give the same medians in every cell."
    cap = ("Median macro-F1 per cell (six runs, primary view), the sensitivity analysis of "
           "\\autoref{sec:6:LevelModelQuality}; \\emph{Base.}: baseline F1 on the same test rows." + note
           + " " + ARMS_REF)
    lines += [rf"  \caption[Median F1 per cell]{{{cap}}}", r"  \label{tab:D:median}",
              r"  \begin{tabular}{lrrrrr}", r"    \toprule", r"    Dataset & Base. & A & B & C & D \\",
              r"    \midrule"] + body + [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    write("results_median", "\n".join(lines) + "\n")
    return len(CORE_ORDER) + len(POST_ORDER), diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(REPO), "thesis-crisp-dm-runs"))
    args = ap.parse_args()
    runs_dir = os.path.abspath(args.runs_dir)
    print("lade Runs ...")
    runs, core_man, post_man = load(runs_dir)
    print("Abgleich ...")
    core_c, post_c = check_against_derived(runs, runs_dir)
    check_against_kap6_tables(core_c, post_c)
    leak = load_leakage(runs_dir, core_man)
    check_leakage(leak)
    base_core = j("core.json")["baseline_test"]
    base_post = j("postcutoff.json")["baseline_test"]
    for rs in leak.values():   # Leakage-Runs messen gegen dieselbe Baseline wie der Kern
        for r in rs:
            close(r["baseline_f1"], base_core[r["dataset"]], f"leak baseline {r['run_id']}")
    ok("leakage_baseline = core baseline_test")

    rows = {"runs": run_table(runs), "leakage": leakage_table(leak)}
    common = ("Datasets of the core suite grouped by difficulty tercile (easy, medium, hard), then the "
              "post-cutoff control. \\emph{Base.}: baseline F1 on the same test rows. SD: sample standard "
              "deviation over the runs of the cell. ")
    rows["primary"] = cell_table(
        "results_primary", "tab:D:primary", "Mean and SD of F1 per cell, primary view",
        "Mean and run-to-run standard deviation of macro-F1 per cell, primary view: all six runs of every "
        "cell delivered valid predictions (\\autoref{sec:3:Variables}). The means of the core suite are "
        "those of \\autoref{tab:6:f1cells}, the SDs those of \\autoref{tab:6:runsd}; for the post-cutoff "
        "control see \\autoref{tab:6:postcutoff}. " + common + ARMS_REF,
        core_c, post_c, "f1_measured", base_core=base_core, base_post=base_post)
    rows["combined"] = cell_table(
        "results_combined", "tab:D:combined", "Mean and SD of F1 per cell, combined score",
        "Mean and standard deviation of the combined score per cell: the F1 of passed runs, zero for every "
        "run that did not pass (sensitivity view of \\autoref{sec:6:LevelModelQuality}). " + common + ARMS_REF,
        core_c, post_c, "f1_combined", base_core=base_core, base_post=base_post)
    rows["passed"] = cell_table(
        "results_passed", "tab:D:passed", "Mean and SD of F1 per cell, passed runs only",
        "Mean and standard deviation of macro-F1 per cell over the passed runs only (sensitivity view of "
        "\\autoref{sec:6:LevelModelQuality}). A superscript gives the number of passed runs where it is "
        "below six. " + common + ARMS_REF,
        core_c, post_c, "f1_on_success", show_n=True, base_core=base_core, base_post=base_post)
    rows["median"], diffs = median_table(core_c, post_c, base_core, base_post)
    CHECKS["rows"] = rows
    CHECKS["median_view_differences"] = [list(map(str, d)) for d in diffs]
    with open(os.path.join(OUT, "results_check.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(CHECKS, f, indent=1)
    print("  -> thesis/appendix/generated/results_check.json")
    print("Zeilen:", rows)


if __name__ == "__main__":
    main()
