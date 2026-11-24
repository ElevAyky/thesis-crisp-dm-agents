#!/usr/bin/env python
"""Abbildungen + Tabelle fuer Kap. 7.2 (Prozess-Kodierung der Agenten-Laeufe).

Liest thesis/facts/derived/process_coding.json (erzeugt von thesis/scripts/code_process.py
aus den Transkripten der sauberen Runs main-*, Rang 1) sowie fuer die Tabellen-Fussnote
thesis/facts/derived/coding_agreement.json (coding_agreement.py) und schreibt
  - thesis/figs/fig7_process.pdf   Anteil der Runs je Code, C vs. D (A/B als Referenz), Kern
  - thesis/figs/fig7_sequence.pdf  (a) Phasenfolgen C vs. D, (b) Modellfamilien je Run, A-D, Kern
  - thesis/tables/tab7_process.tex Raten je Code und Arm, Kern + Post-Cutoff, Fisher C vs. D (Holm)
  - thesis/tables/tab7_student_health.tex  student_health: Weg zum Endmodell je Arm (Kap. 7.3)
Stil, Arm-Farben, Marker, Breiten und Schrift aus thesis/scripts/make_figures.py (Kap. 5/6).

Aufruf aus dem Repo-Root:
    uv run python thesis/scripts/code_process.py          # zuerst: Kodierung aktualisieren
    uv run python thesis/scripts/make_figures_kap7.py [--preview-dir <Ordner fuer PNG>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

import make_figures as mf  # noqa: E402  (gleicher Ordner; Stil + Konstanten)
from make_figures import (CM, COLOR, FULL, GRID, INK, INK2, LETTER, MARKER, MID,  # noqa: E402
                          MUTED, SURFACE, arm_label, mark, ms, setup_style)

REPO = Path(__file__).resolve().parents[2]
CODING = REPO / "thesis" / "facts" / "derived" / "process_coding.json"
AGREEMENT = REPO / "thesis" / "facts" / "derived" / "coding_agreement.json"
ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
BY_LETTER = {v: k for k, v in LETTER.items()}

# A/B: vom Kontrakt verlangt (keine freie Wahl) bzw. vom Kontrakt erfragt (Kap. 7, codebook.md Paragraf 7)
ENFORCED_AB = {"eval_baseline",        # Modeling-Kontrakt: dummy_score; Model-Gate: val_score > dummy_score
               "model_test_design"}    # Modeling-Kontrakt: val_score auf Holdout oder CV (prompts.MODEL)
PROMPTED_AB = {"du_leakage"}           # BU: critical_columns "suspected ID or leakage columns"; DU: excluded_columns
# Codes mit eingeschraenkter Lesart (kap7.md F7-912, F7-914, F7-916 bis F7-919, F7-922, F7-923, F7-927,
# F7-928): Fussnotenzeichen. du_class_dist und prep_explicit: Untergrenze um <= 1 Run je Arm (Pruefbefund R7-02)
CAVEAT = {"bu_business_context": "a", "bu_metric_rationale": "b", "du_class_dist": "b",
          "du_quality_ext": "b", "prep_explicit": "b", "eval_business": "b", "doc_crisp_sections": "b",
          "prep_construct": "c", "eval_review_next": "c", "monolith": "d"}

# Zeilen der Abbildung/Tabelle: (Code, englisches Label, CRISP-1.0-Aufgabe)
GROUPS = [
    ("Business understanding", [
        ("bu_explicit", "Target and metric stated before modeling", "1.3"),
        ("bu_metric_rationale", "Metric choice justified", "1.3"),
        ("bu_success_criterion", "Success criterion stated before modeling", "1.1, 1.3"),
        ("bu_business_context", "Business framing (objectives, costs)", "1.1, 1.2"),
    ]),
    ("Data understanding", [
        ("du_inspect_before_model", "Data inspected before modeling", "2.2"),
        ("du_class_dist", "Class distribution checked", "2.3"),
        ("du_missing", "Missing values checked", "2.4"),
        ("du_quality_ext", "Duplicates, constants or IDs checked", "2.4"),
        ("du_stats", "Descriptive statistics", "2.2, 2.3"),
        ("du_relations", "Attribute relations explored", "2.3"),
        ("du_leakage", "Leakage or ID columns considered", "2.4"),
    ]),
    ("Data preparation", [
        ("prep_explicit", "Explicit preparation steps", "3.1--3.5"),
        ("prep_select", "Named columns excluded", "3.1"),
        ("prep_construct", "Derived attributes constructed", "3.3"),
    ]),
    ("Modeling", [
        ("model_multi_family", "Two or more model families", "4.1, 4.4"),
        ("model_test_design", "Internal validation (CV or hold-out)", "4.2"),
        ("model_cv", "Cross-validation", "4.2"),
        ("model_tuning", "Hyperparameter search or revision", "4.3, 4.4"),
        ("model_assess_detail", "Per-class assessment", "4.4"),
    ]),
    ("Evaluation", [
        ("eval_baseline", "Compared with a baseline", "5.1"),
        ("eval_business", "Assessed against the objective", "5.1"),
        ("eval_review_next", "Process review or next steps", "5.2, 5.3"),
    ]),
    ("Reports", [
        ("doc_notes", "Notes file written", "outputs"),
        ("doc_crisp_sections", "Notes structured by CRISP-DM phases", "outputs"),
    ]),
]
FLOW = [
    ("canonical_order", "Phases in canonical order"),
    ("implicit_crisp", "All five phases, in canonical order"),
    ("backjump_any", "Return to an earlier phase"),
    ("monolith", "One script, inspection to prediction"),
]
PHASE_LABEL = {"DU": "DU", "PREP": "Prep", "MODEL": "Model", "EVAL": "Eval"}


def load() -> dict:
    with open(CODING, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name: str, out_dir: Path, preview_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None,
                                                    "Creator": "thesis/scripts/make_figures_kap7.py"})
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  -> {name}.pdf")


def rate(S: dict, code: str, letter: str):
    v = S["rates"].get(code, {}).get(letter)
    if letter in "AB" and code in ENFORCED_AB and v is not None:
        return "enforced"
    if v is None or v == "enforced":
        return v
    return v[0] / v[1], v[0], v[1]


def pct(x, _=None) -> str:
    return f"{x:.0%}".replace("%", " %")


# ======================================================================================

def fig7_process(d: dict, out: Path, prev: Path | None) -> None:
    S = d["summary"]["scopes"]["core"]
    rows = []                        # (y, code, label) ; Gruppen-Ueberschriften separat
    heads = []
    y = 0.0
    for gname, codes in GROUPS:
        heads.append((y, gname))
        y += 0.85
        for code, label, _ in codes:
            rows.append((y, code, label))
            y += 1.0
        y += 0.35
    height = (y * 0.33 + 1.6) * CM
    fig, ax = plt.subplots(figsize=(FULL, height), layout="constrained")
    for yy, code, label in rows:
        c, dd = rate(S, code, "C"), rate(S, code, "D")
        ax.hlines(yy, min(c[0], dd[0]), max(c[0], dd[0]), color=GRID, lw=2.4, zorder=1)
        enforced = False
        for letter in ("A", "B"):
            r = rate(S, code, letter)
            if isinstance(r, tuple):
                arm = BY_LETTER[letter]
                ax.plot(r[0], yy, ls="none", marker=MARKER[arm], ms=ms(arm, 4.0), mfc=SURFACE,
                        mec=COLOR[arm], mew=0.8, zorder=2)
            elif r == "enforced":
                enforced = True
        if enforced:
            ax.text(1.035, yy, "A, B: enforced", va="center", ha="left", fontsize=6.4, color=MUTED)
        elif code in PROMPTED_AB:
            ax.text(1.035, yy, "A, B: prompted", va="center", ha="left", fontsize=6.4, color=MUTED)
        dy = 0.17 if abs(c[0] - dd[0]) < 0.025 else 0.0     # gleiche Werte: Marker nicht verdecken
        mark(ax, c[0], yy - dy, "crisp_react", base=5.0, zorder=4)
        mark(ax, dd[0], yy + dy, "naked_react", base=5.0, zorder=3)
        holm = S["fisher_C_vs_D"][code]["p_holm"]
        star = " *" if holm < 0.05 else ""
        dagger = " †" if code in CAVEAT else ""
        ax.text(-0.02, yy, label + dagger + star, va="center", ha="right", fontsize=7.2, color=INK,
                transform=ax.get_yaxis_transform())
        print(f"     {code:24s} C {c[1]}/{c[2]}  D {dd[1]}/{dd[2]}  p_holm {holm:.4f}")
    for yy, gname in heads:
        ax.text(-0.02, yy, gname, va="center", ha="right", fontsize=7.4, color=INK, weight="bold",
                transform=ax.get_yaxis_transform())
    ax.set_ylim(y - 0.3, -0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.02, 1.16)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pct(x) if 0 <= x <= 1 else ""))
    ax.spines["bottom"].set_bounds(0, 1)
    for x in (0.25, 0.5, 0.75, 1.0):
        ax.axvline(x, color=MID, lw=0.6, zorder=0)
    ax.set_xlabel("Share of runs (core suite, 78 runs per arm)")
    handles = [Line2D([], [], ls="none", marker=MARKER[a], ms=ms(a, 5.0), mfc=COLOR[a], mec=SURFACE,
                      mew=0.6, label=arm_label(a)) for a in ("crisp_react", "naked_react")]
    handles += [Line2D([], [], ls="none", marker=MARKER[a], ms=ms(a, 4.0), mfc=SURFACE, mec=COLOR[a],
                       mew=0.8, label=arm_label(a) + " (reference)") for a in ("hard_fsm", "soft_fsm")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=7.0)
    save(fig, "fig7_process", out, prev)


def _seq_label(seq: str) -> str:
    return " → ".join(PHASE_LABEL[p] for p in seq.split())


def fig7_sequence(d: dict, out: Path, prev: Path | None) -> None:
    S = d["summary"]["scopes"]["core"]
    seqC = Counter(S["numeric"]["C"]["phase_sequence"])
    seqD = Counter(S["numeric"]["D"]["phase_sequence"])
    total = seqC + seqD
    top = [s for s, _ in total.most_common() if total[s] >= 4][:6]
    other_c = sum(v for s, v in seqC.items() if s not in top)
    other_d = sum(v for s, v in seqD.items() if s not in top)
    labels = [_seq_label(s) for s in top] + ["other sequences"]
    vc = [seqC[s] for s in top] + [other_c]
    vd = [seqD[s] for s in top] + [other_d]
    for lab, a, b in zip(labels, vc, vd):
        print(f"     {lab:45s} C {a:2d}  D {b:2d}")

    fig = plt.figure(figsize=(FULL, 6.6 * CM), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    h = 0.36
    for i, (a, b) in enumerate(zip(vc, vd)):
        for val, dy, arm in ((a, -h / 2, "crisp_react"), (b, h / 2, "naked_react")):
            ax.barh(i + dy, val, height=h - 0.04, color=COLOR[arm], edgecolor=SURFACE, lw=0.6)
            ax.text(val + 0.8, i + dy, str(val), va="center", ha="left", fontsize=6.6, color=INK2)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.0)
    ax.set_ylim(len(labels) - 0.45, -0.55)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 62)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("Runs (of 78 per arm)")
    ax.set_title("(a) Phase sequence, including returns")
    ax.legend(handles=[Line2D([], [], ls="none", marker="s", ms=5.5, mfc=COLOR[a], mec=COLOR[a],
                              label=arm_label(a)) for a in ("crisp_react", "naked_react")],
              loc="lower right", fontsize=6.8)

    # (b) Modellfamilien je Run: gestapelte Anteile 1 / 2 / 3 / >= 4, Grau-Rampe (ordinal)
    ax2 = fig.add_subplot(gs[0, 1])
    bins = [("1", lambda k: k == 1), ("2", lambda k: k == 2), ("3", lambda k: k == 3), ("≥ 4", lambda k: k >= 4)]
    greys = ["#e6e5e0", "#bcbab3", "#8e8c85", "#5f5d58"]
    inks = [INK, INK, SURFACE, SURFACE]
    for i, arm in enumerate(ARMS):
        dist = {int(k): v for k, v in S["numeric"][LETTER[arm]]["n_families_dist"].items()}
        n = sum(dist.values())
        left = 0.0
        parts = []
        for (lab, f), g, ink in zip(bins, greys, inks):
            k = sum(v for kk, v in dist.items() if f(kk))
            share = k / n
            if share > 0:
                ax2.barh(i, share, left=left, height=0.62, color=g, edgecolor=SURFACE, lw=1.0)
                if share >= 0.05:
                    ax2.text(left + share / 2, i, str(k), va="center", ha="center", fontsize=6.2, color=ink)
            parts.append(f"{lab}:{k}")
            left += share
        print(f"     families {LETTER[arm]}: " + " ".join(parts) + f"  (n={n})")
    ax2.set_yticks(range(len(ARMS)), [arm_label(a) for a in ARMS], fontsize=7.0)
    for i, arm in enumerate(ARMS):
        ax2.plot(-0.035, i, ls="none", marker=MARKER[arm], ms=ms(arm, 4.4), mfc=COLOR[arm], mec=SURFACE,
                 mew=0.6, clip_on=False, transform=ax2.get_yaxis_transform())
    ax2.tick_params(axis="y", length=0, pad=9)
    ax2.set_ylim(len(ARMS) - 0.5, -0.5)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.xaxis.set_major_formatter(FuncFormatter(pct))
    ax2.set_xlabel("Share of runs (counts in bars)")
    ax2.set_title("(b) Model families per run", pad=15)
    ax2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=g, label=lab) for (lab, _), g in zip(bins, greys)],
               loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, fontsize=6.6,
               handlelength=1.0, columnspacing=0.9, borderaxespad=0.1)
    save(fig, "fig7_sequence", out, prev)


# ======================================================================================

def _cell(S, code, letter) -> str:
    r = S["rates"].get(code, {}).get(letter)
    if r is None:
        return "--"
    if r == "enforced" or (letter in "AB" and code in ENFORCED_AB):
        return "enf."
    if letter in "AB" and code in PROMPTED_AB:
        return f"\\emph{{{r[0]}}}"
    return str(r[0])


def _lab(code: str, label: str) -> str:
    return label + (f"\\textsuperscript{{{CAVEAT[code]}}}" if code in CAVEAT else "")


def _p(x: float) -> str:
    if x < 0.001:
        return "$<$0.001"
    return f"{x:.3f}"


def write_table(d: dict, out_dir: Path) -> None:
    C, P = d["summary"]["scopes"]["core"], d["summary"]["scopes"]["post_cutoff"]
    nC, nP = C["n"], P["n"]
    assert all(nC[x] == 78 for x in "ABCD") and all(nP[x] == 12 for x in "ABCD"), (nC, nP)
    agr = json.loads(AGREEMENT.read_text(encoding="utf-8"))["full_set_checks"]["bu_business_context"]
    L = []
    L.append("%!TEX root = ../thesis.tex")
    L.append("% generiert von thesis/scripts/make_figures_kap7.py aus thesis/facts/derived/process_coding.json"
             " und coding_agreement.json — nicht von Hand aendern")
    L.append("\\begin{table}[htbp]")
    L.append("  \\centering")
    # \footnotesize statt \small: mit \small war die Tabelle 5.87 pt zu hoch fuer die Seite (Paket-3-Build)
    L.append("  \\footnotesize")
    L.append("  \\caption[Process codes of the agent runs]{Process codes of the agent runs: number of runs in which a code "
             "was observed, core suite (78 runs per arm) and post-cutoff control (12 runs per arm). Codes follow "
             "the generic tasks of CRISP-DM~1.0 (column \\emph{Task}) and are assigned by deterministic rules "
             "from the session transcripts. Arms~A and B are coded only for the choices that the state machine "
             "leaves to the worker (--: not coded); \\emph{enf.}: required by the Modeling contract; italic: asked "
             "for by the Business- and Data-Understanding contracts. "
             "$p_\\mathrm{Holm}$: Fisher's exact test of C against D on the core suite, Holm-adjusted over the "
             f"{C['fisher_family_size']} binary codes; exploratory. Restricted readings "
             "(\\autoref{sec:7:CodingTheTranscripts}): \\textsuperscript{a}~includes hits that consist of a notes "
             f"heading only, without them C~{agr['core_C_without_heading_only']} and D~{agr['core_D']} in the core "
             f"suite ($p_\\mathrm{{Holm}} = {agr['holm_p_core_without_heading_only']:.3f}$); "
             "\\textsuperscript{b}~lower bound; \\textsuperscript{c}~never or almost never observed, detection not "
             "validated; \\textsuperscript{d}~not interpreted, since any count on the target column counts as "
             "inspection. Arms as in \\autoref{tab:3:arms}.}")
    L.append("  \\label{tab:7:process}")
    L.append("  \\setlength{\\tabcolsep}{4.5pt}")
    L.append("  \\begin{tabular}{llrrrrrrr}")
    L.append("    \\toprule")
    L.append("     & & \\multicolumn{5}{c}{Core suite} & \\multicolumn{2}{c}{Post-cutoff} \\\\")
    L.append("    \\cmidrule(lr){3-7}\\cmidrule(lr){8-9}")
    L.append("    Code & Task & A & B & C & D & $p_\\mathrm{Holm}$ & C & D \\\\")
    for gname, codes in GROUPS:
        L.append("    \\midrule")
        L.append(f"    \\multicolumn{{9}}{{l}}{{\\emph{{{gname}}}}} \\\\")
        for code, label, task in codes:
            ph = C["fisher_C_vs_D"][code]["p_holm"]
            L.append(f"    \\quad {_lab(code, label)} & {task} & {_cell(C, code, 'A')} & {_cell(C, code, 'B')} & "
                     f"{_cell(C, code, 'C')} & {_cell(C, code, 'D')} & {_p(ph)} & "
                     f"{_cell(P, code, 'C')} & {_cell(P, code, 'D')} \\\\")
    L.append("    \\midrule")
    L.append("    \\multicolumn{9}{l}{\\emph{Process flow (free arms only)}} \\\\")
    for code, label in FLOW:
        ph = C["fisher_C_vs_D"][code]["p_holm"]
        L.append(f"    \\quad {_lab(code, label)} & -- & -- & -- & {_cell(C, code, 'C')} & {_cell(C, code, 'D')} & "
                 f"{_p(ph)} & {_cell(P, code, 'C')} & {_cell(P, code, 'D')} \\\\")
    numC, numP = C["numeric"], P["numeric"]

    def med(scope, letter, key):
        x = scope[letter][key]
        return f"{x['median']:g}" if x else "--"
    L.append("    \\midrule")
    L.append("    \\multicolumn{9}{l}{\\emph{Medians per run}} \\\\")
    L.append(f"    \\quad Tool actions (free arms only) & -- & -- & -- & {med(numC, 'C', 'n_actions')} & "
             f"{med(numC, 'D', 'n_actions')} & -- & {med(numP, 'C', 'n_actions')} & {med(numP, 'D', 'n_actions')} \\\\")
    L.append(f"    \\quad Model families & 4.1 & {med(numC, 'A', 'n_families')} & "
             f"{med(numC, 'B', 'n_families')} & {med(numC, 'C', 'n_families')} & {med(numC, 'D', 'n_families')} & "
             f"-- & {med(numP, 'C', 'n_families')} & {med(numP, 'D', 'n_families')} \\\\")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{table}")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "tab7_process.tex"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {p.name}")


# Einziger freier Run auf student_health, der seine gewichtete Variante gegen eine ungewichtete verglich
# (fit(..., sample_weight=...)); vom Kodierer nicht als Tuning erkannt -> Handbefund kap7.md F7-929, F6-532
UNWEIGHTED_COMPARISON = {"main-student_health-crisp_react-r4"}
POSTCUTOFF = REPO / "thesis" / "facts" / "derived" / "postcutoff.json"


def write_table_student_health(d: dict, out_dir: Path) -> None:
    """tab7_student_health.tex: die 24 Runs auf student_health nach dem Weg zum Endmodell (Kap. 7.3).

    Gewichtung, Modellfamilien, Tuning und Ergebnis aus process_coding.json; Gate-Ruecksprung der
    A/B-Runs aus postcutoff.json (eval_to_model_runs_fsm). Fakten: F6-506, F6-524, F6-525, F6-526,
    F6-527, F7-282, F7-284, F7-929, F7-931."""
    runs = {k: r for k, r in d["runs"].items() if r["dataset"] == "student_health"}
    assert len(runs) == 24, len(runs)
    back = set(json.loads(POSTCUTOFF.read_text(encoding="utf-8"))["eval_to_model_runs_fsm"])
    routes = ["several", "gate", "unweighted", "single", "none"]
    count = {rt: Counter() for rt in routes}
    passed = {rt: 0 for rt in routes}
    arm_passed = Counter()
    for k, r in runs.items():
        v, a = r["values"], r["letter"]
        if not v["model_class_weight"]:
            assert a in "CD", k
            rt = "none"
        elif a in "AB":
            assert v["model_multi_family"], k
            rt = "gate" if k in back else "several"
        elif k in UNWEIGHTED_COMPARISON:
            rt = "unweighted"
        else:
            assert not v["model_multi_family"] and not v["model_tuning"], k
            rt = "single"
        count[rt][a] += 1
        passed[rt] += bool(r["success"])
        arm_passed[a] += bool(r["success"])
    assert sum(sum(c.values()) for c in count.values()) == 24
    # "single": eine Modellfamilie, kein Tuning (assert oben); nur gegen den Dummy geprueft laut Code-Sichtung F6-526
    label = {"several": "No return; several families compared, weighted model kept",
             "gate": "Returned by the Evaluation gate; weighting removed or weakened",
             "unweighted": "Weighted model compared with an unweighted variant",
             "single": "One weighted model, compared only with a dummy",
             "none": "No class weights"}
    L = ["%!TEX root = ../thesis.tex",
         "% generiert von thesis/scripts/make_figures_kap7.py aus thesis/facts/derived/process_coding.json"
         " und postcutoff.json — nicht von Hand aendern",
         "\\begin{table}[htbp]", "  \\centering", "  \\small",
         "  \\caption[Routes on student health]{Runs on student health (post-cutoff control, six per arm) by the "
         "route to the final model. \\emph{Balanced class weights set}: the code of the run sets balanced class "
         "or sample weights at some point. Runs of A and B are assigned by whether the Evaluation gate returned "
         "them to Modeling (event log); that the seven runs without a return kept a weighted model and that the "
         "five with a return removed or weakened the weighting rests on an inspection of their code, as does the "
         "dummy as the only comparison of the eight free runs in the fourth row "
         "(\\autoref{sec:6:PostCutoffControl}). The one run of C that compared its weighted model with an "
         "unweighted variant is not counted as tuning by the coder, because the weights enter at fitting and not "
         "in the model constructor (\\autoref{sec:7:CodingTheTranscripts}). "
         "\\emph{Passed}: not significantly worse than the baseline; last column: passed runs of the row. "
         "Arms as in \\autoref{tab:3:arms}.}",
         "  \\label{tab:7:studenthealth}",
         "  \\begin{tabular}{lrrrrr}",
         "    \\toprule",
         "    Route to the final model & A & B & C & D & Passed \\\\",
         "    \\midrule",
         "    \\multicolumn{6}{l}{\\emph{Balanced class weights set}} \\\\"]
    for rt in routes:
        if rt == "none":
            L.append("    \\midrule")
        c = count[rt]
        ind = "" if rt == "none" else "\\quad "
        L.append(f"    {ind}{label[rt]} & {c['A']} & {c['B']} & {c['C']} & {c['D']} & "
                 f"{passed[rt]} of {sum(c.values())} \\\\")
    L.append("    \\midrule")
    L.append(f"    Passed (of 6) & {arm_passed['A']} & {arm_passed['B']} & {arm_passed['C']} & {arm_passed['D']} & "
             f"{sum(arm_passed.values())} of 24 \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "tab7_student_health.tex"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {p.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=REPO / "thesis" / "figs")
    ap.add_argument("--table-dir", type=Path, default=REPO / "thesis" / "tables")
    ap.add_argument("--preview-dir", type=Path, default=None,
                    help="zusaetzlich PNG-Vorschau (300 dpi) hierhin; nicht ins Repo legen")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # Pfeile in den Konsolen-Kennzahlen (Windows cp1252)
    setup_style()
    d = load()
    print("fig7_process")
    fig7_process(d, args.out_dir, args.preview_dir)
    print("fig7_sequence")
    fig7_sequence(d, args.out_dir, args.preview_dir)
    write_table(d, args.table_dir)
    write_table_student_health(d, args.table_dir)
    _ = mf  # Stilmodul bewusst importiert


if __name__ == "__main__":
    main()
