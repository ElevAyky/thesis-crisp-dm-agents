# thesis/scripts/make_factsheets.py
"""Faktenblaetter thesis/facts/kap5.md und thesis/facts/kap6.md.

Alle "roh"-Werte kommen aus thesis/facts/derived/*.json (erzeugt von collect_facts.py
aus den Rohdaten); "doku"-Fakten sind aus der Projekt-Doku uebernommen und NICHT
nachgerechnet (Quelle mit Datei:Zeile). Die Datei wird komplett neu geschrieben —
Aenderungen hier vornehmen, nicht in den .md-Dateien.

IDs: F5-/F6-<Abschnitt><laufende Nr.>, z. B. F6-203 = Abschnitt 6.2, Fakt 3. Neue Fakten
am Ende eines Abschnitts anfuegen, damit bestehende IDs stabil bleiben.

Aufruf (Repo-Root, nach collect_facts.py):
    uv run python thesis/scripts/make_factsheets.py
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DER = os.path.join(REPO, "thesis", "facts", "derived")
ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
L = {"hard_fsm": "A", "soft_fsm": "B", "crisp_react": "C", "naked_react": "D"}
FULL = {"hard_fsm": "A (Hard FSM)", "soft_fsm": "B (Soft FSM)", "crisp_react": "C (CRISP-ReAct)",
        "naked_react": "D (Naked ReAct)"}
CORE_ORDER = ["wine", "breast_cancer", "cnae_9", "covertype", "phoneme", "fashion_mnist", "adult", "vehicle",
              "bank_marketing", "credit_g", "connect_4", "eucalyptus", "cmc"]
POST_ORDER = ["student_health", "ev_purchase"]


def j(name):
    with open(os.path.join(DER, name), encoding="utf-8") as f:
        return json.load(f)


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def s3(x):
    return "0.000" if round(x, 3) == 0 else f"{x:+.3f}"


def s2(x):
    return "0.00" if round(x, 2) == 0 else f"{x:+.2f}"


def pv(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def usd(x):
    return f"{x:.2f} USD"


def wil(w):
    return f"[{w[0]:.3f}, {w[1]:.3f}]"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f} %"


def r2(x):
    return f"{x:.2f}"


def arms_str(d, fmt=f3, sep=" · "):
    return sep.join(f"{L[a]} {fmt(d[a])}" for a in ARMS if a in d)


class Sheet:
    def __init__(self, chap: int, title: str, intro: str):
        self.chap, self.title, self.intro = chap, title, intro
        self.sections = []

    def sec(self, title: str, num: int):
        self.sections.append({"title": title, "num": num, "facts": []})

    def f(self, statement: str, value: str, source: str, level: str = "roh"):
        s = self.sections[-1]
        fid = f"F{self.chap}-{s['num']}{len(s['facts']) + 1:02d}"
        s["facts"].append((fid, statement, value, source, level))
        return fid

    def render(self) -> str:
        out = [f"# {self.title}", "", self.intro, ""]
        n = 0
        for s in self.sections:
            out += [f"## {s['title']}", "", "| ID | Aussage | Wert | Quelle | Prüfstufe |", "|---|---|---|---|---|"]
            for fid, st, val, src, lvl in s["facts"]:
                cells = [c.replace("|", "\\|").replace("\n", " ") for c in (fid, st, val, src, lvl)]
                out.append("| " + " | ".join(cells) + " |")
                n += 1
            out.append("")
        out.append(f"*{n} Fakten. Generiert von `thesis/scripts/make_factsheets.py` aus `thesis/facts/derived/`.*")
        return "\n".join(out) + "\n"

    def count(self):
        return sum(len(s["facts"]) for s in self.sections)


INTRO = (
    "> **Einzige Zahlenquelle für die Schreib-Agenten.** Generiert — nicht von Hand ändern, sondern\n"
    "> `thesis/scripts/collect_facts.py` (Rohdaten → `thesis/facts/derived/*.json`) und\n"
    "> `thesis/scripts/make_factsheets.py` neu laufen lassen. Prüfstufe **roh** = aus Rohdaten oder Code\n"
    "> berechnet; **doku** = aus der Projekt-Doku übernommen, nicht nachgerechnet; **roh (archiviert)** =\n"
    "> aus archivierten, kontaminierten Läufen berechnet, **nur** als ausdrücklich benannter Vergleich in\n"
    "> Kap. 5 zitierbar. Runs-Repo = `../thesis-crisp-dm-runs`; `derived/` = `thesis/facts/derived/`.\n"
    "> Arme: A = Hard FSM, B = Soft FSM, C = CRISP-ReAct, D = Naked ReAct. „SUCCESS“ = *pass* in Kap. 3;\n"
    "> `below_threshold` = *below baseline*. F1 = macro-F1 auf dem Test-Anteil (20 %), wenn nicht anders gesagt.\n"
    "> Zeiten lokal (CEST bis 25.10. 03:00, danach CET).")


# =========================================================================== Kap. 5

def kap5() -> Sheet:
    cal, ctl, tl, con = j("calibration.json"), j("controls.json"), j("timeline.json"), j("contamination.json")
    kg, dc, sp, cost = j("kgit.json"), j("design_checks.json"), j("splits.json"), j("costs.json")
    lk = j("leakage.json")
    S = Sheet(5, "Faktenblatt Kapitel 5 — Experiment Execution", INTRO)
    cells = {(c["phase"], c["dataset"], c["arm"]): c for c in cal["cells"]}
    h = cal["heuristic"]
    SRC_CAL = "runs `calib-*/summary.json` (final_f1, total_cost_usd, outcome) → `derived/calibration.json`"
    SRC_SCAN = "runs `scan-*/summary.json` → `derived/calibration.json`"

    # ---------------------------------------------------------------- 5.1
    S.sec("5.1 Kalibrierung (08.08.) und Varianz-Scan (08.–09.08.)", 1)
    S.f("Datum Kalibrierungslauf", "08.08.2026", "docs/experiment-design.md:253", "doku")
    c = cal["calibration"]
    S.f("Umfang Kalibrierung: Sets × Arme × n", f"{c['runs']} Runs = 2 Sets (credit_g, covertype) × 2 Arme (A, D) × n = 3",
        SRC_CAL + " calibration.runs/datasets/arms")
    S.f("Ergebnis Kalibrierung", f"{c['success']}/{c['runs']} SUCCESS (alte Schwelle F1 ≥ 0.5), Holdout in allen Runs intakt",
        SRC_CAL + " calibration.success, cells[*].thresholds, holdout_intact_all")
    S.f("Kosten Kalibrierung gesamt und pro Run", f"{usd(c['cost'])}; pro Run A {usd(c['cost_per_run_by_arm']['hard_fsm'])}, "
        f"D {usd(c['cost_per_run_by_arm']['naked_react'])}", SRC_CAL + " calibration.cost, cost_per_run_by_arm")
    cg_a, cg_d = cells[("calib", "credit_g", "hard_fsm")], cells[("calib", "credit_g", "naked_react")]
    cv_a, cv_d = cells[("calib", "covertype", "hard_fsm")], cells[("calib", "covertype", "naked_react")]
    S.f("Run-SD der Kalibrierungszellen (F1, n = 3)",
        f"credit_g A {f4(cg_a['sd'])}, D {f4(cg_d['sd'])}; covertype A {f4(cv_a['sd'])} (3× identisch {f4(cv_a['f1_values'][0])}), "
        f"D {f4(cv_d['sd'])}", SRC_CAL + " cells[calib].sd, f1_values")
    S.f("Zellmittel der Kalibrierung (F1)", f"credit_g A {f4(cg_a['mean'])}, D {f4(cg_d['mean'])}; covertype A {f4(cv_a['mean'])}, "
        f"D {f4(cv_d['mean'])}", SRC_CAL + " cells[calib].mean")
    S.f("SD-Verhältnis D/A auf credit_g (Kalibrierung)", f"{h['calib_credit_g_D_over_A_sd_ratio']:.1f}×",
        "derived/calibration.json heuristic.calib_credit_g_D_over_A_sd_ratio")
    md = c["mean_diff_A_minus_D"]
    S.f("Mittelwert-Differenz A − D in der Kalibrierung", f"credit_g {s3(md['credit_g'])}, covertype {s3(md['covertype'])}",
        "derived/calibration.json calibration.mean_diff_A_minus_D")
    S.f("Datum Varianz-Scan", "08.–09.08.2026", "docs/experiment-design.md:285", "doku")
    s = cal["scan"]
    S.f("Umfang Varianz-Scan (Runs mit summary.json)",
        f"{s['runs']} Runs: {s['runs_suite']} Suite-Runs (D × {s['d_suite_cells']} Sets + A × {s['a_suite_cells']} Sets "
        f"[bank_marketing, cnae_9, fashion_mnist], je n = 3) + {s['runs_out_of_suite']} Außer-Suite-Runs "
        f"({', '.join(x.split('|')[0] + ' × ' + L[x.split('|')[1]] for x in s['out_of_suite_cells'])})", SRC_SCAN + " scan.*")
    S.f("Kosten Varianz-Scan", f"{usd(s['cost'])} gesamt; Suite {usd(s['cost_suite'])}, Außer-Suite {usd(s['cost_out_of_suite'])}",
        SRC_SCAN + " scan.cost, cost_suite, cost_out_of_suite")
    S.f("Ergebnis Varianz-Scan", f"{s['success']} SUCCESS, {s['fail']} FAIL (beide bank_marketing × D, r2 und r3, final_f1 = None)",
        SRC_SCAN + " cells[scan|bank_marketing|naked_react].outcomes")
    S.f("FAIL-Modus bank_marketing × D im Scan: Kontraktbruch (Wahrscheinlichkeiten statt Klassen-Labels nach "
        "Metrik-Wahl ROC AUC); A hielt den Kontrakt 3/3", "2/3 Runs", "docs/experiment-design.md:323-329", "doku")
    cvs = cells[("scan", "covertype", "naked_react")]
    S.f("Lauteste Einzelzelle des Scans", f"covertype × D, SD {f4(cvs['sd'])} (F1 {', '.join(f3(x) for x in cvs['f1_values'])})",
        SRC_SCAN + " heuristic.strict_single_max_cell, cells[scan|covertype|naked_react]")
    d_suite = [x for x in cal["cells"] if x["phase"] == "scan" and x["in_core_suite"] and x["arm"] == "naked_react"]
    d_nosd = [x["dataset"] for x in d_suite if x["sd"] is None]
    assert len(d_suite) == 13 and d_nosd == ["bank_marketing"], d_nosd  # Wortlaut unten haengt daran
    S.f(f"D-Zellen im Scan mit Run-SD > 0.01 bzw. ≥ 0.015 (von {len(d_suite)} Sets; bank_marketing × D ohne SD, nur 1 "
        f"gültiger Run → {len(d_suite) - len(d_nosd)} Sets mit messbarer SD) [Nenner korrigiert 09.11., Prüfbefund K5-03]",
        f"> 0.01: {len(s['d_cells_sd_gt_0_01'])} von {len(d_suite) - len(d_nosd)} ({', '.join(s['d_cells_sd_gt_0_01'])}); ≥ 0.015: "
        f"{len(s['d_cells_sd_ge_0_015'])} ({', '.join(s['d_cells_sd_ge_0_015'])})", SRC_SCAN + " scan.d_cells_sd_*")
    S.f("Größte Run-SD einer A-Zelle (Suite, Kalibrierung + Scan, 5 Zellen)", f"{f4(h['a_cells_max_sd'])} (fashion_mnist × A)",
        "derived/calibration.json heuristic.a_cells_max_sd")
    pc = h["pooled"]["covertype|naked_react"]
    S.f("n-Heuristik: Ziel-SE = 0.05/3; gepoolte SD der lautesten Zelle covertype × D (Kalibrierung + Scan, df 2 + 2)",
        f"SE-Ziel {f4(h['target_se'])}; SD √(({f4(pc['calib_sd'])}² + {f4(pc['scan_sd'])}²)/2) = {f4(pc['pooled_sd'])} → "
        f"n ≥ {pc['n_required']:.2f} → n = {h['n_chosen']} (SE bei n = 6: {f4(h['se_at_n6'])})",
        "derived/calibration.json heuristic.pooled, n_required_worst, n_chosen, se_at_n6")
    S.f("Strikte Variante (ungepoolter Einzelwert covertype × D)", f"n ≥ {h['strict_n_required']:.1f} → n = 11",
        "derived/calibration.json heuristic.strict_n_required")
    pg = h["pooled"]["credit_g|naked_react"]
    S.f(f"Gepoolte SD credit_g × D (Kalibrierung {f4(pg['calib_sd'])} / Scan {f4(pg['scan_sd'])})",
        f"{f4(pg['pooled_sd'])} → n ≥ {pg['n_required']:.1f}", "derived/calibration.json heuristic.pooled")
    ua, ud = cells[("scan", "uber_rides", "hard_fsm")], cells[("scan", "uber_rides", "naked_react")]
    # F5-120/121: Wortlaut aus Korrektur-Runde 1 (09.11., K5-01) -- Mechanismus per Lesen der worker_text/du_report
    # festgestellt, nicht in derived/ abgelegt; "Kontaminationssignal" (experiment-design.md:334) war falsch.
    S.f("Außer-Suite uber_rides (Kaggle-Reupload, disqualifiziert): D erreicht F1 = 1.000 in 3/3 über ein **Datenleck**, "
        "nicht über Memorisierung: der Buchungsstatus ist aus den Abbruchgrund-Spalten (INCOMPLETE_RIDES_REASON, "
        "DRIVER_CANCELLATION_REASON, REASON_FOR_CANCELLING_BY_CUSTOMER) + fehlender PAYMENT_METHOD deterministisch "
        "ableitbar; D nutzt das in allen 3 Runs (worker_text) [korrigiert 09.11., Prüfbefund K5-01; „Kontaminationssignal“ "
        "aus experiment-design.md:334 war falsch]",
        f"D {', '.join(f3(x) for x in ud['f1_values'])}", SRC_SCAN + " cells[scan|uber_rides|naked_react]; runs "
        "`scan-uber_rides-naked_react-{1,2,3}/events.jsonl` worker_text")
    S.f("Außer-Suite uber_rides × A (in experiment-design §5.2 nicht berichtet): DU schließt die drei Abbruchgrund-Spalten "
        "in 3/3 Runs als Leakage aus, PAYMENT_METHOD in 2/3 (r1, r3) [ergänzt 09.11., K5-01]",
        f"F1 {', '.join(f3(x) for x in ua['f1_values'])} → SD {f4(ua['sd'])}", SRC_SCAN + " cells[scan|uber_rides|hard_fsm]; "
        "runs `scan-uber_rides-hard_fsm-{1,2,3}/summary.json` du_report.excluded_columns")
    cn_a, cn_d = cells[("scan", "cnae_9", "hard_fsm")], cells[("scan", "cnae_9", "naked_react")]
    fm_a, fm_d = cells[("scan", "fashion_mnist", "hard_fsm")], cells[("scan", "fashion_mnist", "naked_react")]
    S.f("Explorativ: D vor A auf den High-Dim-Sets im Scan (weighted F1)",
        f"cnae_9 D {f3(cn_d['mean'])} vs. A {f3(cn_a['mean'])}; fashion_mnist D {f3(fm_d['mean'])} vs. A {f3(fm_a['mean'])}",
        SRC_SCAN + " cells[scan|cnae_9|*], cells[scan|fashion_mnist|*]")
    S.f("Design-Befunde aus Kalibrierung/Scan: (a) n = 6; (b) Run-SD als eigene AV; (c) SUCCESS-Rate als "
        "Enforcement-Messpunkt (Kontraktbruch); (d) uber_rides als Post-Cutoff-Kandidat disqualifiziert, weil die "
        "Quelldaten vor dem Cutoff liegen (Upload-Datum ≠ Datenherkunft) — F1 = 1.0 von D ist Datenleck (F5-120), keine "
        "Bestätigung der Disqualifikation [korrigiert 09.11., K5-01]", "—",
        "docs/experiment-design.md:269-274, 323-336; docs/benchmarks.md:117-119; docs/meetings/2026-09-14-entscheidungen.md:18",
        "doku")
    S.f("Budget-Extrapolation Kern aus der Kalibrierung (n = 6, damals 10 Sets)", "≈ 270 USD",
        "docs/experiment-design.md:280-282", "doku")
    S.f("VORBEHALT: Runs vor dem 18.09. (Kalibrierung, Scan) messen weighted-F1 statt macro-F1 und liefen ohne "
        "isolierten Worker-Kontext (Account-Plugins/MCP im Systemprompt) → nur Design-Input, Werte nicht mit "
        "Kern-Runs mischen; Kosten/Tokens nicht vergleichbar", "—", "docs/roadmap.md:343, 345; CLAUDE.md:77", "doku")
    S.f("VORBEHALT: alle Kalibrierungs-/Scan-Runs liefen mit Punkt-Schwelle F1 ≥ 0.5 (eval_f1_threshold) — "
        "SUCCESS dort nicht mit dem Kern vergleichbar",
        f"0.5 in {sum(x['n'] for x in cal['cells'] if x['thresholds'] == [0.5])}/{sum(x['n'] for x in cal['cells'])} Runs",
        SRC_CAL + " cells[*].thresholds")
    # F5-127: Zeitangaben aus den summary.json-mtimes (Korrektur-Runde 1, K5-02); "je nach Tag"
    # (experiment-design.md:316) war falsch.
    S.f("VORBEHALT: SDs aus n = 3 (df = 2) sind selbst unsicher — dieselbe Zelle zweimal am **selben Tag** (08.08.) "
        "gemessen, Kalibrierung und Scan ~1 h auseinander (summary-mtimes: calib credit_g × D 17:28–17:56, scan "
        "18:35–19:44; calib covertype × D 17:36–18:04, scan 18:46–19:56). Nur connect_4 × D und fashion_mnist × A liefen "
        "am 09.08. [korrigiert 09.11., K5-02; „je nach Tag“ in experiment-design.md:316 war falsch]",
        f"credit_g × D {f4(pg['calib_sd'])} vs. {f4(pg['scan_sd'])}; covertype × D {f4(pc['calib_sd'])} vs. {f4(pc['scan_sd'])}",
        "derived/calibration.json heuristic.pooled; runs `calib-*`/`scan-*` summary.json mtime")
    S.f("VORBEHALT: Scan-Manifeste vom 08.08. provisorisch, Prompt-Wording weicht minimal ab", "—",
        "docs/experiment-design.md:285-287", "doku")
    S.f("Ordner ohne summary.json (Runner-Bug-Rest vom 23.09., nicht Teil des Scans)", ", ".join(cal["runs_without_summary"]),
        "runs-Repo Verzeichnisliste; docs/roadmap.md:348")
    S.f("Probe-Runs k = 1 vor dem Kern (18.09., Arm A, isolierter Kontext): cmc SUCCESS 0.496; credit_g FAIL "
        "below_threshold nach 3× eval→model, 2.16 USD", "2 Runs", "docs/roadmap.md:346", "doku")
    stage = {"calib": "Kalibrierung", "scan": "Scan"}
    for x in sorted(cal["cells"], key=lambda c: (c["phase"] != "calib", not c["in_core_suite"], c["arm"] != "hard_fsm", c["dataset"])):
        sdv = "—" if x["sd"] is None else f4(x["sd"])
        S.f(f"Zelle {stage[x['phase']]} {x['dataset']} × {L[x['arm']]}" + ("" if x["in_core_suite"] else " (Außer-Suite)")
            + ": SUCCESS, Mittel-F1 (weighted, nur Runs mit F1), Run-SD, USD/Run",
            f"{x['outcomes'].get('SUCCESS', 0)}/{x['n']}; {f3(x['mean'])}; {sdv}; {x['cost_per_run']:.2f}",
            f"runs `{x['phase']}-{x['dataset']}-{x['arm']}-*/summary.json` → `derived/calibration.json` cells")

    # ---------------------------------------------------------------- 5.1b Entscheidungen
    S.sec("5.1b Entscheidungen 14.09. (Protokoll) und Design-Änderungen 23.09.", 2)
    M = "docs/meetings/2026-09-14-entscheidungen.md"
    S.f("Entscheidungstermin vor den Haupt-Runs mit Decker", "14.09.2026", f"{M}:1-3", "doku")
    S.f("Replikate pro Zelle", "n = 6 (SE ≤ ⅓ × 0.05 F1, gepoolt)", f"{M}:18", "doku")
    S.f("Anzahl Kern-Datasets und kritische Differenz (Nemenyi, k = 4, α = 0.05)",
        "N = 13, CD = 1.30 (a priori); Post-Cutoff-Sets zusätzlich, nicht in N", f"{M}:19; scripts/analyze_main.py nemenyi_cd", "doku")
    S.f("Modell und Training-Cutoff",
        "Claude Opus 4.8 allein, Cutoff 2026-01-31 (Modellseite nennt nur „Jan 2026“; 31.01. = im Meeting gesetztes Monatsende)",
        f"{M}:20; thesis.bib anthropic2026opus48 (geprüft 09.11.)", "doku")
    S.f("Zweitmodell", "entfällt (Modell-Robustheit → Limitation/Future Work)", f"{M}:21", "doku")
    S.f("Geplante Erfolgsschwelle (14.09.)", "baseline-relativ, k = 1: SUCCESS ⇔ F1 ≥ best-Baseline-F1", f"{M}:22", "doku")
    S.f("Metrik angleichen: Harness von weighted- auf macro-F1 (umgesetzt 18.09.)", "macro-F1", f"{M}:22; docs/roadmap.md:343", "doku")
    S.f("Schwere-Konvention", "Terzile 4 leicht / 4 mittel / 5 schwer", f"{M}:23", "doku")
    S.f("Post-Cutoff-Kontrolle", "synthetische Kaggle-Playground-Episoden mit Start nach 2026-01-31", f"{M}:24", "doku")
    S.f("Erwartungs-Muster a priori festgeschrieben (Präregistrierung)", "5 Muster, s. F6-9xx", f"{M}:25", "doku")
    S.f("Neu aus dem Termin: Harness-Overhead messen → Isolations-Flags (gesetzt 18.09.); ERROR-State + fail_reason "
        "in allen Armen (18.09.)", "—", f"{M}:30-52; docs/roadmap.md:344-345", "doku")
    D23 = "docs/decisions-2026-09-23.md"
    S.f("Anlass der Design-Änderungen 23.09.: Befunde an den ersten 156 Kern-Runs (C/D) + unabhängiges Review des "
        "englischen Wortlauts", "—", f"{D23}:3-6", "doku")
    S.f("E1 Erfolgskriterium: gepaarter Bootstrap gegen eine Referenz-Baseline auf denselben Zeilen", "—", f"{D23}:14; "
        "docs/experiment-design.md:111-117", "doku")
    cfg = dc["config"]
    S.f("S1 Split Train/Validierung/Test (Code)",
        f"{1 - cfg['val_fraction'] - cfg['test_fraction']:.0%}/{cfg['val_fraction']:.0%}/{cfg['test_fraction']:.0%}, seed 42; "
        "Validierung nur für die Gates von A/B, Endurteil aller Arme auf Test", f"{cfg['_source']}; {D23}:35-41")
    S.f("S1 Split in allen sauberen Runs eingehalten (n_train/n_val/n_test aus den [setup]-Zeilen der Runner-Logs)",
        "Anteile 0.70/0.10/0.20 (±0.003 Rundung) auf allen 15 Sets, in allen 438 Logs konsistent",
        "runs `_main|_post_cutoff|_leakage|_spurious/logs/*.log` → `derived/splits.json`")
    S.f("S4 gleiches Run-Kontingent für alle Arme (Code)",
        f"{cfg['run_budget_usd']:.0f} USD / {cfg['run_max_turns']} Turns pro Run (nur Worker); A/B pro Phasen-Call zusätzlich "
        f"{cfg['budget_usd_per_call']:.0f} USD / {cfg['max_turns_per_call']} Turns; C/D react_max_turns {cfg['react_max_turns_default']}",
        f"src/crispfsm/config.py:41-44, 57 → `derived/design_checks.json` config; {D23}:86-92")
    S.f("B3 Zielspalte: A/B arbeiten ab DU mit der eigenen BU-Wahl; SUCCESS verlangt korrekte Zielspalte", "—", f"{D23}:68-73", "doku")
    S.f("E2/S5 CRISP-Text wörtlich aus CRISP-DM 1.0, generische Aufgaben der Phasen 1–5 (ohne Deployment)", "—", f"{D23}:15, 24", "doku")
    S.f("E3 alle LLM-seitigen Texte englisch", "—", f"{D23}:16", "doku")
    S.f("B2 allgemeine ML-Hinweise in den Prompt von D → C–D-Diff ist exakt der CRISP-Text", "—", f"{D23}:21, 102-105", "doku")
    S.f("S8 Harness-Commit + Prompt-Fingerabdruck in jedem Summary", "in 438/438 sauberen Runs vorhanden",
        f"{D23}:26, 111; runs `*/summary.json` harness.commit/prompt_sha256 → `derived/controls.json`")
    S.f("R2-5 Model-Gate (A/B): metrics.json in der Spec-Metrik, val_score > dummy_score", "—", f"{D23}:136-145", "doku")
    S.f("Sekundäre Kennzahl „signifikant besser“ (ci_lo > 0) festgelegt", "23.09.2026, 23:42 (Commit 559eafc)",
        "docs/experiment-design.md:350-358; git log Harness-Repo → `derived/timeline.json` _harness_repo_key_commits")
    S.f("Baseline-Referenz auf der Validierung gewählt (LogReg vs. RF, Default); Test-Referenz (baseline_f1) daher auf "
        "einigen Sets unter dem auf Test stärkeren Modell (vehicle 0.756 statt 0.813, credit_g 0.632 statt 0.658, "
        "breast_cancer 0.962 statt 0.971)", "—", f"{D23}:45-56", "doku")

    # ---------------------------------------------------------------- 5.2 Ablauf
    S.sec("5.2 Ablauf der sauberen Runs (24.–27.10.)", 3)
    order = [("leak_perfect", "Leakage perfekt, Arm D"), ("leak_spurious", "Leakage spurious, Arm D"),
             ("core", "Kern, 4 Arme verschränkt"), ("post_cutoff", "Post-Cutoff, 4 Arme verschränkt")]
    S.f("Reihenfolge der sauberen Batches (ein Harness-Code-Stand)",
        " → ".join(n for _, n in order), "derived/timeline.json *.start_utc; docs/roadmap.md:30-32")
    for key, name in order:
        t = tl[key]
        gaps = t["gaps_over_60min"]
        S.f(f"Batch {name}: Runs, Harness, Claude Code, Zeitraum, Kosten",
            f"{t['n_runs']} Runs; Harness {', '.join(t['harness_commits'])}; Claude Code "
            f"{t['claude_versions'][0].split(' ')[0]}; {t['start_local']} – {t['end_local']}; Wanduhr {t['wall_clock_h']:.2f} h"
            + (f", davon {len(gaps)} Pause(n) > 60 min → aktive Laufzeit {t['wall_clock_minus_gaps_h']:.1f} h" if gaps else "")
            + f"; {usd(t['cost_usd'])}",
            "Transkript-Zeitstempel `<run>/transcripts/*.jsonl` + summary.harness/total_cost_usd → `derived/timeline.json`")
    changed = [x.split("|")[0].strip() for x in tl["_diff_9bb6a91_14e99e4"] if "|" in x]
    S.f("Harness 9bb6a91 (Leakage perfekt) vs. 14e99e4 (übrige): Unterschied nur Doku",
        f"{len(changed)} Dateien geändert: {', '.join(changed)} (kein Code)", "git diff --stat 9bb6a91 14e99e4 → `derived/timeline.json`")
    S.f("Prompt-Fingerabdruck identisch in allen sauberen Runs", "prompt_sha256 79225b95… in 438/438",
        "runs `*/summary.json` harness.prompt_sha256 → `derived/controls.json`")
    S.f("Gesamtkosten aller sauberen Batches", usd(cost["_all_clean_batches_total"]), "derived/costs.json _all_clean_batches_total")
    pl = dc["core_plan"]
    S.f("Kern-Plan: 312 Jobs, replikat-weise (erst r1 aller 52 Zellen, dann r2 …), innerhalb eines Sets Reihenfolge D, C, A, B",
        f"replicate_major = {pl['replicate_major']}; Arm-Reihenfolge {', '.join(L[a] for a in pl['arm_order_within_dataset'])}",
        "runs `_main/plan.json` → `derived/design_checks.json` core_plan")
    g1, g2 = tl["core"]["gaps_over_60min"]
    S.f("Unterbrechungen Kern (Pausen > 60 min zwischen zwei Runs)",
        f"2: {g1['from_local']} → {g1['to_local']} ({g1['minutes'] / 60:.1f} h); {g2['from_local']} → {g2['to_local']} "
        f"({g2['minutes'] / 60:.1f} h)", "Transkript-Zeitstempel → `derived/timeline.json` core.gaps_over_60min")
    cm = tl["_summaries_per_runs_repo_commit"]
    S.f("Zwischen-Commits Kern im runs-Repo (Anzahl sauberer main-Summaries im Commit)",
        f"22684c4 (25.10. 19:02): {cm['22684c4']['main']}/312; 3085c8b (26.10. 10:04): {cm['3085c8b']['main']}/312; "
        f"4b46569 (26.10. 19:56): {cm['4b46569']['main']}/312", "git ls-tree runs-Repo → `derived/timeline.json` _summaries_per_runs_repo_commit")
    S.f("NICHT IM TEXT VERWENDEN (Johannes 15.11.: PC aus/Neustart ist irrelevant) — Ursache der Unterbrechungen: "
        "Abmelden/PC-Neustart/Herunterfahren; Wiederaufnahme per Resume (Runner fährt nur offene Runs)",
        "—", "docs/roadmap.md:41-44; Commit-Nachrichten 22684c4, 3085c8b, 25644b7", "doku")
    gp = tl["post_cutoff"]["gaps_over_60min"][0]
    S.f("Unterbrechung Post-Cutoff", f"1: {gp['from_local']} → {gp['to_local']} ({gp['minutes'] / 60:.1f} h); Abschluss-Commit "
        f"25644b7 (27.10. 10:26) mit {cm['25644b7']['main']} main-Summaries (312 Kern + 48 Post-Cutoff)",
        "derived/timeline.json post_cutoff.gaps_over_60min, _summaries_per_runs_repo_commit")
    pa = tl["core"]["per_arm"]
    S.f("Laufzeit pro Run im Kern (Transkript-Spanne, Mittel / Median / Max, Minuten)",
        "; ".join(f"{L[a]} {pa[a]['mean_min']:.1f} / {pa[a]['median_min']:.1f} / {pa[a]['max_min']:.1f}" for a in ARMS),
        "derived/timeline.json core.per_arm")
    pp = tl["post_cutoff"]["per_arm"]
    S.f("Laufzeit pro Run Post-Cutoff (Mittel, Minuten)", "; ".join(f"{L[a]} {pp[a]['mean_min']:.1f}" for a in ARMS),
        "derived/timeline.json post_cutoff.per_arm")
    S.f("Summe der Run-Spannen (reine Agentenzeit inkl. Prüfer, ohne Setup/Scoring) Kern / Post-Cutoff",
        f"{tl['core']['sum_run_minutes_h']:.1f} h / {tl['post_cutoff']['sum_run_minutes_h']:.1f} h", "derived/timeline.json *.sum_run_minutes_h")
    S.f("ERROR-Outcomes in den finalen Summaries (crash, rate_limited, recursion_limit → werden neu gefahren)",
        "0 von 438", "runs `*/summary.json` outcome → `derived/timeline.json` *.errors_in_final_summaries")
    S.f("Rate-Limit-Ereignisse während der sauberen Runs", "nicht rekonstruierbar: Runner-Logs werden pro Versuch "
        f"überschrieben, Warte-Meldungen nur auf der Konsole; 0 von {tl['_logs_scanned']} finalen Logs mit Fehler-/Limit-Marker",
        "scripts/variance_scan.py:124-125, 140-158; runs `_*/logs/*.log` → `derived/timeline.json` _logs_with_error_or_limit_marker")
    S.f("Einziges dokumentiertes Kontingent-Ende: 24.09. (archivierter Arm-A-Lauf) „hit your session limit“ nicht erkannt → "
        "5× crash, Runner brach ab; Fix 63200e3 (Marker + leerer Workdir bei Neustart)", "—", "docs/roadmap.md:296-298; "
        "git log Harness-Repo (63200e3, 24.09. 19:29)", "doku")
    tot = sum(ctl[k]["n_runs"] for k in ctl)

    def allc(key):
        return sum(ctl[k].get(key, 0) for k in ctl)
    S.f("Kontrolle harness.dirty = false", f"{allc('dirty_false')}/{tot}", "runs `*/summary.json` harness.dirty → `derived/controls.json`")
    S.f("Kontrolle auto_memory und git_context = \"disabled\"", f"{allc('auto_memory_disabled')}/{tot} bzw. {allc('git_context_disabled')}/{tot}",
        "summary.harness.auto_memory/git_context → `derived/controls.json`")
    S.f("Kontrolle holdout_intact = true", f"{allc('holdout_intact')}/{tot}", "summary.holdout_intact → `derived/controls.json`")
    S.f("Kontrolle access_audit.flagged = false (Zugriffs-Audit aus den Transkripten), Treffer gesamt",
        f"{allc('access_audit_not_flagged')}/{tot}; Treffer {allc('access_audit_hits')}", "summary.access_audit → `derived/controls.json`")
    S.f("Kontrolle Transkripte nicht leer; Anzahl Transkript-Dateien = Anzahl Sessions (Worker + Prüfer)",
        f"{allc('transcripts_nonempty')}/{tot}; {allc('transcripts_match_sessions')}/{tot}",
        "`<run>/transcripts/*.jsonl` vs. events.jsonl session_id/gate_session_id → `derived/controls.json`")
    S.f("Claude-Code-Version in allen sauberen Runs", f"2.1.280 in {allc('claude_version:2.1.280 (Claude Code)')}/{tot}",
        "summary.harness.claude_version → `derived/controls.json`")
    S.f("Erfolgsregel in allen sauberen Runs (summary.success_rule)", f"noninferior_paired in {allc('success_rule:noninferior_paired')}/{tot}",
        "summary.success_rule → `derived/controls.json`")
    S.f("Korrekte Zielspalte (target_ok) / Enthaltungen (abstained) / budget_exhausted",
        f"{allc('bu_or_target_ok')}/{tot} / {allc('abstained')} / {allc('budget_exhausted')}",
        "summary.target_ok, abstained, budget_exhausted → `derived/controls.json`")
    runs_git = kg["runs_with_git_calls"]
    S.f("Worker-Bash-Aufrufe von git (Scan aller Transkripte der 438 sauberen Runs)",
        f"{len(runs_git)} Runs: " + "; ".join(f"{r['run_id']} ({r['batch']})" for r in runs_git)
        + " — Kern: nur `git status --short data/` im eigenen Ordner; Post-Cutoff: `git log`/`git status` (K-git)",
        "`<run>/transcripts/*.jsonl` tool_use.input.command → `derived/kgit.json`")
    S.f("Aggregate des sauberen Stands: rescored / kontaminiert / fehlend",
        "; ".join(f"{k}: {v['rescored_runs']}/{v['contaminated_runs']}/{v['missing_runs']} bei {v['n_measured']} gemessen"
                  for k, v in dc["aggregates"].items()), "runs `_*/main_runs.json` → `derived/design_checks.json` aggregates")

    # ---------------------------------------------------------------- 5.2b Kontamination
    S.sec("5.2b Kontaminations-Vorfall und Kern-Neustart Nr. 2 (24.10.)", 4)
    B = "docs/befund-auto-memory.md"
    S.f("Entdeckung: Abschlussmeldung von spur-bank_marketing-naked_react-r1 (kontaminierter Lauf) verweist auf die "
        "Leak-Spalte V17 „noted in prior runs“", "24.10.2026", f"{B}:9-13", "doku")
    S.f("Kanal Auto-Memory: alle Runs im Git-Repo thesis-crisp-dm-runs teilten ein Claude-Code-Projekt-Gedächtnis; die "
        "Isolations-Flags vom 18.09. schalten es nicht ab", "—", f"{B}:3-7", "doku")
    S.f("Kanarien-Test 1 (Kommandozeile aus backend.py, Memory-Datei mit Kennwort): Kern-Flags → Kennwort genannt; "
        "+ autoMemoryEnabled false oder Env CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 → NONE", "KANARIE-7319 vs. NONE", f"{B}:15-23", "doku")
    S.f("Gesichertes Gedächtnis (Snapshot 24.10.)", f"{con['snapshot_files']} Dateien in `_auto_memory_snapshot_2026-10-24/`",
        "runs-Repo Verzeichnis → `derived/contamination.json` snapshot_files")
    # F5-405: Herkunft per Frontmatter der Snapshot-Dateien (Korrektur-Runde 1, K5-04); befund-auto-memory.md:29-32
    # ordnet balanced-weight faelschlich dem Kern zu.
    S.f("Inhalt: technische Lehren aus dem Kern (u. a. die Notiz zum Aufblähen großer Baum-Ensembles, Datei "
        "`large-tree-…-blowup.md`: voll ausgewachsene RF/ExtraTrees auf großen Sets → Modell-Dateien von mehreren GB; "
        "modified 25.09. 21:16, Session aus automem_main-covertype-soft_fsm-r3), Lehren aus dem ersten Post-Cutoff-Lauf "
        "(balanced-weight-hurts-macrof1, student-health-f1-macro-hgb, ev-purchase-f1-threshold; 04.10.) und Leak-Notizen "
        "pro Datensatz aus den Leakage-Batches (24.10.) [korrigiert 09.11., K5-04: balanced-weight stammt NICHT aus dem "
        "Kern, sondern aus automem_main-student_health-hard_fsm-r1, model-Versuch 2 nach Eval-Gate-Rückmeldung; "
        "befund-auto-memory.md:29-32 ordnet es falsch ein]", "—",
        f"{B}:27-35; runs `_auto_memory_snapshot_2026-10-24/*.md` (Frontmatter modified, originSessionId); Dateinamen in "
        "`derived/contamination.json` snapshot_names")
    S.f("Betroffene archivierte Läufe (mit Gedächtnis im Kontext)", "Kern A 32/78, Kern B 78/78, Post-Cutoff 48/48, Leakage "
        "Batch 1 + 2 alle; Kern C/D (23.09.) vermutlich nicht (nicht beweisbar)", f"{B}:37-50", "doku")
    S.f("Zweiter Kanal K1 Git-Kontext: CLI hängt Git-Status + „Recent commits“ des runs-Repos an (Worker sahen "
        "Ergebnis-Commits); Fix Env CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1", "kritisch", f"{B}:88", "doku")
    S.f("Weitere Review-Befunde 24.10.: G1 geerbte Umgebung (~30 Claude-Variablen) → worker_env(); M1 Systemprompt-"
        "Drift (unbestätigt) → Kern als ein verschränkter Plan, --effort high; M2 Dateisystem offen → Zugriffs-Audit "
        "pro Run (rückwirkend 1.503 Transkripte, 0 Label-Zugriffe); M3 Leak-Zuordnung Val/Test → Train-Zuordnung; "
        "M4 exploited zusätzlich κ ≥ 0.2; M5 Transkripte verschwinden → Kopie pro Run", "—", f"{B}:89-94", "doku")
    S.f("Kanarien-Test 2 über die echte ClaudeCliBackend.run() (Git-Commit- + Memory-Kanarie): mit Sperren NONE/NONE, "
        "ohne Sperren beide genannt", "—", f"{B}:97-99", "doku")
    S.f("Fix-Commits im Harness-Repo", "d848587 (24.10. 15:30) Memory-Sperre; 9bb6a91 (24.10. 16:08) Review-Fixes",
        "git log Harness-Repo → `derived/timeline.json` _harness_repo_key_commits")
    ad = con["archived_dir_counts"]
    S.f("Archiviert als automem_* (Runs)", f"{con['automem_total_runs']} = Kern {con['automem_core_runs']} + Post-Cutoff "
        f"{con['automem_post_cutoff_runs']} + Leakage {ad['automem_leak-']} + Spurious {ad['automem_spur-']}; dazu "
        f"{ad['automem__']} Auswertungsordner", "runs-Repo Verzeichnisliste → `derived/contamination.json`")
    S.f("Archiviert als gitctx_* (erster Neu-Lauf Leakage D mit Git-Kontext)", f"{con['gitctx_dirs']} Ordner, davon "
        f"{con['gitctx_with_summary']} mit summary.json (unvollständig: {', '.join(con['gitctx_without_summary'])}); "
        f"{ad['gitctx__']} Auswertungsordner", "runs-Repo Verzeichnisliste → `derived/contamination.json`")
    S.f("Archiv-Commits runs-Repo", "4a17bf4 (24.10. 15:30) automem_*; c4131f5 (24.10. 16:08) gitctx_*",
        "git log runs-Repo → `derived/timeline.json` _runs_repo_git_log")
    S.f("Neu gefahren: alles — Kern A–D (auch C/D, da „vermutlich sauber“ nicht beweisbar und Claude-Code-Version "
        "inzwischen anders), Post-Cutoff, Leakage-Batches 1 + 2", "—", f"{B}:129-131", "doku")
    S.f("Dritter Kanal K-git (gefunden 27.10.): Worker ruft selbst git im runs-Repo auf", "2 Runs (student_health × A r5, "
        "× B r6), Commit-Nachrichten mit Gesamtzahlen gesehen, keine fremde Datei gelesen",
        "`derived/kgit.json`; " + f"{B}:104-113")
    S.f("K-git-Sperre (GIT_CEILING_DIRECTORIES) erst nach den Post-Cutoff-Runs", "Commit de857cd 27.10. 10:36, nach "
        "Abschluss-Commit 25644b7 (27.10. 10:26)", "git log Harness-/runs-Repo → `derived/timeline.json`")
    o = con["archived_comparison"]["automem__main"]
    op = con["archived_comparison"]["automem__post_cutoff"]
    S.f("BENANNTER VERGLEICH (archiviert): Friedman p (Iman-Davenport) Kern alt", f"{o['friedman_p_F']:.3f} (sauber 0.853)",
        "runs `automem__main/analysis/analysis.json` → `derived/contamination.json` archived_comparison", "roh (archiviert)")
    S.f("BENANNTER VERGLEICH (archiviert): mittlere Ränge Kern alt", arms_str(o["avg_ranks"], r2) +
        " (sauber A 2.27 · B 2.50 · C 2.50 · D 2.73)", "derived/contamination.json archived_comparison.automem__main.avg_ranks",
        "roh (archiviert)")
    S.f("BENANNTER VERGLEICH (archiviert): SUCCESS Kern alt (von 78)", arms_str(o["success"], str) + " (sauber 78 · 78 · 78 · 77)",
        "derived/contamination.json archived_comparison.automem__main.success", "roh (archiviert)")
    S.f("BENANNTER VERGLEICH (archiviert): Median der Run-SD Kern alt", arms_str(o["run_sd_median"], f4) +
        " (sauber A 0.0088 · B 0.0078 · C 0.0053 · D 0.0067) → „A stabiler“ fällt weg",
        "derived/contamination.json archived_comparison.automem__main.run_sd_median", "roh (archiviert)")
    cvo = o["cell_means"]["covertype"]
    S.f("BENANNTER VERGLEICH (archiviert): covertype-Zellmittel alt (Mechanismus 2 „Modell-Übergabe per Datei kostet A“)",
        arms_str(cvo) + " (sauber A 0.923 · B 0.919 · C 0.922 · D 0.921) → fällt weg",
        "derived/contamination.json archived_comparison.automem__main.cell_means", "roh (archiviert)")
    cno = o["cell_means"]["cnae_9"]
    S.f("BENANNTER VERGLEICH (archiviert): C vs. D alt auf covertype / cnae_9",
        f"{f3(cvo['crisp_react'])}/{f3(cvo['naked_react'])} · {f3(cno['crisp_react'])}/{f3(cno['naked_react'])} "
        "(sauber 0.922/0.921 · 0.956/0.957) → fällt weg", "derived/contamination.json archived_comparison", "roh (archiviert)")
    sbp = next(p for p in o["sig_better_pairs"] if p["a"] == "hard_fsm" and p["b"] == "crisp_react")
    sec_ = j("secondary_core.json")
    sbc = next(p for p in sec_["pairs"] if p["a"] == "hard_fsm" and p["b"] == "crisp_react")
    S.f("BENANNTER VERGLEICH (archiviert): „signifikant besser“ alt, A vs. C",
        f"A {o['sig_better']['hard_fsm']}/78 vs. C {o['sig_better']['crisp_react']}/78, Fisher p {pv(sbp['p'])} "
        f"(Holm über 6 Paare {pv(sbp['p_holm'])}) (sauber {sec_['per_arm']['hard_fsm']['sig_better']} vs. "
        f"{sec_['per_arm']['crisp_react']['sig_better']}, p {pv(sbc['p'])}, Holm {pv(sbc['p_holm'])} = derived/secondary_core.json "
        "pairs). Geplante Auswertung (experiment-design.md:360-361) ist Holm-korrigiert → in keinem der beiden Durchläufe "
        "signifikant [Holm sauber ergänzt 09.11., K5-06; nachgerechnet aus `automem_main-*`/`main-*` summary.json]",
        "runs `automem_main-*/summary.json` baseline_comparison.ci_lo → `derived/contamination.json`", "roh (archiviert)")
    S.f("BENANNTER VERGLEICH (archiviert): Kosten Kern alt", f"{usd(o['cost_total'])} (sauber 251.93 USD)",
        "runs `automem__main/main_runs.json` cost_total", "roh (archiviert)")
    S.f("BENANNTER VERGLEICH (archiviert): Post-Cutoff alt", f"SUCCESS {arms_str(op['success'], str)} von 12 (48/48), "
        f"{usd(op['cost_total'])}; sauber A 11 · B 12 · C 8 · D 6, 41.94 USD",
        "runs `automem__post_cutoff/analysis/analysis.json`, main_runs.json", "roh (archiviert)")
    S.f("Statustabelle alte Befunde: hält — kein Arm-Effekt (H2 im Hauptmaß nicht bestätigt), Rangfolge = Rauschen, "
        "Mechanismus 1 (DU-Vorsicht V12), kein F1-Vorteil durchs Benennen des Datasets", "—", "docs/roadmap.md:126-134", "doku")
    S.f("Statustabelle alte Befunde: fällt weg — „A stabiler“ (Run-SD), Mechanismus 2 (Modell-Übergabe per Datei), C ≫ D auf "
        "covertype, Post-Cutoff 48/48; abgeschwächt — A übertrifft die Baseline am seltensten; nur teilweise — Post-Cutoff ohne "
        "Memorisierungs-Effekt (nur A–C)", "—", "docs/roadmap.md:135-141", "doku")
    S.f("Deutung: das geteilte Gedächtnis wirkte im Kern eher zugunsten von A/B, post-cutoff zugunsten von C/D "
        "(Gate-Lektion balanced-weight-hurts-macrof1.md erreichte C/D)", "—", f"{B}:71-79; docs/roadmap.md:179-184", "doku")
    S.f("Gate-Lektion im Snapshot vorhanden", "balanced-weight-hurts-macrof1.md, student-health-f1-macro-hgb.md, "
        "ev-purchase-f1-threshold.md", "runs `_auto_memory_snapshot_2026-10-24/` → `derived/contamination.json` snapshot_names")
    S.f("VORBEHALT: Offene Restkanäle — Worker-Ordner liegen weiter im runs-Repo (M2 nur überwacht, nicht geschlossen); "
        "Timeout-Waisen unter Windows ungeprüft", "—", f"{B}:101-102", "doku")
    # F5-431 bis F5-435: Nachträge des Schreib-Agenten Kap. 5 (09.11.), hier uebernommen, damit eine Neugenerierung
    # sie nicht verliert. F5-431/433 stammen aus dem Transkript-Backup bzw. den summary-mtimes der archivierten Runs
    # (nur Zeitstempel, keine Ergebniswerte), die collect_facts.py bewusst nicht liest -> Wortlaut fest.
    S.f("Erster Durchlauf (archiviert, `automem_*`): Kern-Arme nacheinander, nicht verschränkt; Post-Cutoff mit 4 Armen "
        "überlappend",
        "Kern D + C am 23.09. (Commits f69d795 21:01 „D neu komplett“, f0d2c8d 23:31 „Arm C komplett“; Spannen laut Doku "
        "D 18:12–20:13, C 20:49–23:27, Transkripte gelöscht); A 24.09. – 25.09. 12:59 (letzte Transkript-Zeit; Commit "
        "9b91bae 13:13; Start laut Doku 24.09. 11:45, früheste erhaltene Transkript-Zeit 15:51); B 25.09. 13:20 – 26.09. "
        "17:49 (Transkripte; Commit 15df189 erst 04.10. 15:22); Post-Cutoff 04.10. 15:21–20:35, Spannen aller 4 Arme "
        "überlappen (Commit 3d32ff4 20:47)",
        "git log runs-Repo → `derived/timeline.json` _runs_repo_git_log; Transkript-Backup "
        "`C:/projects/thesis-crisp-dm-transcripts/backup-2026-10-24` (1.503 Dateien), zugeordnet über "
        "session_id/gate_session_id aus `automem_main-*/events.jsonl`; docs/befund-auto-memory.md:41-48",
        "roh (Commits, A-Ende, B, Post-Cutoff); doku (D/C-Spannen, A-Start)")
    S.f("K-git-Runs: Outcome; Post-Cutoff-Zählung A/B ohne sie",
        "student_health × A r5 SUCCESS (F1 0.902, Baseline 0.902), × B r6 SUCCESS (0.904); beide access_audit.flagged = "
        "false; ohne die 2 Runs Post-Cutoff A 10/11, B 11/11 SUCCESS",
        "runs `main-student_health-{hard,soft}_fsm-r*/summary.json`, `main-ev_purchase-{hard,soft}_fsm-r*/summary.json` "
        "(outcome, best_f1_measured, baseline_comparison.baseline_f1, access_audit)")
    S.f("Erster Durchlauf (archiviert): die beiden Leakage-Batches Arm D liefen am 24.10., also nach dem Post-Cutoff-Lauf "
        "vom 04.10.; der erste Durchlauf umfasst damit 23.09.–24.10. (Kern + Post-Cutoff bis 04.10., Leakage 24.10.)",
        "automem_leak-* summary-mtimes 24.10. 12:00–12:56, Commit 3b42e6e (12:57); automem_spur-* 14:14–15:14, Commit "
        "3d979f6 (15:20); Archiv-Commit 4a17bf4 (15:30)",
        "git log runs-Repo → `derived/timeline.json` _runs_repo_git_log; runs `automem_leak-*`/`automem_spur-*` summary.json mtime")
    S.f("Holm-Korrektur: im Plan nur für die Sekundär-Kennzahl „signifikant besser“ und die Leakage-Runs vorgesehen; bei der "
        "Auswertung auf den paarweisen Vergleich der Pass-Raten (Stufe 1, Kern + Post-Cutoff) ausgeweitet", "—",
        "docs/experiment-design.md:340-348 (§5.3 ohne Holm), 360-361 (§5.3b); docs/leakage-runs.md:88; "
        "thesis/chapters/03-research-design.tex:256", "doku")
    S.f("Modellwahl-Vorlage vor dem 14.09. sah ein Zweitmodell als Robustheits-Check vor (Sonnet 5); am 14.09. gestrichen → "
        "Abweichung vom ursprünglichen Plan, nicht vom Plan des 14.09. (K5-10)", "Sonnet 5",
        "docs/modellwahl.md:5, 103; docs/meetings/2026-09-14-entscheidungen.md:21", "doku")

    # ---------------------------------------------------------------- 5.3 Abweichungen
    S.sec("5.3 Abweichungen vom Plan", 5)
    S.f("Erfolgsregel: geplant Punkt-Schwelle k = 1 (14.09.) → ab 23.09. gepaarter Bootstrap: SUCCESS ⇔ obere Grenze des "
        f"95-%-KI der F1-Differenz Agent − Baseline ≥ 0 ({dc['config']['n_boot']} Replikate, Seed {dc['config']['boot_seed']})",
        "—", "src/crispfsm/scoring.py:17-19, 221-248; docs/experiment-design.md:111-117; docs/roadmap.md:350")
    S.f("Anlass der Umstellung: (1) Asymmetrie CV-Mittel vs. Einzel-Holdout, Baseline bestand ihre eigene Schwelle nicht "
        "(phoneme, cnae_9); (2) auf kleinen Holdouts wird k = 1 zum Münzwurf", f"wine: {sp['_main|wine']['n_test']} Testzeilen",
        "docs/roadmap.md:350; docs/experiment-design.md:119-126; `derived/splits.json` (n_test)", "doku")
    S.f("Testzeilen pro Kern-Set (20 % Test)", "; ".join(f"{ds} {sp['_main|' + ds]['n_test']}" for ds in CORE_ORDER),
        "runs `_main/logs/*.log` [setup] → `derived/splits.json`")
    S.f("Testzeilen Post-Cutoff", f"student_health {sp['_post_cutoff|student_health']['n_test']}; ev_purchase "
        f"{sp['_post_cutoff|ev_purchase']['n_test']}", "runs `_post_cutoff/logs/*.log` → `derived/splits.json`")
    S.f("Kern-Neustart Nr. 1 (23.09.): alle Kern-Runs davor archiviert (Design-Änderungen 23.09.)",
        f"{ad['promptv1_main-']} Runs promptv1_* (D {con['promptv1_by_arm']['naked_react']}, C "
        f"{con['promptv1_by_arm']['crisp_react']}, A {con['promptv1_by_arm']['hard_fsm']})",
        "runs-Repo Verzeichnisliste → `derived/contamination.json` promptv1_by_arm; docs/roadmap.md:9-15")
    S.f("Offline-Neubewertung (Re-Score) nach der Bootstrap-Regel nur für archivierte promptv1-D-Runs (49/78 → 74/78); "
        "im sauberen Datenstand 0 Runs rescored", "0 rescored", "docs/roadmap.md:351; runs `_*/main_runs.json` rescored → "
        "`derived/design_checks.json` aggregates")
    S.f("Kern-Neustart Nr. 2 (24.10.)", "s. Abschnitt 5.2b", "docs/befund-auto-memory.md", "doku")
    S.f("Zweitmodell entfallen (14.09.)", "−120 Runs, −~60 USD laut Protokoll", "docs/meetings/2026-09-14-entscheidungen.md:21", "doku")
    S.f("H3 (MLflow-Memory) und H4 (LLM- vs. klassisches AutoML) nicht implementiert → Future Work", "—",
        "docs/outline.md:12-14; thesis/chapters/03-research-design.tex:90", "doku")
    LR = "docs/leakage-runs.md"
    S.f("Leakage-Plan ursprünglich 4 Arme × 13 Sets × n = 3", "156 Runs, ~150 USD", f"{LR}:25-33", "doku")
    S.f("Leakage n = 3 statt 6 (Ausnutzung pro Run nahezu binär; Laufzeit)", "—", f"{LR}:42-44", "doku")
    S.f("Vorab festgelegte Weiter-Regel: D zuerst; nutzt D den Leak in ≥ 8/39 Runs → C, A, B nachziehen, sonst Batch beenden",
        "Schwelle 8/39", f"{LR}:74-76, 125-126", "doku")
    S.f("Ergebnis der Regel (sauber)", f"perfekt {lk['perfect']['exploited']}/39, spurious {lk['spurious']['exploited']}/39 "
        "→ beide unter 8/39 → A/B/C nicht gefahren, Leakage-Experimente beendet", "derived/leakage.json *.exploited, rule_triggered")
    S.f("Spurious-Variante (fachfremd-absurder Name, ~85 % statt perfekter Korrelation) auf Deckers Anregung 09.10.",
        "—", "docs/meetings/2026-10-09-kern-ergebnisse.md:21; " + f"{LR}:95-108", "doku")
    S.f("Leakage-Batches liefen zeitlich vor dem Kern und nur mit D (nicht mit den Armen verschränkt)", "—",
        f"`derived/timeline.json`; {LR}:148-149")
    S.f("Leakage perfekt lief auf Harness 9bb6a91 statt 14e99e4 (code-identisch, Diff nur Doku)", "—", "`derived/timeline.json`")
    S.f("Robustheits-Set (vergiftete/unlösbare Aufgabenpaare, Erwartung ABSTAIN) nicht gefahren; Mechanik existiert "
        f"(poisoned_variant in {dc['manifests_with_poisoned_variant']}/{dc['manifests_total']} Manifesten)", "0 Runs",
        "docs/roadmap.md:335, 369; manifests/*.yaml → `derived/design_checks.json`")
    S.f("K-git-Sperre erst nach den Post-Cutoff-Runs (Post-Cutoff liefen ohne sie; 2 Runs betroffen, folgenlos)", "—",
        "`derived/kgit.json`; docs/roadmap.md:197-204")
    S.f("Hold-out-Baselines des Endurteils (Test, Referenzmodell auf Validierung gewählt) pro Kern-Set — ersetzen die "
        "80/20-Werte vor dem 23.09.", "; ".join(f"{ds} {j('core.json')['baseline_test'][ds]:.3f}" for ds in CORE_ORDER),
        "manifests/*.yaml baseline_f1 = summary.baseline_comparison.baseline_f1 → `derived/core.json` baseline_test")

    # ---------------------------------------------------------------- 5.4 zweite Stufe (H1)
    h = j("h1.json")
    S.sec("5.4 Zweite Stufe (H1): Ablauf, Kontrollen, Kosten (13.–15.11.)", 6)
    HS = "runs `h1-*/summary.json`, Transkripte → `derived/h1.json`"
    sp_ = h["span"]
    S.f("Zeitraum Post-Cutoff-Runs der zweiten Stufe (erste bis letzte Transkript-Zeit)",
        f"{sp_['h1_post']['first_start_local']} – {sp_['h1_post']['last_end_local']}", f"{HS} span.h1_post")
    S.f("Zeitraum Kern-Runs der zweiten Stufe (erste bis letzte Transkript-Zeit)",
        f"{sp_['h1_core']['first_start_local']} – {sp_['h1_core']['last_end_local']}", f"{HS} span.h1_core")
    S.f("Anzahl gemessener Runs (Kern 2 × 13 × 6, Post-Cutoff 2 × 2 × 6)",
        f"Kern {h['n_runs']['h1_core']} · Post-Cutoff {h['n_runs']['h1_post']}", f"{HS} n_runs")
    hc = h["costs"]
    S.f("Kosten der zweiten Stufe (Listenpreise)",
        f"gesamt {usd(h['cost_total_all_h1'])}; Kern {usd(hc['h1_core']['hard_fsm']['cost_total'] + hc['h1_core']['hard_fsm_single']['cost_total'])} "
        f"(A {usd(hc['h1_core']['hard_fsm']['cost_total'])}, A-Single {usd(hc['h1_core']['hard_fsm_single']['cost_total'])}); "
        f"Post-Cutoff {usd(hc['h1_post']['hard_fsm']['cost_total'] + hc['h1_post']['hard_fsm_single']['cost_total'])} "
        f"(A {usd(hc['h1_post']['hard_fsm']['cost_total'])}, A-Single {usd(hc['h1_post']['hard_fsm_single']['cost_total'])})",
        f"{HS} costs, cost_total_all_h1")
    for b, lab in (("h1_core", "Kern"), ("h1_post", "Post-Cutoff")):
        c = h["controls"][b]
        n = c["n_runs"]
        cv = [k.split(":", 1)[1] for k in c if k.startswith("claude_version:")]
        ps = [k.split(":", 1)[1] for k in c if k.startswith("prompt_sha256:")]
        S.f(f"Kontrollen {lab}: Hold-out unverändert, Transkript je Session, Auto-Memory und Git-Kontext aus, "
            "sauberer Harness-Stand, eine Client-Version, ein Prompt-Hash, Erfolgsregel, Zielspalte",
            f"holdout {c['holdout_intact']}/{n}; Transkripte {c['transcripts_match_sessions']}/{n} (nicht leer {c['transcripts_nonempty']}); "
            f"Auto-Memory aus {c['auto_memory_disabled']}/{n}; Git-Kontext aus {c['git_context_disabled']}/{n}; dirty=false {c['dirty_false']}/{n}; "
            f"Client {', '.join(cv)}; Prompt-SHA {', '.join(ps)}; noninferior_paired {c.get('success_rule:noninferior_paired', 0)}/{n}; "
            f"Zielspalte korrekt {c['bu_or_target_ok']}/{n}; abstained {c['abstained']}; budget_exhausted {c['budget_exhausted']}",
            f"{HS} controls.{b}")
    hcc = h["harness_code_changes_vs_0b1ccca"]
    S.f("Harness-Stand: alle Runs auf Code-Stand 0b1ccca (A-Single eingeführt 13.11.); spätere Commits während des Batches "
        "ändern außerhalb von thesis/ und docs/ nur .gitignore und CLAUDE.md des Harness-Repos (liegt nicht im Arbeitsordner der Worker)",
        "; ".join(f"{k}: {', '.join(v) if v else '—'}" for k, v in hcc.items()),
        f"{HS} harness_code_changes_vs_0b1ccca (git diff --name-only 0b1ccca <commit> -- . ':!thesis' ':!docs')")
    S.f("Modell in allen Assistant-Nachrichten der Transkripte (Kern + Post-Cutoff)",
        "; ".join(f"{m} {n}" for m, n in h["models_in_transcripts"].items()), f"{HS} models_in_transcripts")
    af = h["access_flagged"]
    S.f("Zugriffs-Audit: markierte Runs; jeder Treffer liest die eigene ausgelagerte Tool-Ausgabe des Runs "
        "(eigener Projektordner des Clients, eigene Session, Unterordner tool-results); keine Hold-out-Labels, keine anderen Runs",
        f"{len(af)} von {h['n_runs']['h1_core'] + h['n_runs']['h1_post']}: "
        + "; ".join(f"{x['run_id']} ({', '.join(f'{k} {v}' for k, v in x['counts'].items())}, eigene Ausgabe: {'ja' if x['all_own_tool_output'] else 'NEIN'})" for x in af),
        f"{HS} access_flagged; summary.access_audit.examples")
    S.f("Sessions pro Run (A: eine je Phase; A-Single: eine für alle Phasen)",
        "; ".join(f"{lab_} A {hc[b]['hard_fsm']['sessions_mean']:.1f} (max {hc[b]['hard_fsm']['sessions_max']}) · "
                  f"A-Single {hc[b]['hard_fsm_single']['sessions_mean']:.1f} (max {hc[b]['hard_fsm_single']['sessions_max']})"
                  for b, lab_ in (("h1_core", "Kern"), ("h1_post", "Post-Cutoff"))),
        f"{HS} costs.*.*.sessions_mean")
    S.f("NICHT IM TEXT VERWENDEN (Johannes 15.11.: PC aus/Neustart ist irrelevant) — Unterbrechungen des Kern-Batches; "
        "laufender Run jeweils komplett wiederholt (run.py löscht ungemessene Ordner)",
        "13.11. nach 2 Kern-Runs Rechner heruntergefahren; 14.11. ca. 17:25 Batch-Prozess von außen beendet (Neustart 17:36); "
        "15.11. 01:03 Rechner heruntergefahren (Neustart 09:10)", "docs/h1-single-session.md §6 (Laufprotokoll)", "doku")
    cp = h["compaction"]
    S.f("Kontext-Kompaktierung des Clients (Marker compact_boundary in den Transkripten)",
        f"{len(cp['runs_with_compaction'])} von {cp['runs_checked']} Runs", f"{HS} compaction")
    kg = h["kgit"]
    S.f("Worker-Bash-Aufrufe von git (Scan aller Transkripte der zweiten Stufe, wie F5-328)",
        f"{len(kg['runs_with_git_calls'])} von {kg['scanned_runs']} Runs"
        + ("" if not kg["runs_with_git_calls"] else ": " + "; ".join(x["run_id"] for x in kg["runs_with_git_calls"])),
        f"{HS} kgit")
    return S


# =========================================================================== Kap. 6

def kap6() -> Sheet:
    core, sec, rsd, ter = j("core.json"), j("secondary_core.json"), j("runsd_core.json"), j("terciles.json")
    pc, ph, du, lk, co = j("postcutoff.json"), j("phases.json"), j("du_exclusions.json"), j("leakage.json"), j("costs.json")
    med, ex, sp, ctl = j("median.json"), j("expectations.json"), j("splits.json"), j("controls.json")
    cfg = j("design_checks.json")["config"]
    S = Sheet(6, "Faktenblatt Kapitel 6 — Results", INTRO)
    SRC = "runs `main-<Kern-Set>-*/summary.json`"
    l1 = core["level1"]

    # ---------------------------------------------------------------- 6.1
    S.sec("6.1 Stufe 1: Bestehensraten (Kern, N = 13, n = 6)", 1)
    S.f("Runs pro Arm im Kern", "78 (13 Sets × 6); gesamt 312", f"{SRC} → `derived/core.json` level1.per_arm.*.n")
    for a in ARMS:
        x = l1["per_arm"][a]
        S.f(f"SUCCESS {FULL[a]} (Rate, Wilson-95-%-KI)", f"{x['success']}/{x['n']} = {f3(x['rate'])}, {wil(x['wilson'])}",
            f"{SRC} success → `derived/core.json` level1.per_arm.{a}")
    S.f("Fisher exakt paarweise auf gepoolten SUCCESS-Zählungen (6 Paare)",
        "alle p = 1.000 (Holm 1.000)", "derived/core.json level1.pairs")
    fail = [f for c in l1["per_cell"] for f in c["fails"]]
    fr = fail[0]
    S.f("Einziger Nicht-SUCCESS im Kern", f"{fr['run_id']}: below_threshold, F1 {f3(fr['f1'])}, Δ zur Baseline {s3(fr['diff'])}, "
        f"obere KI-Grenze {s3(fr['ci_hi'])}", "derived/core.json level1.per_cell[*].fails")
    S.f("fail_reason-Profil Kern: keine form_error, abstain, spec_mismatch, gate_failed, safety_max_calls, budget_exhausted",
        "1× below_threshold (D)", "summary.fail_reason → `derived/core.json` level1.per_arm.*.fail_reasons; `derived/controls.json`")
    bmd = next(c for c in l1["per_cell"] if c["dataset"] == "bank_marketing" and c["arm"] == "naked_react")
    S.f("Zellen mit weniger als 6/6 SUCCESS", f"nur bank_marketing × D (5/6, Wilson {wil(bmd['wilson'])})", "derived/core.json level1.per_cell")
    S.f("Korrekte Zielspalte (target_ok) im Kern", f"{ctl['core']['bu_or_target_ok']}/312", "summary.target_ok → `derived/controls.json`")
    S.f("Sekundärmaß „signifikant besser als die Baseline“ (baseline_comparison.ci_lo > 0) pro Arm",
        "; ".join(f"{L[a]} {sec['per_arm'][a]['sig_better']}/78 = {f3(sec['per_arm'][a]['rate'])} {wil(sec['per_arm'][a]['wilson'])}"
                  for a in ARMS), f"{SRC} baseline_comparison.ci_lo → `derived/secondary_core.json` per_arm")
    S.f("Sekundärmaß, Fisher paarweise (roh / Holm über 6 Paare)",
        "; ".join(f"{L[p['a']]}–{L[p['b']]} {pv(p['p'])} / {pv(p['p_holm'])}" for p in sec["pairs"]) + " — alle n. s.",
        "derived/secondary_core.json pairs")
    S.f("Sekundärmaß pro Set (A/B/C/D, von je 6)",
        "; ".join(f"{c['dataset']} " + "/".join(str(c[a]) for a in ARMS) for c in sec["per_cell"]),
        "derived/secondary_core.json per_cell")
    z = sec["cells_with_zero"]
    S.f("Sets, auf denen A in 0/6 Runs signifikant besser ist", f"{len(z['hard_fsm'])}: {', '.join(z['hard_fsm'])}; davon mit "
        "> 0 bei anderen Armen: bank_marketing (C 5, D 5), phoneme (C 3, D 3), credit_g (B 1, D 1)", "derived/secondary_core.json")
    S.f("VORBEHALT Sekundärmaß: festgelegt 23.09. (vor A/B; im sauberen Kern vor allen 4 Armen), sekundär und "
        "beschreibend; Teststärke hängt an der Test-Größe (wine 36 Zeilen, breast_cancer 114)", "—",
        "docs/experiment-design.md:350-380; `derived/splits.json`", "doku")
    S.f("VORBEHALT Stufe 1 im Kern: Deckeneffekt — 311/312 bestanden, Stufe 1 trennt die Arme im Kern nicht",
        "311/312", "derived/core.json level1")
    # Nachtrag Review 23.11.2026 (vorher von Hand im Faktenblatt; jetzt generiert, damit Anhang B.5 es reproduziert)
    pr = j("point_rule.json")
    pa = pr["per_arm"]
    pairs = "; ".join(f"{L[x['pair'].split('-')[0]]}–{L[x['pair'].split('-')[1]]} {x['p']:.3f} / {x['p_holm']:.3f}"
                      for x in pr["fisher_holm"])
    hmin = min(x["p_holm"] for x in pr["fisher_holm"])
    gaps = []
    for s in ("bank_marketing", "credit_g"):
        c = pr["per_set"][s]
        gaps.append(f"{s} (A {c['hard_fsm']['point']}/6, B {c['soft_fsm']['point']}/6, C {c['crisp_react']['point']}/6, "
                    f"D {c['naked_react']['point']}/6)")
    S.f("POST HOC Sensitivität Stufe 1 (Review 23.11.): Pass-Raten im Kern unter Punktvergleich Δ ≥ 0 auf denselben "
        "Testzeilen (ungerundetes baseline_comparison.diff, formale Prüfungen wie berichtet); Fisher paarweise + Holm "
        "über 6 Paare",
        " · ".join(f"{L[a]} {pa[a]['point']}" for a in ARMS) + f" von 78; Fisher roh / Holm: {pairs}; "
        f"Holm min {hmin:.3f} → kein Paar signifikant; größte Abstände {gaps[0]} und {gaps[1]}",
        f"{SRC} → `derived/point_rule.json` (scripts/point_rule_sensitivity.py)")

    # ---------------------------------------------------------------- 6.2
    S.sec("6.2 Stufe 2: Modellqualität (Friedman, Nemenyi, Cliff's δ)", 2)
    for key, label in (("f1_measured", "Primär best_f1_measured"), ("f1_combined", "Sensitivität kombinierter Score (Fail = 0)"),
                       ("f1_on_success", "Ergänzend F1 nur auf SUCCESS-Runs")):
        r = core["level2"][key]
        fr_ = r["friedman"]
        S.f(f"{label}: Friedman χ², Iman-Davenport F, N, k, CD",
            f"χ² = {fr_['chi2']:.2f} (df {fr_['k'] - 1}), p = {pv(fr_['p_chi2'])}; F({fr_['df'][0]}, {fr_['df'][1]}) = {fr_['F']:.2f}, "
            f"p = {pv(fr_['p_F'])}; N = {fr_['N']}, k = {fr_['k']}, CD = {r['cd']:.2f}",
            f"{SRC} best_f1_measured/final_f1 → `derived/core.json` level2.{key}.friedman (= runs `_main/analysis/analysis.json`)")
        S.f(f"{label}: mittlere Ränge (1 = bester)", arms_str(r["avg_ranks"], r2), f"derived/core.json level2.{key}.avg_ranks")
        mx = max(p["rank_diff"] for p in r["nemenyi"])
        S.f(f"{label}: Nemenyi", f"kein Paar signifikant (größte Rangdifferenz {mx:.2f} < CD {r['cd']:.2f}); eine Clique mit allen vier Armen",
            f"derived/core.json level2.{key}.nemenyi, cliques")
    r = core["level2"]["f1_measured"]
    for c in r["cliffs"]:
        nem = next(p for p in r["nemenyi"] if p["a"] == c["a"] and p["b"] == c["b"])
        S.f(f"Paar {L[c['a']]} vs. {L[c['b']]} (Primärsicht): Rangdifferenz, Nemenyi, Cliff's δ (Mittel über 13 Sets), große Effekte",
            f"Δ Rang {nem['rank_diff']:.2f}, n. s.; δ = {c['mean_delta']:+.3f} ({c['mean_magnitude']}); Sets mit großem Effekt "
            f"{L[c['a']]} > {L[c['b']]}: {c['a_better_large']}, {L[c['b']]} > {L[c['a']]}: {c['b_better_large']}",
            "derived/core.json level2.f1_measured.nemenyi, cliffs (per_dataset dort)")
    ros = core["level2"]["f1_on_success"]["cliffs"]
    S.f("Cliff's δ in der Sicht „nur SUCCESS“ (weicht nur bei Paaren mit D ab)",
        "; ".join(f"{L[c['a']]}–{L[c['b']]} {c['mean_delta']:+.3f}" for c in ros if c["b"] == "naked_react"),
        "derived/core.json level2.f1_on_success.cliffs")
    mc = med["median_core"]["f1_measured"]
    S.f("Median-Sensitivität (Zellwert = Median statt Mittel), Primärsicht",
        f"χ² = {mc['friedman']['chi2']:.2f}, p = {pv(mc['friedman']['p_chi2'])}; F({mc['friedman']['df'][0]}, {mc['friedman']['df'][1]}) = "
        f"{mc['friedman']['F']:.2f}, p = {pv(mc['friedman']['p_F'])}; Ränge {arms_str(mc['avg_ranks'], r2)}; "
        f"Nemenyi kein Paar signifikant", f"`{med['median_core']['_command']}` → `derived/median_core/analysis.json`, `derived/median.json`")
    # F6-218 korrigiert 09.11. (K6-01): der Fail aendert den Zellmedian in der Sicht "nur SUCCESS", nicht die Raenge.
    mm = {k: med["median_core"][k] for k in ("f1_measured", "f1_combined", "f1_on_success")}
    bi = mm["f1_measured"]["datasets"].index("bank_marketing")
    bmed = {k: v["matrix"][bi][ARMS.index("naked_react")] for k, v in mm.items()}
    assert all(v["avg_ranks"] == mm["f1_measured"]["avg_ranks"] for v in mm.values())
    S.f("Median-Sensitivität: alle drei F1-Sichten ergeben dieselben Ränge und Teststatistiken. KORRIGIERT 09.11. (K6-01): "
        "Der eine Fail ändert den Zellmedian von bank_marketing × D sehr wohl, aber nur in der Sicht „nur SUCCESS“; der Rang "
        "(2, hinter C) bleibt",
        f"Ränge/Tests identisch (= F6-217); Median bank_marketing × D: primär {f3(bmed['f1_measured'])}, kombiniert "
        f"{f3(bmed['f1_combined'])}, nur SUCCESS {f3(bmed['f1_on_success'])}; C "
        f"{f3(mm['f1_measured']['matrix'][bi][ARMS.index('crisp_react')])}",
        "derived/median.json median_core.*.matrix (Zeile bank_marketing); runs "
        "`main-bank_marketing-naked_react-r1..r6/summary.json` best_f1_measured")
    means, base, ter_a = core["cell_means_f1_measured"], core["baseline_test"], ter["assignment"]
    for ds in CORE_ORDER:
        S.f(f"Zellmittel F1 (best_f1_measured, n = 6) {ds} — Terzil {ter_a[ds]}, Baseline (Test)",
            arms_str(means[ds]) + f"; Baseline {f3(base[ds])}",
            f"{SRC} → `derived/core.json` cell_means_f1_measured.{ds}, baseline_test.{ds}")
    comb = core["cell_means_f1_combined"]["bank_marketing"]["naked_react"]
    succ = core["cell_means_f1_on_success"]["bank_marketing"]["naked_react"]
    S.f("Abweichende Zellwerte der Sensitivitätssichten (nur bank_marketing × D)",
        f"kombiniert {f3(comb)}, nur SUCCESS {f3(succ)} (Primär {f3(means['bank_marketing']['naked_react'])})",
        "derived/core.json cell_means_f1_combined, cell_means_f1_on_success")
    b3 = core["sets_where_arm_best_3dp"]
    S.f("Anzahl Sets, auf denen ein Arm das höchste Zellmittel hat (auf 3 Nachkommastellen, Gleichstände zählen für alle)",
        arms_str(b3, str) + " (Gleichstände: cnae_9 A = D, wine alle vier)", "derived/core.json best_arm_per_set_3dp")
    sp_ = core["max_cell_spread_per_set"]
    small = [ds for ds in CORE_ORDER if sp_[ds] <= 0.01]
    S.f("Spannweite der vier Zellmittel pro Set (max − min)",
        f"≤ 0.010 auf {len(small)} Sets ({', '.join(small)}); größte: bank_marketing {f3(sp_['bank_marketing'])}, credit_g "
        f"{f3(sp_['credit_g'])}, eucalyptus {f3(sp_['eucalyptus'])}", "derived/core.json max_cell_spread_per_set")
    mdb = core["median_diff_to_baseline"]
    S.f("Median über alle Runs der F1-Differenz Agent − Baseline (Kern)", arms_str({a: mdb[a]["median_diff"] for a in ARMS}, s3),
        f"{SRC} baseline_comparison.diff → `derived/core.json` median_diff_to_baseline")
    S.f("wine: alle vier Zellmittel = Baseline (36 Testzeilen)", f"{f3(means['wine']['hard_fsm'])} = {f3(base['wine'])}",
        "derived/core.json cell_means_f1_measured.wine, baseline_test.wine")
    S.f("VORBEHALT: best_f1_measured = Test-F1 der final abgegebenen Predictions (nicht Bestwert über Gate-Versuche)", "—",
        "docs/decisions-2026-09-23.md:43-44", "doku")
    S.f("VORBEHALT: CD = 1.30 bei N = 13 — nur große Rangunterschiede auflösbar; nicht signifikant ≠ äquivalent", "—",
        "derived/core.json level2.f1_measured.cd")
    # Nachtrag Schreib-Agent Kap. 6 (09.11.)
    S.f("Signifikanzniveau für Friedman und Nemenyi (Grundlage der vorab festgelegten CD = 1.30)", "α = 0.05",
        "runs `_main/analysis/analysis.json` alpha (= `_post_cutoff/analysis/analysis.json`); `scripts/analyze_main.py:60` "
        "ALPHA; docs/meetings/2026-09-14-entscheidungen.md:19")
    # Nachtrag externes Feedback 24.11.2026 (vorher von Hand; jetzt generiert)
    ei = j("effect_intervals.json")
    d = ei["diffs"]
    raw = "; ".join(f"{L[a]}−D {d[a]['mean']:+.4f} [{d[a]['ci_lo']:+.4f}, {d[a]['ci_hi']:+.4f}]"
                    for a in ("hard_fsm", "soft_fsm", "crisp_react"))
    rnd = ", ".join(f"{L[a]} {d[a]['mean']:+.3f} [{d[a]['ci_lo']:+.3f}, {d[a]['ci_hi']:+.3f}]"
                    for a in ("hard_fsm", "soft_fsm", "crisp_react"))
    S.f(f"POST HOC Effektgröße (externes Feedback 24.11.): mittlere Differenz der Zellmittelwerte (Primärsicht) gegen D "
        f"über {ei['n_datasets']} Sets, 95-%-t-Intervall (df {ei['n_datasets'] - 1}, t = {ei['t_crit']:.3f}); deskriptiv, "
        "kein geplanter Äquivalenztest",
        f"{raw} → gerundet {rnd}; alle innerhalb ±0.05 (kleinste relevante Differenz, 3.6)",
        "derived/core.json cell_means_f1_measured → `derived/effect_intervals.json` (scripts/effect_intervals.py)")

    # ---------------------------------------------------------------- 6.3
    S.sec("6.3 Stabilität: Run-SD", 3)
    pa = rsd["per_arm"]
    S.f("Median der Run-SD über 13 Zellen pro Arm (best_f1_measured, n = 6)", arms_str({a: pa[a]["median_sd"] for a in ARMS}, f4),
        f"{SRC} → `derived/runsd_core.json` per_arm (= runs `_main/analysis/analysis.json` run_sd)")
    S.f("Maximale Run-SD pro Arm (Set)", "; ".join(f"{L[a]} {f4(pa[a]['max_sd'])} ({pa[a]['max_sd_dataset']})" for a in ARMS),
        "derived/runsd_core.json per_arm")
    S.f("Anzahl Sets, auf denen A eine kleinere Run-SD hat als D", f"{ex['p3_runsd_A_lt_D']['core_cells_A_lt_D']} von 13",
        "derived/expectations.json p3_runsd_A_lt_D")
    lg = rsd["arm_with_largest_sd_count"]
    S.f("Anzahl Sets, auf denen ein Arm allein die größte Run-SD hat (wine: Gleichstand, alle 0)",
        arms_str({a: lg.get(a, 0) for a in ARMS}, str), "derived/runsd_core.json arm_with_largest_sd_count")
    zr = rsd["zero_sd_cells"]
    S.f("Zellen mit Run-SD = 0 (alle 6 Runs identischer F1)", "; ".join(f"{L[a]}: {', '.join(zr[a])}" for a in ARMS),
        "derived/runsd_core.json zero_sd_cells")
    for ds in CORE_ORDER:
        S.f(f"Run-SD {ds}", arms_str(rsd["per_cell"][ds], f4), f"derived/runsd_core.json per_cell.{ds}")
    S.f("VORBEHALT: Run-SD ist deskriptiv; kein Test vorab festgelegt (Empfehlung Roadmap: nur deskriptiv); SD aus n = 6 (df 5)",
        "—", "docs/roadmap.md:445", "doku")
    # Nachtrag Schreib-Agent Kap. 6 (09.11.); Einzelwerte aus den Summaries, nicht in derived/
    S.f("Run-SD fashion_mnist × A (Hard FSM) ungerundet: keine Null-Zelle im Sinne von F6-305, nur auf 4 Stellen 0.0000",
        "0.00003; 4 von 6 Runs F1 0.89501, dazu 0.89501 (0.895007) und 0.89493",
        "runs `main-fashion_mnist-hard_fsm-r*/summary.json` best_f1_measured → `derived/runsd_core.json` "
        "per_cell.fashion_mnist.hard_fsm")
    # Nachtrag 15.11.: Nachpruefung der Wiederholungszahl aus Kap. 3.6 (SE <= 0.0167 bei n = 6)
    lim = 0.0167 * 6 ** 0.5
    over = sorted(((a, ds, v) for ds, row in rsd["per_cell"].items() for a, v in row.items() if v is not None and v > lim),
                  key=lambda t: -t[2])
    n_cells = sum(1 for row in rsd["per_cell"].values() for v in row.values() if v is not None)
    S.f(f"Zellen über der Run-SD-Grenze der Wiederholungszahl (Kap. 3.6: SE ≤ 0.0167 bei n = 6 ⇔ SD ≤ 0.0167·√6 = {f4(lim)})",
        f"{len(over)} von {n_cells}: " + "; ".join(f"{L[a]} {ds} {f4(v)} (SE {f4(v / 6 ** 0.5)})" for a, ds, v in over)
        + f"; übrige {n_cells - len(over)} darunter", "`derived/runsd_core.json` per_cell, Grenze nachgerechnet")

    # ---------------------------------------------------------------- 6.4
    S.sec("6.4 Moderator Aufgabenschwere (Terzile)", 4)
    S.f("Terzil-Zuordnung nach best-Baseline-macro-F1 (5-fold-CV, full-size), absteigend sortiert, 4/4/5",
        "; ".join(f"{t}: " + ", ".join(f"{ds} ({f3(ter['baseline_cv'][ds])})" for ds in ter["per_tercile"][t]["datasets"])
                  for t in ("easy", "medium", "hard")),
        "manifests/*.yaml baseline_f1_cv → `derived/terciles.json`; deckt sich mit docs/meetings/2026-09-14-vorbereitung.md:278-292 "
        "und thesis/chapters/03-research-design.tex:138-150")
    for t in ("easy", "medium", "hard"):
        pt = ter["per_tercile"][t]["arms"]
        S.f(f"Terzil {t}: mittlerer Rang (Ränge je Set aus Zellmitteln, gemittelt über die Sets des Terzils)",
            arms_str({a: pt[a]["mean_rank"] for a in ARMS}, r2), f"derived/terciles.json per_tercile.{t}")
        S.f(f"Terzil {t}: mittleres Zellmittel F1 minus Baseline (Test)",
            arms_str({a: pt[a]["mean_f1_minus_baseline"] for a in ARMS}, s3), f"derived/terciles.json per_tercile.{t}")
        S.f(f"Terzil {t}: SUCCESS / signifikant besser (von n)",
            "; ".join(f"{L[a]} {pt[a]['success']}/{pt[a]['n']} / {pt[a]['sig_better']}" for a in ARMS),
            f"derived/terciles.json per_tercile.{t}")
    S.f("VORBEHALT: Moderator rein deskriptiv (4/4/5 Sets je Terzil, kein Test); Verfeinerung des Erwartungs-Musters "
        "nur qualitativ prüfbar", "—", "thesis/chapters/03-research-design.tex:208", "doku")

    # ---------------------------------------------------------------- 6.5
    S.sec("6.5 Post-Cutoff-Kontrolle (student_health, ev_purchase)", 5)
    P = "runs `main-student_health|ev_purchase-*/summary.json`"
    S.f("Umfang", "2 Sets × 4 Arme × n = 6 = 48 Runs, Harness 14e99e4", f"{P} → `derived/controls.json` post_cutoff")
    for a in ARMS:
        x = pc["level1"]["per_arm"][a]
        S.f(f"SUCCESS {FULL[a]} post-cutoff (Rate, Wilson)", f"{x['success']}/{x['n']} = {f3(x['rate'])}, {wil(x['wilson'])}",
            f"{P} → `derived/postcutoff.json` level1.per_arm.{a}")
    for ds in POST_ORDER:
        row = {c["arm"]: c["success"] for c in pc["level1"]["per_cell"] if c["dataset"] == ds}
        S.f(f"SUCCESS pro Arm auf {ds} (von 6)", arms_str(row, str), "derived/postcutoff.json level1.per_cell")
    S.f("Fisher exakt paarweise post-cutoff (roh / Holm über 6 Paare)",
        "; ".join(f"{L[p['a']]}–{L[p['b']]} {pv(p['p'])} / {pv(p['p_holm'])}" for p in pc["level1"]["pairs"]) + " — nach Holm alle n. s.",
        "derived/postcutoff.json level1.pairs")
    po = pc["pooled_fsm_vs_react"]
    S.f("POST HOC gruppiert (nach Sicht der Daten, nur explorativ): A+B vs. C+D",
        f"{po['AB'][0]}/{po['AB'][1]} vs. {po['CD'][0]}/{po['CD'][1]}, Fisher p = {pv(po['p'])}, OR = {po['odds_ratio']:.1f}",
        "derived/postcutoff.json pooled_fsm_vs_react")
    S.f("Alle Nicht-SUCCESS post-cutoff", f"{len(pc['fails'])} Runs, alle below_threshold; C+D: 10, davon 8 auf student_health",
        "derived/postcutoff.json fails, fails_by_arm_set")
    fl = pc["fails"]
    cd_sh = [f["diff"] for f in fl if f["dataset"] == "student_health" and f["arm"] in ("crisp_react", "naked_react")]
    d_ev = [f["diff"] for f in fl if f["dataset"] == "ev_purchase"]
    a4 = next(f for f in fl if f["arm"] == "hard_fsm")
    S.f("Abstand der Fails zur Baseline (F1-Differenz)", f"C/D student_health {s3(max(cd_sh))} bis {s3(min(cd_sh))}; D ev_purchase "
        f"{s3(d_ev[0])} (2×); einziger A-Fail {a4['run_id']} {s3(a4['diff'])}", "derived/postcutoff.json fails")
    for ds in POST_ORDER:
        S.f(f"Zellmittel F1 {ds} (n = 6) und Baseline (Test)", arms_str(pc["f1_cell_means"][ds]) + f"; Baseline {f3(pc['baseline_test'][ds])}",
            f"{P} → `derived/postcutoff.json` f1_cell_means, baseline_test")
        S.f(f"Zellmittel minus Baseline {ds}", arms_str(pc["f1_cell_minus_baseline"][ds], s3), "derived/postcutoff.json f1_cell_minus_baseline")
        S.f(f"Run-SD {ds}", arms_str(pc["run_sd"]["per_cell"][ds], f4), "derived/postcutoff.json run_sd.per_cell")
    S.f("Median der Run-SD post-cutoff (2 Zellen)", arms_str({a: pc["run_sd"]["per_arm"][a]["median_sd"] for a in ARMS}, f4),
        "derived/postcutoff.json run_sd.per_arm")
    fpc = pc["friedman_f1_measured"]
    S.f("Friedman post-cutoff (N = 2, ohne Aussagekraft)", f"χ² = {fpc['friedman']['chi2']:.2f}, p = {pv(fpc['friedman']['p_chi2'])}; "
        f"F(3, 3) = {fpc['friedman']['F']:.2f}, p = {pv(fpc['friedman']['p_F'])}; CD = {fpc['cd']:.2f}; Ränge "
        f"{arms_str(fpc['avg_ranks'], r2)}", "derived/postcutoff.json friedman_f1_measured")
    mp = med["median_postcutoff"]["f1_measured"]
    S.f("Median-Sensitivität post-cutoff (Primärsicht)", f"Ränge {arms_str(mp['avg_ranks'], r2)}, p(F) = {pv(mp['friedman']['p_F'])}",
        "derived/median.json median_postcutoff")
    mpost, mcore = pc["median_diff_to_baseline_post"], pc["median_diff_to_baseline_core"]
    S.f("Kontaminations-Kontrolle: Median (Agent − Baseline) post-cutoff vs. Kern",
        "; ".join(f"{L[a]} {s3(mpost[a]['median_diff'])} vs. {s3(mcore[a]['median_diff'])}" for a in ARMS),
        "summary.baseline_comparison.diff → `derived/postcutoff.json` median_diff_to_baseline_*")
    sb = pc["secondary"]["per_arm"]
    S.f("Sekundärmaß „signifikant besser“ post-cutoff (von 12)", arms_str({a: sb[a]["sig_better"] for a in ARMS}, str),
        "derived/postcutoff.json secondary.per_arm")
    S.f("VORBEHALT Sekundärmaß post-cutoff nicht mit dem Kern vergleichbar (≈ 134k/138k Testzeilen machen kleine "
        "Differenzen signifikant)", f"n_test ev_purchase {sp['_post_cutoff|ev_purchase']['n_test']}, student_health "
        f"{sp['_post_cutoff|student_health']['n_test']}", "derived/splits.json; docs/roadmap.md:287-288 (Hinweis gilt weiter)")
    S.f("A/B-Runs auf student_health mit eval→model-Rücksprung (Eval-Gate meldete „signifikant unter der Baseline“)",
        f"{len(pc['eval_to_model_runs_fsm'])}: " + ", ".join(x.replace("main-student_health-", "") for x in pc["eval_to_model_runs_fsm"])
        + " — alle SUCCESS", "events.jsonl Phasenfolge → `derived/postcutoff.json` eval_to_model_runs_fsm")
    S.f("A/B-Runs auf student_health ohne Rücksprung", f"{len(pc['fsm_runs_without_eval_loop_student_health'])}: "
        + ", ".join(x.replace("main-student_health-", "") for x in pc["fsm_runs_without_eval_loop_student_health"])
        + " (darunter der einzige A-Fail hard_fsm-r4)", "derived/postcutoff.json fsm_runs_without_eval_loop_student_health")
    S.f("Mechanismus (Code-Sichtung 27.10.): alle 8 C/D-Fails auf student_health gewichten die Klassen (class_weight="
        "\"balanced\" o. ä.); C/D trainieren in 12/12 student_health-Runs nur HGB und prüfen nur gegen einen Dummy; "
        "C r4 vergleicht gewichtet vs. ungewichtet und besteht. Nachgeprüft 09.11.: alle 8 Fails exakt "
        "class_weight=\"balanced\"; die übrigen 3 SUCCESS-Runs gewichten nicht (s. F6-532)", "—",
        "docs/roadmap.md:158-163; runs `main-student_health-{crisp,naked}_react-r*/work/*.py`", "doku")
    S.f("Mechanismus A/B (Code-Sichtung 27.10.): alle 12 probieren Gewichtung; 7 Runs über Modellvergleich in der "
        "Modeling-Phase (gewichtet behalten, meist RF balanced), 5 über das Eval-Gate (Gewichtung 4× entfernt, 1× "
        "abgeschwächt); Gewichtung ohne Gate-Rückmeldung verworfen: 0/7", "—", "docs/roadmap.md:163-178", "doku")
    S.f("Memorisierungs-Check (Kern): Dataset in Abschlussmeldung/Code benannt A 64 · B 65 · C 59 · D 51 von 78; "
        "F1 benannt − nicht benannt +0.004 = mittlere Differenz INNERHALB gemischter Zellen (12 Zellen mit beiden Sorten), "
        "nicht gepoolt → kein Vorteil; Außenzugriff 0 (1 Fehlalarm urllib.parse in cmc × B r3). Nachgerechnet 09.11. "
        "(K6-07): +0.0036 über 12 Zellen", "—",
        "docs/roadmap.md:119-121; `scripts/memorization_check.py` (read-only, nur Kern-Runs)")
    cb = pc["class_balance"]
    S.f("Klassenverteilung (Train) post-cutoff", "student_health " + "/".join(f"{k} {v}" for k, v in cb["student_health"].items())
        + "; ev_purchase " + "/".join(f"{k} {v}" for k, v in cb["ev_purchase"].items()),
        "summary.du_report.class_balance (A/B) → `derived/postcutoff.json` class_balance")
    S.f("K-git-Runs post-cutoff (student_health A r5, B r6) beide SUCCESS; ohne sie A 10/11, B 11/11", "—",
        "derived/kgit.json, derived/postcutoff.json level1")
    S.f("VORBEHALT: N = 2, explorativ; der Effekt hängt an einer Falle auf einem Set; die A+B/C+D-Gruppierung ist post hoc; "
        "Gate-Vorteil = Zugriff auf eine externe Referenzmessung (Val-Labels beim Harness); Ursache Gewichtung nur korrelativ, "
        "nicht per Ablation; synthetische Playground-Daten", "—", "docs/roadmap.md:189-196; docs/benchmarks.md:164-169", "doku")
    # Nachtraege Schreib-Agent Kap. 6 (09.11.): Code-Sichtung work/*.py bzw. Folgerung aus k und CD
    S.f("student_health × C/D: Klassen-Gewichtung im hinterlassenen Code (grep class_weight/sample_weight/balanc/sampl in "
        "`work/*.py`)", "alle 8 Fails class_weight=\"balanced\"; die 4 SUCCESS-Runs: C r4 gewichtet (sample_weight, "
        "Vergleich gewichtet vs. ungewichtet), C r3, D r3, D r6 ohne jede Gewichtung oder Resampling",
        "runs `main-student_health-{crisp,naked}_react-r1..r6/work/*.py`, summary.success")
    S.f("Post-Cutoff Nemenyi: bei k = 4 ist die größtmögliche Rangdifferenz 3 (Rang 1 vs. 4) < CD "
        f"{fpc['cd']:.2f} → kein Paar trennbar, unabhängig von den Daten", f"3 < {fpc['cd']:.2f}", "F6-519 (CD); k = 4")
    # Nachtrag 15.11. (Kommentar Kap. 4.4): Feature-Datei bleibt nach eval→model im Workdir
    er = j("eval_return_access.json")
    S.f("Runs mit eval→model-Rücksprung (Feature-Datei liegt danach im Workdir): Tool-Aufrufe der wiederholten "
        "Model-Phase, die test_features nennen (Positivkontrolle: Eval-Fenster nennen die Datei)",
        f"Hauptläufe {len(er['main']['runs_with_eval_to_model'])} Runs (alle post-cutoff), zweite Stufe "
        f"{len(er['h1']['runs_with_eval_to_model'])} Runs; davon mit Nennung: "
        f"{len(er['main']['runs_mentioning'])} bzw. {len(er['h1']['runs_mentioning'])} "
        f"({er['main']['tool_calls_min_max'][0]}–{er['main']['tool_calls_min_max'][1]} bzw. "
        f"{er['h1']['tool_calls_min_max'][0]}–{er['h1']['tool_calls_min_max'][1]} Tool-Aufrufe je Run)",
        "events.jsonl + transcripts (Prompt-Fenster der Model-Aufrufe nach dem ersten Eval) → "
        "`derived/eval_return_access.json` (thesis/scripts/collect_facts_eval_return.py)")

    # ---------------------------------------------------------------- 6.6
    S.sec("6.6 Phasen-Diagnostik (RQ2)", 6)
    for b, lab in (("core", "Kern"), ("post_cutoff", "Post-Cutoff")):
        for a in ("hard_fsm", "soft_fsm"):
            x = ph[b][a]
            at = x["attempts_total"]
            S.f(f"{lab} {FULL[a]}: Versuche je Phase gesamt (bu/du/prep/model/eval) bei {x['runs']} Runs",
                "/".join(str(at[p]) for p in ("bu", "du", "prep", "model", "eval")),
                f"summary.attempts → `derived/phases.json` {b}.{a}.attempts_total")
            lp = x["loops"]
            S.f(f"{lab} {FULL[a]}: Runs mit wiederholter Phase; Retries; Rücksprünge",
                f"{x['runs_with'].get('any_repeat', 0)}/{x['runs']} Runs; DU-Retries {lp.get('retry_du', 0)}, Model-Retries "
                f"{lp.get('retry_model', 0)}; eval→model {lp.get('back_eval_to_model', 0)}, du→bu (spec_mismatch) "
                f"{lp.get('back_du_to_bu', 0)}", f"events.jsonl Phasenfolge → `derived/phases.json` {b}.{a}.loops, runs_with")
            S.f(f"{lab} {FULL[a]}: Gate-Fails nach Phase; Ursache der DU-Fails",
                ", ".join(f"{k} {v}" for k, v in x["gate_fail_events_by_phase"].items())
                + f"; DU: Zielspalte in excluded_columns gelistet {x['du_gate_fail_feedback'].get('target_excluded', 0)}/"
                f"{sum(x['du_gate_fail_feedback'].values())}",
                f"events.jsonl gate_passed/gate_feedback → `derived/phases.json` {b}.{a}")
            S.f(f"{lab} {FULL[a]}: Worker-Calls (total_calls) Mittel (min–max); Prüfer-Calls (gate_calls)",
                f"{x['total_calls_mean']:.2f} ({x['total_calls_min']}–{x['total_calls_max']}); gate_calls gesamt {x['gate_calls_total']}",
                f"summary.total_calls/gate_calls → `derived/phases.json` {b}.{a}")
            S.f(f"{lab} {FULL[a]}: Worker-Turns pro Run Mittel / Median / Max; budget_exhausted",
                f"{x['worker_turns_mean']:.1f} / {x['worker_turns_median']:.0f} / {x['worker_turns_max']}; {x['budget_exhausted']}",
                f"summary.worker_turns (= Summe events.worker_num_turns in {x['worker_turns_summary_equals_events']}/{x['runs']}) → `derived/phases.json`")
            S.f(f"{lab} {FULL[a]}: BU-Zielspalte im ersten Versuch korrekt", f"{x['bu_correct_first']}/{x['runs']}",
                "summary.bu_correct_first → `derived/phases.json`")
    for b, lab in (("core", "Kern"), ("post_cutoff", "Post-Cutoff")):
        rt = ph["react_turns"][b]
        S.f(f"{lab} C/D: Worker-Turns der einen Session Mittel / Median / Max",
            "; ".join(f"{L[a]} {rt[a]['mean']:.1f} / {rt[a]['median']:.0f} / {rt[a]['max']}" for a in ("crisp_react", "naked_react")),
            f"events.jsonl worker_num_turns → `derived/phases.json` react_turns.{b}")
    mf = ph["core"]["hard_fsm"]["model_gate_fail_feedback_A"]
    S.f("Model-Gate-Fails Kern A: model.pkl nicht ladbar (eigenes Modul nicht importierbar, „No module named …“)",
        f"{sum(mf.values())}", "derived/phases.json core.hard_fsm.model_gate_fail_feedback_A")
    S.f("Model-Gate-Fails Post-Cutoff A: 1× model.pkl nicht ladbar, 1× Metrikname f1_macro statt Spec-Metrik f1",
        f"{sum(ph['post_cutoff']['hard_fsm']['model_gate_fail_feedback_A'].values())}",
        "derived/phases.json post_cutoff.hard_fsm.model_gate_fail_feedback_A")
    rtz = sum(ph["react_turns"][b][a]["summary_field_zero"] for b in ("core", "post_cutoff") for a in ("crisp_react", "naked_react"))
    S.f("DATENLÜCKE: summary.worker_turns und worker_cost_usd sind für alle C/D-Runs 0 (React-Pfad füllt sie nicht); "
        "Turns daher aus events.jsonl, Kosten aus usage_total", f"{rtz}/180 C/D-Runs",
        "src/crispfsm/react.py:100-125 (kein worker_turns/worker_cost_usd); `derived/phases.json` react_turns.*.summary_field_zero")
    v12 = du["v12_excluded_A_B"]
    S.f("DU-Ausschluss V12 (Anrufdauer) auf bank_marketing, A/B (du_report.excluded_columns)",
        f"A {v12['hard_fsm']}/6, B {v12['soft_fsm']}/6; Zellmittel F1 A {f3(du['bank_marketing']['hard_fsm']['f1_mean'])}, "
        f"B {f3(du['bank_marketing']['soft_fsm']['f1_mean'])} (C {f3(core['cell_means_f1_measured']['bank_marketing']['crisp_react'])}, "
        f"D {f3(core['cell_means_f1_measured']['bank_marketing']['naked_react'])})", "summary.du_report.excluded_columns → `derived/du_exclusions.json`")
    S.f("V12 bei C/D (Code-Sichtung `work/*.py`, 09.11.): C schließt V12 in 1/6 aus (r2 → 0.710), D in 1/6 (r5 → 0.678, "
        "zusätzlich class_weight=\"balanced\" = der einzige Kern-Fail); D r1 benennt V12 als Leakage und behält es", "C 1/6, D 1/6",
        "docs/roadmap.md:107-112", "doku")
    S.f("Die Default-Baseline nutzt V12 (Leakage-Vorsicht kostet F1 per Konstruktion); A/B bestehen trotzdem", f"baseline_f1 "
        f"{f3(core['baseline_test']['bank_marketing'])}", "docs/roadmap.md:113-115; `derived/core.json` baseline_test", "doku")
    for a in ("hard_fsm", "soft_fsm"):
        e = du["eucalyptus"][a]
        S.f(f"DU-Ausschlüsse eucalyptus {FULL[a]} (Spalte: Runs von 6)",
            ", ".join(f"{k} {v}" for k, v in e["excluded_column_counts"].items()) + f"; Zellmittel F1 {f3(e['f1_mean'])}",
            "summary.du_report.excluded_columns → `derived/du_exclusions.json` eucalyptus")
    S.f("eucalyptus C/D Zellmittel zum Vergleich", f"C {f3(core['cell_means_f1_measured']['eucalyptus']['crisp_react'])}, "
        f"D {f3(core['cell_means_f1_measured']['eucalyptus']['naked_react'])}", "derived/core.json cell_means_f1_measured.eucalyptus")
    S.f("VORBEHALT: Phasen-Diagnostik deskriptiv (RQ2); für C/D keine Phasengrenzen messbar (Kodierung der D-Transkripte = Paket 2)",
        "—", "docs/schreib-workflow.md:44", "doku")
    # Nachtraege Schreib-Agent Kap. 6 (09.11.): direkt aus events.jsonl/Transkripten, nicht in derived/
    S.f("Model-Gate-Fails Kern A nach Set und Ursache",
        "5 Runs: bank_marketing r3, r4, r5, r6 + adult r6; alle „model.pkl cannot be loaded: No module named …“, weil der "
        "Worker eine eigene Threshold-Wrapper-Klasse in einem eigenen Modul definierte (ThresholdModel 3×, "
        "ThresholdClassifier 1×, ThresholdedHGB 1×)",
        "runs `main-*-hard_fsm-*/events.jsonl` (phase model, gate_passed false, gate_feedback); Klassendefinition per Suche "
        "in `transcripts/*.jsonl`")
    S.f("Einziger Model-Gate-Fail Kern B",
        "main-bank_marketing-soft_fsm-r2: model.pkl entpickelt zu einem Dict (Schlüssel model, threshold, positive_label) "
        "statt zu einem Estimator, kein .predict() (Prüfer-Verdikt)", "runs `main-bank_marketing-soft_fsm-r2/events.jsonl` gate_feedback")
    S.f("Runs mit DU-Gate-Fail pro Set (Ursache stets Zielspalte in excluded_columns, s. F6-603/609)",
        "Kern A 24 Runs auf 9 Sets: bank_marketing 5, connect_4 3, covertype 3, fashion_mnist 3, wine 3, credit_g 2, "
        "phoneme 2, vehicle 2, cmc 1; Kern B 19 Runs auf 8 Sets: covertype 4, connect_4 3, fashion_mnist 3, vehicle 3, "
        "bank_marketing 2, phoneme 2, cmc 1, credit_g 1; Post-Cutoff A student_health 2; B student_health 2, ev_purchase 1",
        "runs `main-*-{hard,soft}_fsm-*/events.jsonl` (phase du, gate_passed false)")
    # Nachtrag 09.11.: eucalyptus-Ausschluesse C/D (Pruefbefund K6-04), thesis/scripts/eucalyptus_exclusions.py
    eu = j("eucalyptus_exclusions.json")
    ck, SRC_EU = eu["checks"], "`derived/eucalyptus_exclusions.json` (thesis/scripts/eucalyptus_exclusions.py)"
    S.f("eucalyptus C/D, Sichtung der abgegebenen Pipeline (alle 12 Runs, `work/model.py`): Code-Beleg (Datei:Zeile) "
        "stimmt mit Einfluss-Test überein (Spalte im Test-Input 10× permutiert, model.py neu ausgeführt; nicht genutzt ⇔ "
        "keine Vorhersage ändert sich); Neu-Ausführung reproduziert die abgegebenen Vorhersagen exakt; im Transkript liegt "
        "der letzte model.py-Lauf nach der letzten Änderung, kein anderer Befehl schreibt danach predictions.parquet",
        f"Beleg = Test {ck['cd_static_equals_dynamic']}/{ck['cd_runs']}; reproduziert {ck['cd_reproduced']}/{ck['cd_runs']} "
        f"(je {eu['runs']['main-eucalyptus-crisp_react-r1']['influence_test']['test_rows']} Testzeilen); "
        f"Transkript {ck['cd_transcript_ok']}/{ck['cd_runs']}; {eu['n_features']} Feature-Spalten",
        SRC_EU + " checks; runs `main-eucalyptus-{crisp,naked}_react-r*/work/model.py`, `transcripts/*.jsonl`")
    for a, arm in (("C", "crisp_react"), ("D", "naked_react")):
        rs = [eu["runs"][f"main-eucalyptus-{arm}-r{k}"] for k in range(1, 7)]
        S.f(f"eucalyptus {FULL[arm]}: vom abgegebenen Modell nicht genutzte Spalten pro Run → best_f1_measured (outcome)",
            "; ".join(f"r{r['rep']} {', '.join(r['excluded_columns']) or 'keine'} → {f3(r['best_f1_measured'])}"
                      for r in rs) + f"; alle {', '.join(eu['by_arm'][a]['outcomes'])}",
            SRC_EU + " runs.*.excluded_columns, evidence (Datei:Zeile), best_f1_measured aus summary.json")
    S.f("eucalyptus C/D: nicht genutzte Spalten (Spalte: Runs von 6); Runs mit allen Feature-Spalten; Runs ohne mind. "
        "eine der ID-Spalten Abbrev/Map_Ref/PMCno (die A in 6/6 ausschließt, F6-633)",
        "; ".join(f"{a}: " + ", ".join(f"{k} {v}" for k, v in eu["by_arm"][a]["excluded_column_counts"].items())
                  + f"; alle {eu['n_features']} Spalten in {len(eu['by_arm'][a]['runs_all_features'])}/6 "
                  f"(r{', r'.join(map(str, eu['by_arm'][a]['runs_all_features']))}); ID-Spalte ausgeschlossen in "
                  f"{len(eu['by_arm'][a]['runs_excluding_any_id_column'])}/6 "
                  f"(r{', r'.join(map(str, eu['by_arm'][a]['runs_excluding_any_id_column']))})" for a in ("C", "D")),
        SRC_EU + " by_arm")
    S.f("eucalyptus C/D: F1-Mittel (Bereich) der Runs, die mind. eine ID-Spalte ausschließen, vs. der Runs, die alle drei "
        "behalten; je 3 Runs, deskriptiv",
        "; ".join(f"{a} {f3(b['f1_mean_excluding_id'])} ({f3(b['f1_range_excluding_id'][0])}–{f3(b['f1_range_excluding_id'][1])}) "
                  f"vs. {f3(b['f1_mean_keeping_id'])} ({f3(b['f1_range_keeping_id'][0])}–{f3(b['f1_range_keeping_id'][1])})"
                  for a, b in ((a, eu["by_arm"][a]) for a in ("C", "D"))) + "; Bereiche überlappen in beiden Armen",
        SRC_EU + " by_arm.*.f1_mean_*/f1_range_*")
    S.f("Kontrolle A/B mit derselben Methode (Eval-Script mit gespeichertem model.pkl + Preprocessor, Einfluss-Test): "
        "nicht genutzte Spalten = du_report.excluded_columns; Vorhersagen exakt reproduziert",
        "; ".join(f"{eu['runs'][r]['arm']} r{eu['runs'][r]['rep']}: {', '.join(eu['runs'][r]['influence_test']['excluded_dynamic'])} "
                  f"(= DU-Report: {'ja' if eu['runs'][r]['du_report_equals_dynamic'] else 'NEIN'})" for r in ck["control_runs"])
        + f"; reproduziert {ck['control_reproduced']}/{len(ck['control_runs'])}",
        SRC_EU + " runs.main-eucalyptus-hard_fsm-r3, runs.main-eucalyptus-soft_fsm-r1")
    d5 = eu["runs"]["main-eucalyptus-naked_react-r5"]
    wk = min(eu["runs"].values(), key=lambda r: r["influence_test"]["weakest_used_column"]["permutations_with_change"])
    S.f("VORBEHALT eucalyptus-Sichtung: Einfluss-Test bestätigt den Code-Beleg, ersetzt ihn nicht (schwächste genutzte "
        "Spalte wirkt nur in wenigen Permutationen, Ortsspalten untereinander redundant); D r5 probierte nach dem einzigen "
        "model.py-Lauf per python -c Varianten mit weniger Ausschlüssen (CV), übernahm sie aber nicht → abgegeben mit 4 Ausschlüssen",
        f"min. wirksame Permutationen einer genutzten Spalte {ck['min_permutations_with_change_used_columns']}/10 "
        f"({wk['arm']} r{wk['rep']} {wk['influence_test']['weakest_used_column']['column']}); "
        f"D r5 abgegeben: {', '.join(d5['excluded_columns'])}",
        SRC_EU + " runs.*.influence_test.weakest_used_column; runs `main-eucalyptus-naked_react-r5/transcripts/*.jsonl` Aktion 6")

    # ---------------------------------------------------------------- 6.7
    S.sec("6.7 Robustheit: Leakage und unlösbare Aufgaben", 7)
    S.f("Leakage-Design perfekt: eine getarnte Spalte im Feature-Stil, Train perfekt mit dem Ziel korreliert, Val/Test "
        "permutiert (wertlos); Baseline-Predictions sauber", "—", "docs/leakage-runs.md:14-23, 144-145", "doku")
    S.f("Leakage-Design spurious: fachfremd-absurder Name, Train ≈ 85 % trennend (kontinuierlich), Val/Test permutiert", "—",
        "docs/leakage-runs.md:100-108", "doku")
    S.f("Messung leakage_exploited (Code): Cohen's κ zwischen Prediction und Leak-Label; exploited ⇔ z ≥ Z_CRIT und κ ≥ KAPPA_MIN",
        f"z ≥ {cfg['leak_z_crit']}, κ ≥ {cfg['leak_kappa_min']}", "src/crispfsm/leakage.py:25-33 → `derived/design_checks.json` config")
    for v, lab in (("perfect", "perfekt"), ("spurious", "spurious")):
        x = lk[v]
        S.f(f"Leakage {lab}: Runs, Arm, Leak ausgenutzt (Wilson)", f"{x['runs']} Runs, nur D; {x['exploited']}/{x['runs']} "
            f"{wil(x['exploited_wilson'])}", f"runs `{'leak' if v == 'perfect' else 'spur'}-*/summary.json` leakage_exploited → `derived/leakage.json` {v}")
        S.f(f"Leakage {lab}: ausgenutzte Runs (κ, z, Ergebnis)",
            "; ".join(f"{d['run_id']} κ {d['kappa']:.2f}, z {d['z']:.1f}, {'SUCCESS' if d['success'] else 'FAIL'} (Δ {s3(d['diff'])})"
                      for d in x["exploited_detail"]), f"derived/leakage.json {v}.exploited_detail")
        fails = {ds: y["runs"] - y["success"] for ds, y in x["per_set"].items() if y["runs"] - y["success"]}
        S.f(f"Leakage {lab}: SUCCESS / FAIL (Sets mit FAIL)", f"{x['success']}/{x['fail']} (" + ", ".join(f"{k} {n_}" for k, n_ in fails.items())
            + "); alle FAIL below_threshold", f"derived/leakage.json {v}")
        S.f(f"Leakage {lab}: Kosten, Harness", f"{usd(x['cost'])}; {', '.join(x['harness_commits'])}", f"derived/leakage.json {v}")
        S.f(f"Leakage {lab}: größtes κ unter den nicht ausgenutzten Runs", f"{x['kappa_max_not_exploited']:.2f}",
            f"derived/leakage.json {v}.kappa_max_not_exploited")
    for ds in CORE_ORDER:
        p_, s_ = lk["perfect"]["per_set"][ds], lk["spurious"]["per_set"][ds]
        S.f(f"Leakage pro Set {ds}: perfekt (Spalte, ausgenutzt, κ-Mittel, SUCCESS, ΔF1 zu Kern-D) / spurious (dito)",
            f"perfekt `{p_['column']}`: {p_['exploited']}/3, κ {s2(p_['kappa_mean'])}, {p_['success']}/3, ΔF1 {s3(p_['delta_f1_vs_core_D'])}; "
            f"spurious `{s_['column']}`: {s_['exploited']}/3, κ {s2(s_['kappa_mean'])}, {s_['success']}/3, ΔF1 {s3(s_['delta_f1_vs_core_D'])}",
            f"derived/leakage.json perfect/spurious.per_set.{ds}")
    S.f("credit_g spurious r1 folgt dem Leak deutlich (κ 0.76) und besteht trotzdem (obere KI-Grenze +0.063 bei 200 "
        "Testzeilen) → SUCCESS ist auf kleinen Holdouts kein Leakage-Schutz", "—", "derived/leakage.json spurious.exploited_detail; derived/splits.json")
    S.f("wine perfekt: κ 0.20–0.24, aber nicht ausgenutzt (z < 3.09 bei 36 Testzeilen)", "—", "derived/leakage.json perfect.per_set.wine")
    S.f("Weiter-Regel (≥ 8/39) in beiden Batches nicht ausgelöst → A, B, C auf Leakage nicht gefahren", "perfekt 1/39, spurious 4/39",
        "derived/leakage.json *.rule_triggered; docs/leakage-runs.md:125-127")
    S.f("Erwartung §7 (nur Spurious-Batch, vorab festgeschrieben; spurious häufiger als perfekt; am häufigsten auf cmc, "
        "credit_g, eucalyptus, connect_4, kaum auf breast_cancer, wine, cnae_9): häufiger ja (4 vs. 1); auf den genannten "
        "schweren Sets nur 1 von 4 (credit_g; cmc, eucalyptus, connect_4 nein; dafür adult 1, bank_marketing 2); leichte "
        "Sets 0. NICHT identisch mit der Arm-Erwartung aus Kap. 3 (03-research-design.tex: A/B weniger Ausnutzung + höhere "
        "Pass-Raten als C/D) — die ist mangels A/B/C-Runs nicht prüfbar (K6-02)", "—",
        "derived/leakage.json; docs/leakage-runs.md:127-130; runs `spur-*/summary.json` leakage_reliance.exploited")
    S.f("D nennt die Leak-Spalte im eigenen Code/Transkript (perfekt 34/39 im kontaminierten Lauf; sauberer Lauf nicht "
        "nachgezählt)", "—", "docs/leakage-runs.md:79-80", "doku")
    S.f("VORBEHALT Leakage: nur Arm D → kein Arm-Vergleich; n = 3; Batches vor dem Kern, nicht verschränkt; auf "
        "anonymisierten Sets (bank_marketing, cnae_9, phoneme, fashion_mnist) fällt jeder lesbare Name schon durchs Schema auf",
        "—", "docs/leakage-runs.md:117-118, 148-149", "doku")
    S.f("Vergiftete/unlösbare Aufgaben (ABSTAIN-Raten): nicht gefahren; ABSTAIN in allen 438 sauberen Runs", "0",
        "summary.abstained → `derived/controls.json`; docs/roadmap.md:335, 369")
    S.f("Natürlicher Leak im Kern: V12 auf bank_marketing (s. 6.6)", "A/B 6/6 ausgeschlossen", "derived/du_exclusions.json")
    # Nachtrag 16.11. (Kommentar Kap. 5.4): Trennschaerfe der Weiter-Regel unter der geplanten Auswertung
    # (Fisher exakt paarweise ueber die 6 Armpaare, Holm; docs/leakage-runs.md §6). Bester Fall fuer einen
    # Unterschied: ein Arm nutzt den Leak nie (0/39). Holm-Schritt 1 verlangt p < 0.05/6.
    from scipy.stats import fisher_exact
    fp = {k: fisher_exact([[k, 39 - k], [0, 39]])[1] for k in (1, 4, 7, 8)}
    S.f("Trennschärfe der Weiter-Regel: Fisher exakt (zweiseitig), D mit k/39 gegen einen Arm mit 0/39; Holm über "
        "6 Armpaare verlangt im ersten Schritt p < 0.05/6", f"k = 8: p = {fp[8]:.4f}; k = 7: p = {fp[7]:.4f}; "
        f"k = 4: p = {fp[4]:.3f}; k = 1: p = {fp[1]:.2f}; Holm-Grenze {0.05 / 6:.4f} → 8 = kleinstes k mit möglichem "
        "signifikantem Unterschied", "scipy.stats.fisher_exact (berechnet in make_factsheets.py); Auswertungsplan "
        "docs/leakage-runs.md:86-88")

    # ---------------------------------------------------------------- 6.8
    S.sec("6.8 Kosten und Token", 8)
    for b, lab in (("core", "Kern"), ("post_cutoff", "Post-Cutoff")):
        c = co[b]
        pa = c["per_arm"]
        S.f(f"{lab}: Kosten gesamt und pro Arm", f"{usd(c['total'])}; " + arms_str({a: pa[a]["cost_total"] for a in ARMS}, r2) + " USD",
            f"summary.total_cost_usd → `derived/costs.json` {b}")
        S.f(f"{lab}: Kosten pro Run Mittel / Median / Max (USD)",
            "; ".join(f"{L[a]} {pa[a]['cost_per_run_mean']:.3f} / {pa[a]['cost_per_run_median']:.3f} / {pa[a]['cost_per_run_max']:.2f}" for a in ARMS),
            f"derived/costs.json {b}.per_arm")
        r_ = c["ratios_cost_per_run"]
        S.f(f"{lab}: Verhältnis der Kosten pro Run", ", ".join(f"{k} {v:.2f}" for k, v in r_.items()), f"derived/costs.json {b}.ratios_cost_per_run")
        for a in ARMS:
            t = pa[a]["tokens"]
            tr = pa[a]["tokens_per_run"]
            S.f(f"{lab} {FULL[a]}: Token gesamt input / output / cache_read / cache_creation",
                f"{t['input_tokens']:,} / {t['output_tokens']:,} / {t['cache_read_tokens']:,} / {t['cache_creation_tokens']:,}",
                f"summary.usage_total → `derived/costs.json` {b}.per_arm.{a}.tokens")
            S.f(f"{lab} {FULL[a]}: Token pro Run (Mittel) input / output / cache_read / cache_creation",
                f"{tr['input_tokens']:,.0f} / {tr['output_tokens']:,.0f} / {tr['cache_read_tokens']:,.0f} / {tr['cache_creation_tokens']:,.0f}",
                f"derived/costs.json {b}.per_arm.{a}.tokens_per_run")
            sh = pa[a]["cost_share_by_token_type"]
            S.f(f"{lab} {FULL[a]}: Kostenanteil nach Token-Art output / cache_creation / cache_read / input",
                f"{pct(sh['output_tokens'])} / {pct(sh['cache_creation_tokens'])} / {pct(sh['cache_read_tokens'])} / {pct(sh['input_tokens'])}",
                f"derived/costs.json {b}.per_arm.{a}.cost_share_by_token_type")
        bs = pa["soft_fsm"]
        S.f(f"{lab}: Prüfer-Kosten (Gate-Calls) von B", f"{usd(bs['gate_cost_events'])} = {pct(bs['gate_cost_share'])} der B-Kosten; "
            f"B-Worker {usd(bs['cost_total'] - bs['gate_cost_events'])} (A gesamt {usd(pa['hard_fsm']['cost_total'])})",
            f"events.jsonl gate_usage × Listenpreis → `derived/costs.json` {b}.per_arm.soft_fsm.gate_cost_events")
    for v, lab in (("leak_perfect", "Leakage perfekt"), ("leak_spurious", "Leakage spurious")):
        x = co[v]["per_arm"]["naked_react"]
        S.f(f"{lab}: Kosten gesamt / pro Run (D)", f"{usd(x['cost_total'])} / {x['cost_per_run_mean']:.3f} USD", f"derived/costs.json {v}")
    S.f("Alle sauberen Batches zusammen", usd(co["_all_clean_batches_total"]), "derived/costs.json _all_clean_batches_total")
    pr = co["_pricing_usd_per_mtok"]
    S.f("Listenpreise Opus 4.8 (USD pro 1 Mio. Token) input / output / cache_read / cache_creation",
        f"{pr['input']:.2f} / {pr['output']:.2f} / {pr['cache_read']:.2f} / {pr['cache_creation']:.2f}", "src/crispfsm/pricing.py:20-26")
    S.f("Kosten = usage_total × Listenpreis (Harness rechnet selbst; Abo-Nutzung, keine echte API-Rechnung); Nachrechnung "
        "aus events.jsonl (Worker + Prüfer) = summary.total_cost_usd in allen Armen und Batches", "ja",
        "src/crispfsm/run.py:347-352; `derived/costs.json` *.repriced_equals_summary")
    S.f("Harness-Overhead: Fixkosten-Sockel pro Call mit Isolations-Flags ~15k gecachte Token aus Systemprompt + "
        "Tool-Schemas (9 069 Basis + ~6 000 Schemas Read/Write/Bash); 42 182 = Variante A, d. h. Betreiber-Konfiguration mit "
        "Plugins/MCP (`--allowedTools`, Standard-Settings), nicht schlicht „ohne Flags“ (K6-19); gemessen 18.09. vor den "
        "Haupt-Runs (Sonnet 5, 3 Tools, ältere CLI); Kosten nur relativ zwischen Armen vergleichbar, nicht mit API-Zahlen",
        "15 070 vs. 42 182 Token",
        "docs/harness-overhead.md:121-163, 15-40; docs/meetings/2026-09-14-entscheidungen.md:36-42", "doku")
    S.f("Haiku-Titel-Nebencall der CLI steckt in deren total_cost_usd, nicht im usage-Objekt → in den Harness-Kosten nicht enthalten",
        "—", "docs/roadmap.md:345", "doku")

    # ---------------------------------------------------------------- 6.9 Erwartungs-Muster
    S.sec("6.9 Präregistrierte Erwartungs-Muster (14.09.) — beobachtet ja/nein", 9)
    S.f("Quelle: fünf Muster, a priori festgeschrieben und abgenommen", "14.09.2026", "docs/meetings/2026-09-14-entscheidungen.md:25; "
        "thesis/chapters/03-research-design.tex:40-50", "doku")
    p1 = ex["p1_parity_clean"]
    S.f("Muster 1 (saubere Sets: Mittelwert-Parität A ≈ D) — Kern: JA",
        f"Friedman p = {pv(p1['core']['friedman_p_F'])}; Rang A {p1['core']['rank_A']:.2f} vs. D {p1['core']['rank_D']:.2f} (Δ "
        f"{p1['core']['rank_diff_AD']:.2f} < CD {p1['core']['cd']:.2f}); Cliff's δ A–D {p1['core']['cliffs_mean_AD']:+.3f} ({p1['core']['cliffs_mag_AD']})",
        "derived/expectations.json p1_parity_clean.core")
    S.f("Muster 1 — Post-Cutoff: NICHT PRÜFBAR (N = 2); deskriptiv A > D auf beiden Sets",
        f"student_health {f3(p1['post']['f1_A']['student_health'])} vs. {f3(p1['post']['f1_D']['student_health'])}; ev_purchase "
        f"{f3(p1['post']['f1_A']['ev_purchase'])} vs. {f3(p1['post']['f1_D']['ev_purchase'])}", "derived/expectations.json p1_parity_clean.post")
    p2 = ex["p2_traps"]
    S.f("Muster 2 (Fallen: Leakage / unlösbar / dirty → A > D) — injizierte Leaks: NICHT GEPRÜFT (A nicht gefahren, weil "
        "D den Leak selbst meidet)", f"D ausgenutzt perfekt {p2['leak_perfect_D_exploited']}/39, spurious {p2['leak_spurious_D_exploited']}/39",
        "derived/expectations.json p2_traps")
    S.f("Muster 2 — unlösbare Aufgaben: NICHT GEPRÜFT (Robustheits-Set nicht gefahren)", "0 Runs", "derived/design_checks.json; docs/roadmap.md:369")
    bm = p2["bank_marketing_f1"]
    S.f("Muster 2 — natürlicher Leak V12 (Kern): Vorsicht ja (A 6/6 ausgeschlossen), aber A < D im F1. KORRIGIERT 09.11. "
        "(K6-03): SUCCESS A 6/6 vs. D 5/6 ist KEIN Vorteil beim Umgang mit der Falle — der einzige D-Fail (r5) schloss V12 "
        "selbst aus (`LEAK = [\"V12\"]`, dazu class_weight=\"balanced\"); C r2 schließt V12 ebenfalls aus → 0.710 ≈ A/B → A "
        "vorsichtiger, aber weder bei Pass-Rate noch F1 besser",
        f"F1 {arms_str(bm)}; SUCCESS {arms_str(p2['bank_marketing_success'], str)}",
        "derived/expectations.json p2_traps; runs `main-bank_marketing-naked_react-r5/work/model.py:13-14`, "
        "`main-bank_marketing-crisp_react-r2/work/pipeline.py:14`")
    p3 = ex["p3_runsd_A_lt_D"]
    S.f("Muster 3 (Run-SD A < D) — Kern: NEIN", f"Median A {f4(p3['core_median_A'])} vs. D {f4(p3['core_median_D'])}; A < D in "
        f"{p3['core_cells_A_lt_D']}/{p3['core_cells']} Sets", "derived/expectations.json p3_runsd_A_lt_D")
    S.f("Muster 3 — Post-Cutoff: JA (deskriptiv, N = 2)", f"Median A {f4(p3['post_median_A'])} vs. D {f4(p3['post_median_D'])}",
        "derived/expectations.json p3_runsd_A_lt_D")
    p4 = ex["p4_imbalanced_success_A_gt_D"]
    ms = p4["majority_share"]
    S.f("Anteil der Mehrheitsklasse im Train (Imbalance; Definition „unbalanciert“ war nicht vorab festgelegt)",
        "; ".join(f"{ds} {pct(ms[ds], 0)}" for ds in CORE_ORDER + POST_ORDER), "summary.du_report.class_balance → `derived/expectations.json`")
    S.f("Muster 4 (SUCCESS auf unbalancierten Sets A > D) — Kern: Richtung nur durch einen Run (bank_marketing A 6/6 vs. D 5/6), "
        "auf allen übrigen Sets gleich; gepoolt Fisher p = 1.000 → praktisch NEIN", f"einziger Unterschied {p4['core_only_nonpass'][0]}",
        "derived/expectations.json p4_imbalanced_success_A_gt_D")
    pf = p4["post_fisher_AD"]
    S.f("Muster 4 — Post-Cutoff (beide Sets unbalanciert, 86 %/83 % Mehrheitsklasse): Richtung JA, nicht signifikant",
        f"A 11/12 vs. D 6/12, Fisher p = {pv(pf['p'])} (Holm {pv(pf['p_holm'])})", "derived/expectations.json p4_imbalanced_success_A_gt_D.post_fisher_AD")
    p5 = ex["p5_refinement_highdim"]
    S.f("Muster 5 (Verfeinerung: auf High-Dim/Feature-Engineering-lastigen Sets D vorn) — Kern: NEIN",
        f"cnae_9 A {f3(p5['cnae_9']['hard_fsm'])} vs. D {f3(p5['cnae_9']['naked_react'])}; fashion_mnist A "
        f"{f3(p5['fashion_mnist']['hard_fsm'])} vs. D {f3(p5['fashion_mnist']['naked_react'])}", "derived/expectations.json p5_refinement_highdim")
    S.f("Muster 5 — „Enforcement gewinnt, wo reiche Spalten Fallen sind“: im F1 nicht beobachtet (V12-Ausschluss senkt F1); "
        "Post-Cutoff: nicht anwendbar (keine High-Dim-Sets). ACHTUNG (K6-12): Lesart über bank_marketing (16 Features) ist "
        "nachträglich und nicht Teil der in Kap. 3 festgelegten Prüfung (nur cnae-9, fashion-mnist) → im Text gestrichen; "
        "nur der Post-Cutoff-Teil wird noch zitiert", "—", "derived/expectations.json p2_traps, p5_refinement_highdim")
    S.f("Abgleich Doku: Roadmap nennt für Muster 3 und 4 „im Kern beide nicht, post-cutoff beide in vorhergesagter Richtung“ — "
        "deckt sich, Muster 4 im Kern formal A > D um einen Run", "—", "docs/roadmap.md:185-186", "doku")

    # ---------------------------------------------------------------- 6.10 zweite Stufe (H1)
    h = j("h1.json")
    S.sec("6.10 Zweite Stufe (H1): A gegen A-Single (Kern N = 13, n = 6; Post-Cutoff explorativ)", 10)
    HS = "`derived/h1.json`"
    ac = h["analysis"]["core"]
    pr, rp, ov = ac["primary"], ac["replication"], ac["old_vs_single"]
    lab = {"hard_fsm": "A", "hard_fsm_single": "A-Single", "hard_fsm_old": "A (Okt.)"}

    def rates(block):
        return " · ".join(f"{lab[a]} {v['success']}/{v['n']} {wil(v['wilson'])}" for a, v in block["rates"].items())

    def wsr(block, m):
        t = block["metrics"][m]["test"]
        return (f"R+ {t['r_plus']:.1f}, R− {t['r_minus']:.1f}, p = {pv(t['p'])}; {lab[block['x']]} vorn {t['x_ahead']}, "
                f"{lab[block['y']]} vorn {t['y_ahead']}, gleich {t['ties']}; Median-Differenz {s3(t['median_diff'])}")

    S.f("H1a (Bestehensrate), primär: A (neu) gegen A-Single, gleichzeitig gefahren — Fisher exakt, zweiseitig; Urteil",
        f"{rates(pr)}; p = {pv(pr['fisher']['p'])}; H1a {ac['verdicts']['H1a']}", f"{HS} analysis.core.primary.rates/fisher/verdicts")
    S.f("H1b (F1, primär f1_measured): exakter Vorzeichen-Rang-Test (Null-Differenzen halbiert), zweiseitig; Urteil",
        f"{wsr(pr, 'f1_measured')}; H1b {ac['verdicts']['H1b']}", f"{HS} analysis.core.primary.metrics.f1_measured.test")
    S.f("Sensitivität H1b: kombinierter Score (Fail = 0) und F1 nur bestandener Runs",
        f"f1_combined: {wsr(pr, 'f1_combined')} | f1_on_success: {wsr(pr, 'f1_on_success')}",
        f"{HS} analysis.core.primary.metrics.f1_combined/f1_on_success.test")
    m = pr["metrics"]["f1_measured"]
    big = [(ds, v) for ds, v in m["cliffs_per_dataset"].items() if abs(v) >= 0.474]
    S.f("Cliff's δ A − A-Single (pro Set über die Replikate; Romano-Schwellen, large ≥ 0.474)",
        f"Mittel {s3(m['cliffs_mean'])} ({m['cliffs_mean_magnitude']}); large: " + "; ".join(f"{ds} {s2(v)}" for ds, v in big),
        f"{HS} analysis.core.primary.metrics.f1_measured.cliffs_per_dataset")
    cm = h["cell_means_f1_measured"]["A_new_vs_single"]
    S.f("Zellmittel F1 (f1_measured) A / A-Single / Differenz A − A-Single pro Kern-Set",
        "; ".join(f"{ds} {f3(cm[ds][0])} / {f3(cm[ds][1])} / {s3(h['diff_A_new_minus_single'][ds])}" for ds in CORE_ORDER),
        f"{HS} cell_means_f1_measured.A_new_vs_single, diff_A_new_minus_single")
    fr = [x for x in h["failed_runs"] if x["run_id"].split("-")[1] in CORE_ORDER]
    S.f("Nicht bestandene Kern-Runs der zweiten Stufe (alle below baseline)",
        "; ".join(f"{x['run_id']} F1 {f3(x['f1'])} vs. Baseline {f3(x['baseline_f1'])} (KI-Obergrenze {x['ci_hi']:+.4f})" for x in fr),
        f"{HS} failed_runs")
    rs_ = pr["run_sd"]
    S.f("Run-SD A / A-Single (Kern): Median und Maximum über 13 Zellen",
        " · ".join(f"{lab[a]} Median {f4(v['median_sd'])}, max {f4(v['max_sd'])}" for a, v in rs_.items()),
        f"{HS} analysis.core.primary.run_sd")
    S.f("Replikation A (Okt.) gegen A (neu), Kern: Bestehen, Fisher, Vorzeichen-Rang-Test (f1_measured)",
        f"{rates(rp)}; Fisher p = {pv(rp['fisher']['p'])}; {wsr(rp, 'f1_measured')}", f"{HS} analysis.core.replication")
    e4 = h["expectations"]["e4_replication"]
    S.f("Replikation: Zellmittel A (neu) − A (Okt.) pro Kern-Set; Anzahl innerhalb ±0.005 (vorab erwartet: alle)",
        f"innerhalb {e4['core_cells_within_0.005']} von {e4['core_cells']}; außerhalb: "
        + "; ".join(f"{ds} {s3(v)}" for ds, v in e4["outside"].items()) + f"; max |Δ| {f3(e4['max_abs_diff'])}",
        f"{HS} diff_A_new_minus_A_old, expectations.e4_replication")
    dmax = max(h["diff_A_new_minus_single"].items(), key=lambda t: abs(t[1]))
    rmax = max(h["diff_A_new_minus_A_old"].items(), key=lambda t: abs(t[1]))
    S.f("Größte Zelldifferenz A − A-Single vs. größte Zelldifferenz zwischen den beiden A-Chargen",
        f"A − A-Single max |Δ| {f3(abs(dmax[1]))} ({dmax[0]}); A (neu) − A (Okt.) max |Δ| {f3(abs(rmax[1]))} ({rmax[0]})",
        f"{HS} diff_A_new_minus_single, diff_A_new_minus_A_old")
    S.f("Ergänzend A (Okt.) gegen A-Single, Kern: Fisher und Vorzeichen-Rang-Test (f1_measured)",
        f"Fisher p = {pv(ov['fisher']['p'])}; {wsr(ov, 'f1_measured')}", f"{HS} analysis.core.old_vs_single")
    pcut = h["analysis"]["post_cutoff"]
    S.f("Post-Cutoff (explorativ, N = 2): Bestehen gepoolt je Arm (A-Chargen, A-Single, Hauptläufe B–D zum Vergleich)",
        " · ".join(f"{ {'hard_fsm_old': 'A (Okt.)', 'hard_fsm': 'A (neu)', 'hard_fsm_single': 'A-Single', 'soft_fsm': 'B', 'crisp_react': 'C', 'naked_react': 'D'}[a]} {v['success']}/{v['n']}"
                   for a, v in pcut["pooled"].items()),
        f"{HS} analysis.post_cutoff.pooled")
    S.f("Post-Cutoff: Fisher A (neu) gegen A-Single (nur beschreibend)", f"p = {pv(pcut['fisher']['hard_fsm_vs_hard_fsm_single']['p'])}",
        f"{HS} analysis.post_cutoff.fisher")
    rows = {(r["arm"], r["dataset"]): r for r in pcut["rows"]}
    S.f("Post-Cutoff: F1-Mittel (SD) und Bestehen pro Set für A (Okt.), A (neu), A-Single",
        "; ".join(f"{ds}: " + ", ".join(f"{lab[a]} {f3(rows[(a, ds)]['f1_measured_mean'])} ({f4(rows[(a, ds)]['f1_measured_sd'])}), "
                                        f"{rows[(a, ds)]['success']}/{rows[(a, ds)]['n']}"
                                        for a in ("hard_fsm_old", "hard_fsm", "hard_fsm_single")) for ds in POST_ORDER),
        f"{HS} analysis.post_cutoff.rows")
    pf = [x for x in h["failed_runs"] if x["run_id"].split("-")[1] in POST_ORDER]
    S.f("Post-Cutoff: nicht bestandene Runs der zweiten Stufe",
        "; ".join(f"{x['run_id']} F1 {f3(x['f1'])} vs. Baseline {f3(x['baseline_f1'])} (KI-Obergrenze {x['ci_hi']:+.4f})" for x in pf),
        f"{HS} failed_runs")
    S.f("Post-Cutoff: F1-Drift A (neu) − A (Okt.)", "; ".join(f"{ds} {s3(v)}" for ds, v in e4["post_f1_drift_new_minus_old"].items()),
        f"{HS} expectations.e4_replication.post_f1_drift_new_minus_old")
    hc = h["costs"]
    S.f("Kosten pro Run (Mittel) und Calls pro Run: A / A-Single, Kern und Post-Cutoff",
        f"Kern A {hc['h1_core']['hard_fsm']['cost_per_run_mean']:.3f} USD, {hc['h1_core']['hard_fsm']['calls_mean']:.2f} Calls · "
        f"A-Single {hc['h1_core']['hard_fsm_single']['cost_per_run_mean']:.3f} USD, {hc['h1_core']['hard_fsm_single']['calls_mean']:.2f} Calls; "
        f"Post-Cutoff A {hc['h1_post']['hard_fsm']['cost_per_run_mean']:.3f} USD, {hc['h1_post']['hard_fsm']['calls_mean']:.2f} · "
        f"A-Single {hc['h1_post']['hard_fsm_single']['cost_per_run_mean']:.3f} USD, {hc['h1_post']['hard_fsm_single']['calls_mean']:.2f}",
        f"{HS} costs")
    tk = {a: hc["h1_core"][a] for a in ("hard_fsm", "hard_fsm_single")}
    S.f("Kern: Token und Kosten pro Run nach Token-Art (Output, Cache-Lesen, Cache-Schreiben), A / A-Single",
        " · ".join(f"{lab[a]}: Output {v['tokens_per_run']['output_tokens'] / 1e3:.1f}k = {usd(v['cost_by_token_type_per_run']['output_tokens'])}, "
                   f"Cache-Lesen {v['tokens_per_run']['cache_read_tokens'] / 1e3:.0f}k = {usd(v['cost_by_token_type_per_run']['cache_read_tokens'])}, "
                   f"Cache-Schreiben {v['tokens_per_run']['cache_creation_tokens'] / 1e3:.1f}k = {usd(v['cost_by_token_type_per_run']['cache_creation_tokens'])}"
                   for a, v in tk.items()),
        f"{HS} costs.h1_core.*.tokens_per_run, cost_by_token_type_per_run")
    ex_ = h["expectations"]
    e1 = ex_["e1_abs_f1_diff_below_0.01"]
    S.f("Erwartung 1 (vorab 13.11.): Kern kein Unterschied auf beiden Stufen — JA; Zusatz „F1-Differenz je Set < 0.01“ — TEILWEISE",
        f"H1a/H1b {ex_['e1_core_no_difference_both_levels']['H1a']}/{ex_['e1_core_no_difference_both_levels']['H1b']}; "
        f"|Δ| < 0.01 in {e1['count']} von {e1['of']}; darüber: " + "; ".join(f"{ds} {s3(v)}" for ds, v in e1["above"].items()),
        f"docs/h1-single-session.md §5; {HS} expectations.e1_*")
    e2 = ex_["e2_post_single_success"]
    S.f("Erwartung 2: A-Single Post-Cutoff mindestens 10 von 12 — JA", f"{e2['success']}/{e2['n']}",
        f"docs/h1-single-session.md §5; {HS} expectations.e2_post_single_success")
    e3 = ex_["e3_cost_delta_single_minus_a_core"]
    S.f("Erwartung 3: A-Single kostet pro Run 0.30–0.65 USD mehr als A — NEIN, A-Single ist billiger",
        f"Δ (A-Single − A, Kern) {e3['delta_per_run']:+.3f} USD", f"docs/h1-single-session.md §5; {HS} expectations.e3_*")
    S.f("Erwartung 4: Replikation — Kern alle 78 bestanden JA; Post-Cutoff 10–12 von 12 JA; Zellmittel innerhalb 0.005 TEILWEISE",
        f"Kern {e4['core_success_new']}/{e4['core_n_new']}; Post-Cutoff {e4['post_success_new']}/{e4['post_n_new']}; "
        f"innerhalb 0.005: {e4['core_cells_within_0.005']} von {e4['core_cells']}",
        f"docs/h1-single-session.md §5; {HS} expectations.e4_replication")
    cpl = h["completeness"]
    S.f("Vollständigkeit (Urteil nur bei 13 × 6 gemessenen Runs je Arm): primär, Replikation, A (Okt.) gegen A-Single",
        "; ".join(f"{k} {'vollständig' if v['complete'] else 'UNVOLLSTÄNDIG: ' + str(v['problems'])}" for k, v in cpl.items()),
        f"{HS} completeness (= analysis.core.*.completeness)")
    return S


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Faktenblaetter kap5.md/kap6.md aus thesis/facts/derived/")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "thesis", "facts"),
                    help="Zielordner (Default thesis/facts; zum Vergleichen z. B. ein Scratch-Ordner)")
    out_dir = ap.parse_args().out_dir
    os.makedirs(out_dir, exist_ok=True)
    for sheet, name in ((kap5(), "kap5.md"), (kap6(), "kap6.md")):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(sheet.render())
        print(f"  -> thesis/facts/{name}: {sheet.count()} Fakten")


if __name__ == "__main__":
    main()
