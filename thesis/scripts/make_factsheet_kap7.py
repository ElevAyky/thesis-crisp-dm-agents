#!/usr/bin/env python
"""Faktenblatt Kap. 7 (thesis/facts/kap7.md) aus thesis/facts/derived/process_coding.json.

Reihenfolge: extract_actions.py (optional, Lesefassung) -> code_process.py -> dieses Script.
Alle Zahlen kommen aus process_coding.json (Kodierung der Transkripte der sauberen Runs,
Rang 1); Vorbehalte und Uebernahmen sind fester Text. Ein von Hand angehaengter Abschnitt ab der
Markierung VALIDATION_MARK (Validierung, IDs F7-9xx) bleibt beim Neuschreiben erhalten.

Aufruf aus dem Repo-Root:
    uv run python thesis/scripts/make_factsheet_kap7.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[2]
CODING = REPO / "thesis" / "facts" / "derived" / "process_coding.json"
OUT = REPO / "thesis" / "facts" / "kap7.md"
VALIDATION_MARK = "<!-- F7-9xx: Validierung (nicht generiert, bleibt erhalten) -->"
SRC = "derived/process_coding.json"
LABEL = {  # deutsch, kurz
    "bu_explicit": "Ziel + Metrik vor der Modellanpassung festgehalten",
    "bu_metric_rationale": "Metrikwahl begründet (Klassenverteilung)",
    "bu_success_criterion": "Erfolgskriterium vor der Modellanpassung",
    "bu_business_context": "Geschäftliche Rahmung (Ziele, Kosten, Annahmen)",
    "du_inspect_before_model": "Dateninspektion vor der Modellanpassung",
    "du_class_dist": "Klassenverteilung geprüft",
    "du_missing": "Fehlwerte geprüft",
    "du_quality_ext": "Duplikate/Konstanten/IDs geprüft",
    "du_stats": "Deskriptive Statistik",
    "du_relations": "Beziehungen zwischen Attributen erkundet",
    "du_leakage": "Leakage-/ID-Spalten ausdrücklich erwogen (vor der Modellanpassung)",
    "prep_explicit": "Explizite Aufbereitung",
    "prep_select": "Benannte Feature-Spalten ausgeschlossen",
    "prep_construct": "Abgeleitete Attribute",
    "model_multi_family": "≥ 2 Modellfamilien",
    "model_cv": "Kreuzvalidierung",
    "model_test_design": "Interne Validierung (CV oder Holdout)",
    "model_tuning": "Hyperparameter-Suche/-Revision",
    "model_assess_detail": "Bewertung je Klasse (Report/Konfusionsmatrix)",
    "eval_baseline": "Vergleich mit Baseline",
    "eval_business": "Bewertung gegen Ziel/Erfolgskriterium",
    "eval_review_next": "Prozess-Review oder nächste Schritte",
    "doc_notes": "Notizdatei geschrieben",
    "doc_crisp_sections": "Notiz nach CRISP-Phasen gegliedert",
    "canonical_order": "Phasen in kanonischer Reihenfolge",
    "implicit_crisp": "Folgt implizit CRISP-DM (alle fünf + kanonisch)",
    "backjump_any": "≥ 1 Rücksprung in eine frühere Phase",
    "monolith": "Monolith (ein Skript: Inspektion bis Vorhersage)",
}
CRISP = {
    "bu_explicit": "1.3", "bu_metric_rationale": "1.3", "bu_success_criterion": "1.1/1.3",
    "bu_business_context": "1.1/1.2", "du_inspect_before_model": "2.2", "du_class_dist": "2.3",
    "du_missing": "2.4", "du_quality_ext": "2.4", "du_stats": "2.2/2.3", "du_relations": "2.3",
    "du_leakage": "2.4", "prep_explicit": "3.1–3.5", "prep_select": "3.1", "prep_construct": "3.3",
    "model_multi_family": "4.1/4.4", "model_cv": "4.2", "model_test_design": "4.2",
    "model_tuning": "4.3/4.4", "model_assess_detail": "4.4", "eval_baseline": "5.1",
    "eval_business": "5.1", "eval_review_next": "5.2/5.3", "doc_notes": "Outputs",
    "doc_crisp_sections": "Outputs",
}


def kn(v) -> str:
    return "erzwungen" if v == "enforced" else f"{v[0]}/{v[1]}"


def p3(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def orr(x) -> str:
    if isinstance(x, str):
        return {"nan": "—", "inf": "∞"}.get(x, x)
    return f"{x:.2f}"


class Sheet:
    """IDs F7-2xx (Abschnitt 7.2) in festen Teilbereichen je Unterabschnitt, damit sie bei einer
    Neugenerierung stabil bleiben: 7.2.0 ab 201, 7.2.1 ab 221, 7.2.2 ab 241, 7.2.3 ab 271,
    7.2.4 ab 281, 7.2.5 ab 290, Abbildungen ab 297. F7-9xx bleibt frei (Validierung)."""

    def __init__(self):
        self.lines, self.next, self.limit, self.count = [], 0, 0, 0

    def head(self, title, start, limit):
        self.lines += ["", title, "", "| ID | Aussage | Wert | Quelle | Prüfstufe |", "|---|---|---|---|---|"]
        self.next, self.limit = start, limit

    def row(self, text, value, src, level="kodiert"):
        assert self.next <= self.limit, f"ID-Bereich ueberlaufen: F7-{self.next}"
        fid = f"F7-{self.next}"
        self.next += 1
        self.count += 1
        self.lines.append(f"| {fid} | {text} | {value} | {src} | {level} |")
        return fid


def main() -> None:
    d = json.loads(CODING.read_text(encoding="utf-8"))
    S = d["summary"]["scopes"]
    C, P, ALL = S["core"], S["post_cutoff"], S["all"]
    R = d["runs"]
    react = {k: v for k, v in R.items() if v["arm"] in ("crisp_react", "naked_react")}
    fsm = {k: v for k, v in R.items() if v["arm"] in ("hard_fsm", "soft_fsm")}
    sh = Sheet()
    L = sh.lines
    L += ["# Faktenblatt Kapitel 7 — Discussion (7.2 Prozess-Kodierung)", "",
          "> **Zahlenquelle für den Schreib-Agenten zu Kap. 7.2.** Generiert — nicht von Hand ändern, sondern",
          "> `thesis/scripts/code_process.py` (Transkripte → `thesis/facts/derived/process_coding.json`) und",
          "> `thesis/scripts/make_factsheet_kap7.py` neu laufen lassen. Codebuch: `thesis/facts/codebook.md`;",
          "> Lesefassung je Run: `thesis/facts/derived/actions/<run>.md` (`extract_actions.py`, gitignored).",
          "> Prüfstufe **kodiert** = regelbasiert aus den Transkripten der sauberen Runs (Rang 1) kodiert,",
          "> an einer Stichprobe von 24 C/D-Runs blind gegenkodiert (Abschnitt F7-9xx; A/B-Referenzcodes",
          "> nicht gegenkodiert, Einschränkungen je Code in F7-922 bis F7-925); **roh** = direkt aus",
          "> Rohdaten/Code; **doku** = aus Projekt-Doku übernommen. Arme: A = Hard FSM, B = Soft FSM,",
          "> C = CRISP-ReAct, D = Naked ReAct. Kern = 13 Sets × 6 = 78 Runs/Arm; Post-Cutoff = 2 Sets × 6 =",
          "> 12 Runs/Arm. „p“ = Fisher exakt zweiseitig C vs. D, „Holm“ = über die 28 binären Hauptcodes",
          "> (je Umfang). **Alles explorativ**, nicht präregistriert. Runs-Repo = `../thesis-crisp-dm-runs`."]

    # ------------------------------------------------------------------ 7.2.0 Umfang
    sh.head("## 7.2.0 Umfang, Methode, Vorbehalte", 201, 220)
    nC = sum(1 for r in react.values() if r["arm"] == "crisp_react")
    nD = sum(1 for r in react.values() if r["arm"] == "naked_react")
    actC = sum(r["values"]["n_actions"] for r in react.values() if r["arm"] == "crisp_react")
    actD = sum(r["values"]["n_actions"] for r in react.values() if r["arm"] == "naked_react")
    actAB = sum(r["values"]["n_actions"] for r in fsm.values())
    sh.row("Kodierte Runs (alle sauberen `main-*`, Harness 14e99e4)",
           f"{len(R)}: C {nC}, D {nD} voll kodiert (je 78 Kern + 12 Post-Cutoff); A {sum(1 for r in fsm.values() if r['arm']=='hard_fsm')}, "
           f"B {sum(1 for r in fsm.values() if r['arm']=='soft_fsm')} nur Referenz-Codes",
           "runs `main-*/transcripts/*.jsonl` → `thesis/scripts/code_process.py` → " + SRC)
    sh.row("Tool-Aktionen in den kodierten Worker-Sessions (Prüfer-Sessions von B ausgenommen)",
           f"C {actC}, D {actD}; A+B {actAB}", "transcripts → `extract_actions.load_run` → " + SRC + " runs.*.values.n_actions")
    sh.row("Codebuch: 28 binäre Hauptcodes (BU 4, DU 7, PREP 3, MODEL 5, EVAL 3, DOC 2, Ablauf 4) + Hilfscodes; "
              "abgeleitet aus den generischen Aufgaben 1.1–5.3 von CRISP-DM 1.0 (Text in `prompts.CRISP_TASKS`)",
           "—", "thesis/facts/codebook.md §2–5; src/crispfsm/prompts.py:66-354", "doku")
    sh.row("Statistik: Fisher exakt zweiseitig C vs. D je binärem Hauptcode, Holm über 28 Codes, getrennt für "
              "Kern und Post-Cutoff; Zusammenhang mit dem Ergebnis nur deskriptiv", "—", "codebook.md §6; code_process.py summarize", "doku")
    sh.row("VORBEHALT regelbasiert: Code-Muster und Schlüsselwörter erkennen, *dass* etwas im Code/Text steht, "
              "nicht ob es inhaltlich richtig oder wirksam ist; Rücksprünge auf Aktionsebene", "—", "codebook.md §8", "doku")
    sh.row("VORBEHALT nicht präregistriert: Codebuch und Regeln während der Kalibrierung des Kodierers festgelegt, "
              "die Gesamtraten waren dabei sichtbar; neun Regeländerungen an Einzelfällen begründet; Abweichungen vom "
              "Startvorschlag: BU ≤ DU entfällt (anonymisierte Zielspalte erzwingt Inspektion vor der Zielbestimmung), "
              "PREP/MODEL/EVAL derselben Aktion gleichzeitig", "—", "codebook.md §5.1, §8", "doku")
    sh.row("VORBEHALT Prompt-Befolgung: Der D-Prompt (und damit auch C) verlangt Baseline-Vergleich, Prüfung "
              "ungeeigneter Spalten (IDs, Konstanten, Leakage) und Fehlwert-Behandlung; diese Codes messen bei C/D "
              "Befolgung, nicht spontanes Prozesswissen", "—", "src/crispfsm/prompts.py:476-495", "roh")
    sh.row("VORBEHALT Sichtbarkeit: Thinking-Blöcke in den Transkripten leer (nur Signatur); kodiert ist nur "
              "sichtbarer Text, Code und Notizen", "—", "runs `main-*/transcripts/*.jsonl` (type thinking, Feld leer)", "roh")
    sh.row("VORBEHALT Validierung: Kodierer an Einzelfällen kalibriert; blinde Gegenkodierung einer Stichprobe von "
              "24 C/D-Runs durch einen zweiten Kodierer (LLM-Agent, kein Mensch) abgeschlossen; A/B-Referenzcodes "
              "nicht gegenkodiert; Bewertung je Code in F7-922 bis F7-925", "—",
           "F7-901 bis F7-926; derived/coding_agreement.json", "doku")

    # ------------------------------------------------------------------ 7.2.1 Ablauf
    sh.head("## 7.2.1 Ablauf der freien Arme (Kern, 78 Runs je Arm)", 221, 240)
    nC_ = C["numeric"]["C"]
    nD_ = C["numeric"]["D"]

    def dist(dct, k=6):
        items = list(dct.items())[:k]
        return "; ".join(f"{a.replace(' ', ' → ') if a else '(leer)'} {b}" for a, b in items)
    id_order = sh.row("Reihenfolge der Erstauftritte (BU = erste ausdrückliche Zielspalten-Aussage, DU, Prep, Model, Eval)",
           f"C: {dist(nC_['phase_order'])} · D: {dist(nD_['phase_order'])}", SRC + " summary.scopes.core.numeric.*.phase_order")
    viol = [k.replace("main-", "") for k, r in react.items() if r["values"]["order_violations"] and not r["post_cutoff"]]
    fc = C["fisher_C_vs_D"]
    sh.row("Kanonische Reihenfolge (BU < Model, DU < Prep ≤ Model ≤ Eval, fehlende Phasen übersprungen)",
           f"C {kn(fc['canonical_order']['C'])}, D {kn(fc['canonical_order']['D'])}; Verletzungen: {', '.join(viol) or 'keine'} "
           f"(Aufbereitung erst nach dem ersten Modell)", SRC + " runs.*.values.canonical_order, order_violations")
    sh.row("Folgt implizit CRISP-DM (alle fünf Phasen, BU als Ziel + Metrik vor der Modellanpassung, kanonisch)",
           f"C {kn(fc['implicit_crisp']['C'])}, D {kn(fc['implicit_crisp']['D'])}; p {p3(fc['implicit_crisp']['p'])}, "
           f"Holm {p3(fc['implicit_crisp']['p_holm'])}", SRC + " summary.scopes.core.fisher_C_vs_D.implicit_crisp")
    sh.row("Was an „implizit CRISP“ fehlt (Runs; Mehrfachnennung möglich)",
           f"C: {', '.join(f'{a} {b}' for a, b in nC_['phases_missing'].items())} · "
           f"D: {', '.join(f'{a} {b}' for a, b in nD_['phases_missing'].items())}",
           SRC + " summary.scopes.core.numeric.*.phases_missing")
    id_seq = sh.row("Phasenfolge inkl. Rücksprüngen (komprimiert), häufigste",
           f"C: {dist(nC_['phase_sequence'], 5)} · D: {dist(nD_['phase_sequence'], 5)}",
           SRC + " summary.scopes.core.numeric.*.phase_sequence")
    sh.row("Runs mit ≥ 1 Rücksprung",
           f"C {kn(fc['backjump_any']['C'])}, D {kn(fc['backjump_any']['D'])}; p {p3(fc['backjump_any']['p'])}, Holm {p3(fc['backjump_any']['p_holm'])}",
           SRC + " fisher_C_vs_D.backjump_any")
    sh.row("Rücksprünge je Run (Verteilung) und Ziel der Rücksprünge",
           f"C {nC_['n_backjumps_dist']}, Ziele {nC_['backjump_to']} · D {nD_['n_backjumps_dist']}, Ziele {nD_['backjump_to']}",
           SRC + " numeric.*.n_backjumps_dist, backjump_to")
    sh.row("Tool-Aktionen je Run: Median (Mittel; min–max)",
           f"C {nC_['n_actions']['median']:g} ({nC_['n_actions']['mean']:.2f}; {nC_['n_actions']['min']}–{nC_['n_actions']['max']}) · "
           f"D {nD_['n_actions']['median']:g} ({nD_['n_actions']['mean']:.2f}; {nD_['n_actions']['min']}–{nD_['n_actions']['max']})",
           SRC + " numeric.*.n_actions")
    sh.row("Geschriebene .py-Dateien je Run (Mittel; max) und Inline-Python-Aufrufe je Run (Median; max)",
           f"C {nC_['n_scripts']['mean']:.2f}; {nC_['n_scripts']['max']} und {nC_['n_inline_python']['median']:g}; {nC_['n_inline_python']['max']} · "
           f"D {nD_['n_scripts']['mean']:.2f}; {nD_['n_scripts']['max']} und {nD_['n_inline_python']['median']:g}; {nD_['n_inline_python']['max']}",
           SRC + " numeric.*.n_scripts, n_inline_python")
    single = C["rates"]["single_pipeline_script"]
    id_mono = sh.row("Monolith (ein Skript von der Inspektion bis zur Vorhersage) / ein Skript mit Aufbereitung + Modell + Vorhersage",
                     f"C {kn(fc['monolith']['C'])} / {kn(single['C'])}, D {kn(fc['monolith']['D'])} / {kn(single['D'])}; Monolith p {p3(fc['monolith']['p'])}",
                     SRC + " rates.monolith, rates.single_pipeline_script")
    ver = C["rates"]["verify_outputs"]
    _, p_ver = fisher_exact([[ver["C"][0], ver["C"][1] - ver["C"][0]], [ver["D"][0], ver["D"][1] - ver["D"][0]]])
    id_ver = sh.row("Ausgaben nach dem Schreiben geprüft (Hilfscode, nicht in der Holm-Familie: eigene Bash-Aktion "
                    "liest predictions.parquet/result.json; Prüfung im Skript selbst zählt nicht)",
                    f"C {kn(ver['C'])}, D {kn(ver['D'])}; Fisher p {p3(p_ver)} (unkorrigiert)", SRC + " rates.verify_outputs")
    sh.row("Lesart (keine eigene Zahl): Das typische D-Muster ist Inspektion (Schema, Klassenverteilung, Fehlwerte) → "
           "Ziel/Metrik-Aussage → ein Skript mit Aufbereitung, Modell, interner Validierung, Baseline-Vergleich und "
           f"Vorhersage → Abschlussbericht (eigene Ausgabeprüfung davor nur in {kn(ver['D'])}); es entspricht der "
           "Phasenfolge 2 → 3 → 4 → 5 von CRISP-DM, ohne 1.1/1.2 und ohne 5.2/5.3",
           "—", f"{id_order}, {id_seq}, {id_mono}, {id_ver}; 7.2.2", "kodiert")

    # ------------------------------------------------------------------ 7.2.2 Inhalts-Codes Kern
    sh.head("## 7.2.2 Inhalts-Codes C vs. D, Kern (A/B als Referenz, wo nicht erzwungen)", 241, 270)
    order = [c for c in LABEL if c not in ("canonical_order", "implicit_crisp", "backjump_any", "monolith")]
    for code in order:
        f = fc[code]
        rates = C["rates"][code]
        ref = ""
        if "A" in rates:
            ref = f"; Referenz A {kn(rates['A'])}, B {kn(rates['B'])}"
            if code == "model_test_design":
                ref += " (in A/B vom Modeling-Kontrakt verlangt, keine freie Wahl)"
            elif code == "du_leakage":
                ref += " (in A/B von den BU-/DU-Kontrakten erfragt)"
        sh.row(f"{LABEL[code]} (CRISP {CRISP[code]}, Code `{code}`)",
                     f"C {kn(f['C'])}, D {kn(f['D'])}; OR {orr(f['odds_ratio'])}, p {p3(f['p'])}, Holm {p3(f['p_holm'])}{ref}",
                     SRC + f" summary.scopes.core.fisher_C_vs_D.{code}, rates.{code}")
    sig = [c for c in order + ["implicit_crisp", "backjump_any"] if fc[c]["p_holm"] < 0.05]
    sh.row("Codes mit Holm-p < 0.05 (C vs. D, Kern)",
           ", ".join(f"`{c}`" for c in sig), SRC + " summary.scopes.core.fisher_C_vs_D")
    raw = [c for c in fc if fc[c]["p"] < 0.05 and fc[c]["p_holm"] >= 0.05]
    sh.row("Codes mit rohem p < 0.05, aber Holm ≥ 0.05",
           ", ".join(f"`{c}`" for c in raw), SRC + " summary.scopes.core.fisher_C_vs_D")
    aux = C["rates"]
    sh.row("Hilfscodes (C / D): Zielspalte vor Modell benannt; Metrik vor Modell benannt; Leakage/ID irgendwo erwähnt "
              "(inkl. Abschlussbericht); Imputation; Kodierung; Skalierung; Konstanten-Regel; Holdout; Klassen-Gewichtung; "
              "Notiz vor dem Modell",
           " · ".join(f"{kn(aux[c]['C'])} / {kn(aux[c]['D'])}" for c in (
               "bu_target", "bu_metric", "leakage_mention_any", "prep_impute", "prep_encode", "prep_scale",
               "prep_select_rule", "model_holdout", "model_class_weight", "doc_upfront")),
           SRC + " summary.scopes.core.rates (AUX)")

    # ------------------------------------------------------------------ 7.2.3 Modellwahl
    sh.head("## 7.2.3 Modellwahl und Modellbewertung, alle vier Arme (Kern)", 271, 280)
    num = C["numeric"]
    sh.row("Modellfamilien je Run: Mittel, Median (min–max)",
           " · ".join(f"{L_} {num[L_]['n_families']['mean']:.2f}, {num[L_]['n_families']['median']:g} "
                      f"({num[L_]['n_families']['min']}–{num[L_]['n_families']['max']})" for L_ in "ABCD"),
           SRC + " numeric.*.n_families (A/B nur Sessions der Phase model)")
    sh.row("Verteilung Modellfamilien je Run (Familien: Runs)",
           " · ".join(f"{L_} {num[L_]['n_families_dist']}" for L_ in "ABCD"), SRC + " numeric.*.n_families_dist")
    sh.row("Häufigste Familien (Runs, in denen die Familie vorkommt)",
           " · ".join(f"{L_}: " + ", ".join(f"{a} {b}" for a, b in list(num[L_]['family_runs'].items())[:4]) for L_ in "ABCD"),
           SRC + " numeric.*.family_runs")
    sh.row("Art des Tunings (erste zutreffende Regel; None = kein Tuning)",
           " · ".join(f"{L_} {num[L_]['tuning_how']}" for L_ in "ABCD"), SRC + " numeric.*.tuning_how")
    sh.row("Kreuzvalidierung / Holdout (A/B: Testdesign durch Kontrakt vorgegeben, CV vs. Holdout frei)",
           " · ".join(f"{L_} {kn(C['rates']['model_cv'][L_])} / {kn(C['rates']['model_holdout'][L_])}" for L_ in "ABCD"),
           SRC + " rates.model_cv, rates.model_holdout")
    sh.row("Baseline-Vergleich in A/B erzwungen: Modeling-Kontrakt verlangt dummy_score, Model-Gate verlangt "
              "val_score > dummy_score", "—", "src/crispfsm/prompts.py:369-381; src/crispfsm/gates.py:68-85", "roh")

    # ------------------------------------------------------------------ 7.2.4 Post-Cutoff
    sh.head("## 7.2.4 Post-Cutoff (12 Runs je Arm; explorativ, N = 2 Sets)", 281, 289)
    fp = P["fisher_C_vs_D"]
    diffs = [c for c in order if fp[c]["C"] != fp[c]["D"]]
    sh.row("Codes mit unterschiedlicher Zählung C vs. D (C / D von 12)",
           "; ".join(f"`{c}` {fp[c]['C'][0]} / {fp[c]['D'][0]}" for c in diffs) + "; alle Holm ≥ "
           + p3(min(fp[c]["p_holm"] for c in fp)), SRC + " summary.scopes.post_cutoff.fisher_C_vs_D")
    sh.row("Modellfamilien post-cutoff",
           " · ".join(f"{L_}: " + ", ".join(f"{a} {b}" for a, b in P['numeric'][L_]['family_runs'].items()) for L_ in "ABCD"),
           SRC + " summary.scopes.post_cutoff.numeric.*.family_runs")
    tun = sorted(k.replace("main-", "") + f" ({r['values']['model_tuning_how']}, {r['outcome']})"
                 for k, r in react.items() if r["post_cutoff"] and r["values"]["model_tuning"])
    sh.row("C/D-Runs mit Tuning post-cutoff (alle auf ev_purchase)", "; ".join(tun),
           SRC + " runs.*.values.model_tuning(_how)")
    shc = sorted(k.replace("main-", "") + (" gewichtet" if r["values"]["model_class_weight"] else " ungewichtet")
                 + f" {r['outcome']}" for k, r in react.items() if r["dataset"] == "student_health")
    sh.row("student_health × C/D: Klassen-Gewichtung im Code (regelbasiert) und Ergebnis — deckt sich mit F6-532",
           "; ".join(shc), SRC + " runs.*.values.model_class_weight; vgl. F6-526, F6-532")

    # ------------------------------------------------------------------ 7.2.5 Ergebnis
    sh.head("## 7.2.5 Zusammenhang mit dem Ergebnis (deskriptiv, innerhalb C bzw. D)", 290, 296)
    ol = C["outcome_link"]
    pairs = [(abs(v["with"]["mean_diff_set_centered"] - v["without"]["mean_diff_set_centered"]), L_, code,
              v["with"]["mean_diff_set_centered"], v["without"]["mean_diff_set_centered"], v["with"]["n"], v["without"]["n"])
             for L_ in "CD" for code, v in ol[L_].items()
             if v["with"]["n"] >= 5 and v["without"]["n"] >= 5
             and v["with"]["mean_diff_set_centered"] is not None and v["without"]["mean_diff_set_centered"] is not None]
    mx = max(pairs)
    sh.row("Kern: set-zentrierte mittlere F1-Differenz zur Baseline (Differenz minus Mittel aller C+D-Runs des Sets), "
           "Runs mit Code minus Runs ohne Code — größter Betrag über alle Codes mit ≥ 5 Runs in beiden Gruppen, beide Arme",
           f"{mx[0]:.4f} ({mx[1]}, `{mx[2]}`: mit {mx[3]:+.4f}, n {mx[5]}; ohne {mx[4]:+.4f}, n {mx[6]}); "
           f"{len(pairs)} Arm×Code-Paare",
           SRC + " summary.scopes.core.outcome_link")
    sh.row("Kern: Bestehensrate lässt keinen Spielraum (C 78/78, D 77/78; s. F6-104/105)", "—", "F6-104, F6-105", "roh")
    fails = sorted(k.replace("main-", "") for k, r in react.items() if not r["success"])
    fv = [react["main-" + f]["values"] for f in fails]
    sh.row("Alle nicht bestandenen C/D-Runs (alle Umfänge): Anzahl; davon 1 Modellfamilie / ohne Tuning / ohne Rücksprung / "
              "mit Klassen-Gewichtung", f"{len(fails)}; {sum(v['n_families'] == 1 for v in fv)} / "
              f"{sum(not v['model_tuning'] for v in fv)} / {sum(v['n_backjumps'] == 0 for v in fv)} / "
              f"{sum(bool(v['model_class_weight']) for v in fv)} — {', '.join(fails)}",
           SRC + " runs.*.values; outcome aus summary.json")
    oa = ALL["outcome_link"]
    sh.row("Alle Umfänge (90 je Arm): bestanden mit / ohne Rücksprung",
           f"C {oa['C']['backjump_any']['with']['pass']}/{oa['C']['backjump_any']['with']['n']} / "
           f"{oa['C']['backjump_any']['without']['pass']}/{oa['C']['backjump_any']['without']['n']} · "
           f"D {oa['D']['backjump_any']['with']['pass']}/{oa['D']['backjump_any']['with']['n']} / "
           f"{oa['D']['backjump_any']['without']['pass']}/{oa['D']['backjump_any']['without']['n']}",
           SRC + " summary.scopes.all.outcome_link.*.backjump_any")
    op = P["outcome_link"]
    sh.row("Post-Cutoff: bestanden mit / ohne Tuning",
           f"C {op['C']['model_tuning']['with']['pass']}/{op['C']['model_tuning']['with']['n']} / "
           f"{op['C']['model_tuning']['without']['pass']}/{op['C']['model_tuning']['without']['n']} · "
           f"D {op['D']['model_tuning']['with']['pass']}/{op['D']['model_tuning']['with']['n']} / "
           f"{op['D']['model_tuning']['without']['pass']}/{op['D']['model_tuning']['without']['n']}",
           SRC + " summary.scopes.post_cutoff.outcome_link.*.model_tuning")
    sh.row("Post-Cutoff: bestanden mit / ohne Klassen-Gewichtung",
           f"C {op['C']['model_class_weight']['with']['pass']}/{op['C']['model_class_weight']['with']['n']} / "
           f"{op['C']['model_class_weight']['without']['pass']}/{op['C']['model_class_weight']['without']['n']} · "
           f"D {op['D']['model_class_weight']['with']['pass']}/{op['D']['model_class_weight']['with']['n']} / "
           f"{op['D']['model_class_weight']['without']['pass']}/{op['D']['model_class_weight']['without']['n']}",
           SRC + " summary.scopes.post_cutoff.outcome_link.*.model_class_weight")
    sh.row("VORBEHALT: Zusammenhänge rein deskriptiv, auf wenigen Fails (11) und zwei Post-Cutoff-Sets; Codes und "
              "Ergebnis hängen beide am Datensatz; keine Kausalaussage", "—", "codebook.md §6", "doku")

    # ------------------------------------------------------------------ Abbildungen/Tabelle
    sh.head("## 7.2.6 Abbildungen und Tabelle (Entwürfe)", 297, 299)
    sh.row("`thesis/figs/fig7_process.pdf` (15 cm): Anteil der Runs je Code, C und D (gefüllt), A/B als offene "
              "Referenzmarker, Kern; * = Holm-p < 0.05; † = Code mit Vorbehalt (F7-924); A/B bei Baseline und interner "
              "Validierung „enforced“ (Modeling-Kontrakt), bei Leakage „prompted“ (BU-/DU-Kontrakt). Caption: "
              "07-discussion.tex, fig:7:process", "—", "thesis/scripts/make_figures_kap7.py", "kodiert")
    sh.row("`thesis/figs/fig7_sequence.pdf` (15 cm): (a) häufigste Phasenfolgen inkl. Rücksprüngen, C vs. D; "
              "(b) Modellfamilien je Run, A–D, Kern. Captionentwurf: „(a) Most frequent phase sequences of the free arms "
              "in the core suite, compressed to phase changes; a step back to an earlier phase is a return. (b) Number of "
              "model families per run. Arms as in \\autoref{tab:3:arms}.“", "—", "thesis/scripts/make_figures_kap7.py", "kodiert")
    sh.row("`thesis/tables/tab7_process.tex`, Label `tab:7:process`: Zählungen je Code und Arm, Kern + Post-Cutoff, "
              "Holm-p C vs. D; Vorbehalts-Codes mit Fußnotenzeichen a–d, A/B „enf.“/kursiv (erfragt) wie in der Abbildung",
           "—", "thesis/scripts/make_figures_kap7.py write_table", "kodiert")

    # ------------------------------------------------------------------ Übernahmen
    L += ["", "## Übernahmen aus kap6.md (nur Verweise, Werte dort)", "",
          "| Bezug in Kap. 7 | F6-ID | Wofür |", "|---|---|---|",
          "| Ausgangsfrage „D genauso gut“ | F6-201, F6-202, F6-203, F6-215 | Friedman/Nemenyi/Cliff's δ: kein Armunterschied in der Modellqualität |",
          "| Decke Stufe 1 | F6-104, F6-105, F6-116 | C/D bestehen fast alle Kern-Runs; Prozess-Codes können nicht mit dem Bestehen kovariieren |",
          "| Abstand zur Baseline | F6-235, F6-521 | Median-Differenz Agent − Baseline je Arm, Kern vs. Post-Cutoff |",
          "| Stabilität | F6-301 | Run-SD: freie Arme nicht instabiler |",
          "| Memorisierung | F6-528 | Dataset-Benennung arm-unabhängig, kein F1-Vorteil |",
          "| Post-Cutoff-Mechanismus | F6-502, F6-503, F6-504, F6-505, F6-509, F6-526, F6-527, F6-531, F6-532 | HGB + Klassen-Gewichtung bei C/D; Modellvergleich/Eval-Gate bei A/B; Vorbehalte |",
          "| Phasen-Diagnostik A/B | F6-602, F6-603, F6-608, F6-609, F6-627, F6-637, F6-638 | DU-Retries (Zielspalte als ausgeschlossen gelistet), Model-Gate-Fails (A: Threshold-Wrapper im eigenen Modul; B: Dict statt Estimator) |",
          "| Modell und Effort (Kap. 5) | F5-204, F5-408 | Claude Opus 4.8 allein; `--effort high` für alle Calls (src/crispfsm/backend.py:16) |",
          "| Turns C/D | F6-625, F6-626, F6-629 | Worker-Turns der einen Session; summary.worker_turns für C/D leer |",
          "| Vorbehalt Phasengrenzen | F6-636 | „für C/D keine Phasengrenzen messbar“ — durch diese Kodierung adressiert |",
          "| V12/Leakage-Vorsicht | F6-630, F6-631, F6-632, F6-704, F6-709, F6-731 | DU-Ausschluss bei A/B vs. C/D; injizierte Leaks nur bei D gefahren |"]
    L += ["", VALIDATION_MARK, ""]
    old_tail = ""
    if OUT.exists():
        txt = OUT.read_text(encoding="utf-8")
        if VALIDATION_MARK in txt:
            old_tail = txt.split(VALIDATION_MARK, 1)[1].lstrip("\n")
    if not old_tail:
        old_tail = ("## 7.9 Validierung des Kodierers (IDs F7-9xx, frei für den unabhängigen Prüf-Agenten)\n\n"
                    "| ID | Aussage | Wert | Quelle | Prüfstufe |\n|---|---|---|---|---|\n")
    OUT.write_text("\n".join(L) + "\n" + old_tail, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"-> {OUT.relative_to(REPO)} ({sh.count} Einträge)")


if __name__ == "__main__":
    main()
