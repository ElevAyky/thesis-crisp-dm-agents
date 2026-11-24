# thesis/scripts/make_appendix_prompts.py
"""Anhang C (Prompts im Wortlaut) -> thesis/appendix/generated/prompts_*.tex

Importiert die Prompt-Bausteine aus src/crispfsm/ (prompts.py, softgates.py) und rendert genau
die Texte, die Worker und Pruefer bekommen haben — nichts wird abgetippt.

Rendern: mit den Werten der Beispiel-Runs main-adult-<arm>-r1 (Aufgabentext, Zielspalte,
critical_columns aus der BU-Spec, DU-Befunde, Messung des Eval-Gates). Retry-Praeambeln mit
sichtbarem Platzhalter {feedback}.

Pruefungen (Abbruch bei Abweichung):
  1. Code-Stand: prompts.py, softgates.py, gates.py, scoring.py unveraendert seit dem Harness-
     Commit der Runs (14e99e4); alle 360 sauberen main-*-Runs tragen denselben Prompt-Fingerabdruck
     (summary.harness.prompt_sha256), commit 14e99e4, dirty = false.
  2. Replay: fuer JEDEN Worker-Call aller 360 Runs (Arme A-D, Erst- und Wiederholungs-Prompts) und
     jeden Pruefer-Call von Arm B wird der Prompt aus dem Code + dem geloggten Run-Zustand neu
     erzeugt und mit der Nutzer-Nachricht im kopierten Transkript verglichen (byte-genau bis auf
     die Zeilenenden: unter Windows kam jedes \\n als \\r\\n an, s. same_prompt).
  3. Paritaet: C = D + CRISP_GUIDANCE; CRISP-Block jeder Phase steht genau einmal im A/B-Prompt;
     CRISP_GUIDANCE enthaelt die fuenf Bloecke in Reihenfolge.

Satz: Verbatim (fancyvrb) in \\footnotesize, Courier (times-Paket) -> 5.4 pt je Zeichen; bei
441 pt Textbreite passen 81 Zeichen. Zeilen > WRAP werden am letzten Leerzeichen umgebrochen (das
Leerzeichen entfaellt), Folgezeilen tragen einen Haken im linken Rand. Werte aus dem Run stehen
geneigt (\\textsl). Die CRISP-DM-Bloecke stehen einmal (prompts_crisp.tex), in den Prompts als
Verweis-Zeile. Gerade Anfuehrungszeichen als \\textquotesingle, Gedankenstriche in der Grundschrift
(in Courier sehen sie wie '--' aus). Ausgabe auch prompts_check.json (Protokoll der Pruefungen).

Aufruf (Repo-Root):
    uv run python thesis/scripts/make_appendix_prompts.py
    uv run python thesis/scripts/make_appendix_prompts.py --runs-dir ../thesis-crisp-dm-runs
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import re
import subprocess

import yaml

from crispfsm import prompts, softgates
from crispfsm.config import RunConfig
from crispfsm.phases import ALLOWED_TOOLS
from crispfsm.scoring import BaselineComparison, ScoreResult

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "thesis", "appendix", "generated")
HARNESS = "14e99e4"
EXAMPLE = "adult"           # Beispiel-Datensatz fuer das Rendern
EXAMPLE_REP = 1
WRAP = 80                   # Zeichen je Zeile (s. Docstring)
PHASES = ["bu", "du", "prep", "model", "eval"]
PHASE_TITLE = {"bu": "Business Understanding", "du": "Data Understanding", "prep": "Data Preparation",
               "model": "Modeling", "eval": "Evaluation"}
CRISP_RANGE = {"bu": "1.1--1.4", "du": "2.1--2.4", "prep": "3.1--3.5", "model": "4.1--4.4", "eval": "5.1--5.3"}
ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]

# Markierungen im Klartext (vor der LaTeX-Umsetzung): Werte aus dem Run, CRISP-Platzhalter.
MO, MC = "\u27e6", "\u27e7"          # ⟦ ⟧  Run-Wert
PO, PC = "\u27ea", "\u27eb"          # ⟪ ⟫  CRISP-Block (eigene Zeile)
# fancyvrb-commandchars: Escape, Gruppe auf, Gruppe zu — duerfen in keinem Prompt vorkommen.
ESC, BGRP, EGRP = "\\", "@", "^"
FONTSIZE = r"\footnotesize"


# =========================================================================== Rohdaten

def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_events(run_dir):
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def session_prompts(run_dir, session_id):
    """Alle Nutzer-Nachrichten mit Text-Inhalt einer Session in Reihenfolge (= die Prompts, die
    der Harness per stdin geschickt hat; Tool-Ergebnisse sind Listen und fallen heraus)."""
    path = os.path.join(run_dir, "transcripts", session_id + ".jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("type") == "user" and isinstance((e.get("message") or {}).get("content"), str):
                out.append(e["message"]["content"])
    return out


EOL_SEEN = {"crlf": 0, "lf": 0}


def same_prompt(got: str, exp: str) -> bool:
    """Byte-Vergleich Transkript <-> Code. Einzige zugelassene Abweichung: Zeilenenden. Der Harness
    schreibt den Prompt per subprocess.run(text=True, input=...) auf stdin; unter Windows wandelt der
    Text-Modus dabei jedes \\n in \\r\\n um — so hat der Worker den Text bekommen. Gezaehlt in EOL_SEEN."""
    if got == exp:
        EOL_SEEN["lf"] += 1
        return True
    if got == exp.replace("\n", "\r\n"):
        EOL_SEEN["crlf"] += 1
        return True
    return False


def manifests():
    core, post = {}, {}
    for p in sorted(glob.glob(os.path.join(REPO, "manifests", "*.yaml"))):
        with open(p, encoding="utf-8") as f:
            core[os.path.splitext(os.path.basename(p))[0]] = yaml.safe_load(f)
    for p in sorted(glob.glob(os.path.join(REPO, "manifests", "post_cutoff", "*.yaml"))):
        with open(p, encoding="utf-8") as f:
            post[os.path.splitext(os.path.basename(p))[0]] = yaml.safe_load(f)
    return core, post


def clean_runs(runs_dir, datasets):
    out = []
    for d in sorted(os.listdir(runs_dir)):
        parts = d.split("-")
        if d.startswith("main-") and len(parts) == 4 and parts[1] in datasets:
            out.append(d)
    return out


# =========================================================================== Pruefung 1: Code-Stand

def prompt_fingerprint():
    """Wie run.harness_version(): sha256 ueber die Dateien mit LLM-seitigen Texten."""
    src = os.path.join(REPO, "src", "crispfsm")
    h = hashlib.sha256()
    for name in ("prompts.py", "softgates.py", "gates.py", "scoring.py"):
        with open(os.path.join(src, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def check_code_state(runs_dir, run_ids):
    # phases.py fehlt bewusst: 0b1ccca hat dort den Zweig fuer Arm A-Single ergaenzt (zweite Stufe); dass die vier
    # Arme unveraendert bleiben, prueft tests/test_single_session.py. Die Prompt-Texte liegen in den Dateien unten.
    files = [f"src/crispfsm/{n}" for n in ("prompts.py", "softgates.py", "gates.py", "scoring.py", "react.py")]
    if subprocess.run(["git", "cat-file", "-e", f"{HARNESS}^{{commit}}"], cwd=REPO,
                      capture_output=True).returncode == 0:
        diff = subprocess.run(["git", "diff", "--stat", HARNESS, "--", *files], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
        assert not diff, f"Prompt-Dateien weichen von {HARNESS} ab:\n{diff}"
    else:
        # Pruefer-Schnappschuss ohne Historie (Anhang B): es bleibt der Fingerprint-Vergleich unten
        print(f"  Hinweis: Commit {HARNESS} fehlt (Pruefer-Schnappschuss) -> nur Fingerprint-Vergleich")
    fps, commits, dirty = set(), set(), set()
    for rid in run_ids:
        h = jload(os.path.join(runs_dir, rid, "summary.json"))["harness"]
        fps.add(h["prompt_sha256"])
        commits.add(h["commit"][:7])
        dirty.add(h["dirty"])
    assert commits == {HARNESS}, commits
    assert dirty == {False}, dirty
    assert len(fps) == 1, fps
    run_fp, local = fps.pop(), prompt_fingerprint()
    # Gleichheit ist der staerkere Beleg; bei Abweichung bleibt der git-diff-Nachweis oben (z. B.
    # andere Zeilenenden im Checkout) — wird gemeldet, nicht verschwiegen.
    return {"runs": len(run_ids), "fingerprint_runs": run_fp, "fingerprint_local": local,
            "fingerprint_equal": run_fp == local}


# =========================================================================== Pruefung 2: Replay

def worker_prompt(phase, task, target, crit, du_report, resume, feedback, kind):
    if phase == "bu":
        return (prompts.bu_resume_prompt(task, feedback, kind=kind) if resume
                else prompts.bu_initial_prompt(task))
    if phase == "du":
        return (prompts.du_resume_prompt(target, feedback, crit, task_prompt=task, kind=kind) if resume
                else prompts.du_initial_prompt(target, crit, task_prompt=task))
    if resume:
        return prompts.resume_prompt(phase, target, feedback, du_report=du_report, task_prompt=task, kind=kind)
    return prompts.initial_prompt(phase, target, du_report=du_report, task_prompt=task)


def judge_cfg(summary, arm="soft_fsm"):
    assert summary["success_rule"] == "noninferior_paired", summary["success_rule"]
    return RunConfig(run_id=summary["run_id"], workdir=".", arm=arm, eval_metric="f1",
                     baseline_pred_path="manifest", baseline_val_path="manifest",
                     target=(summary.get("task_spec") or {}).get("target_column") or "target")


def judge_eval_inputs(gm):
    """ScoreResult/BaselineComparison aus den geloggten Gate-Metriken; None bei Formfehler
    (die Diagnose des Scorers steht nicht im Event-Log -> nicht rekonstruierbar)."""
    if not gm.get("scored_ok", False):
        return None
    score = ScoreResult(True, "", gm["f1"], gm["accuracy"])
    comp = None
    if gm.get("baseline_f1") is not None:
        comp = BaselineComparison(gm["f1"], gm["baseline_f1"], gm["diff"], gm["ci_lo"], gm["ci_hi"],
                                  gm["noninferior"])
    return score, comp


def replay_fsm(run_dir, summary, events, *, record=None):
    """Spielt die Zustandsfolge von phases.make_phase_node nach und erzeugt jeden Prompt neu.
    Gibt (worker_ok, judge_ok, judge_skipped) zurueck; record[(phase, attempt)] = Texte."""
    task = summary["task_prompt"]
    cfg = judge_cfg(summary, summary["arm"])
    st = {"current_phase": None, "gate_passed": None, "gate_feedback": "", "task_spec": None,
          "du_report": None}
    pos = {}                       # session -> naechste Nachricht
    n_w = n_j = n_skip = 0
    for e in events:
        ph = e["phase"]
        spec = st["task_spec"] or {}
        target = spec.get("target_column") or cfg.target
        crit = spec.get("critical_columns")
        resume = bool(e["resume"])
        if st["current_phase"] == ph and not st["gate_passed"]:
            kind = "own"
        else:
            kind = "earlier" if st["gate_passed"] else "later"
        exp = worker_prompt(ph, task, target, crit, st["du_report"], resume, st["gate_feedback"], kind)
        sid = e["session_id"]
        got = session_prompts(run_dir, sid)
        i = pos.get(sid, 0)
        assert i < len(got), f"{run_dir}: Session {sid} hat zu wenige Prompts"
        assert same_prompt(got[i], exp), f"{run_dir}: Worker-Prompt {ph} Versuch {e['attempt']} weicht ab"
        pos[sid] = i + 1
        n_w += 1
        if record is not None:
            record.setdefault(("worker", ph), dict(task=task, target=target, crit=crit,
                                                   du_report=copy.deepcopy(st["du_report"]), text=exp))
        if e.get("gate_session_id"):
            gm = e.get("gate_metrics") or {}
            if ph == "eval":
                inp = judge_eval_inputs(gm)
                if inp is None:
                    n_skip += 1
                    jexp = None
                else:
                    jexp = softgates.judge_prompt("eval", cfg, score=inp[0], comparison=inp[1], target=target)
            else:
                jexp = softgates.judge_prompt(ph, cfg, target=target)
            if jexp is not None:
                jgot = session_prompts(run_dir, e["gate_session_id"])
                assert len(jgot) == 1 and same_prompt(jgot[0], jexp), \
                    f"{run_dir}: Pruefer-Prompt {ph} Versuch {e['attempt']} weicht ab"
                n_j += 1
                if record is not None:
                    record.setdefault(("judge", ph), dict(target=target, gm=gm, text=jexp, cfg=cfg))
        # Zustand wie phases.py fortschreiben
        gm = e.get("gate_metrics") or {}
        if ph == "bu" and e["gate_passed"] and gm.get("spec") is not None:
            st["task_spec"] = gm["spec"]
        if ph == "du" and e["gate_passed"] and gm.get("report") is not None:
            st["du_report"] = gm["report"]
        st.update(current_phase=ph, gate_passed=e["gate_passed"], gate_feedback=e["gate_feedback"])
    # keine Prompts im Transkript, die das Replay nicht erklaert
    for sid, i in pos.items():
        assert len(session_prompts(run_dir, sid)) == i, f"{run_dir}: Session {sid} hat ueberzaehlige Prompts"
    return n_w, n_j, n_skip


def replay_react(run_dir, summary, events):
    task = summary["task_prompt"]
    exp = (prompts.crisp_react_prompt(task) if summary["arm"] == "crisp_react" else prompts.react_prompt(task))
    assert len(events) == 1 and events[0]["phase"] == "react", run_dir
    got = session_prompts(run_dir, events[0]["session_id"])
    assert len(got) == 1 and same_prompt(got[0], exp), f"{run_dir}: ReAct-Prompt weicht ab"
    return exp


def replay_all(runs_dir, run_ids, core_man, post_man):
    tot = {"worker": 0, "judge": 0, "judge_skipped_form_error": 0, "runs": 0}
    by_arm = {a: 0 for a in ARMS}
    for rid in run_ids:
        d = os.path.join(runs_dir, rid)
        s = jload(os.path.join(d, "summary.json"))
        ds = rid.split("-")[1]
        man = core_man.get(ds) or post_man[ds]
        assert s["task_prompt"] == man["task_prompt"], f"{rid}: task_prompt != Manifest"
        ev = load_events(d)
        if s["arm"] in ("hard_fsm", "soft_fsm"):
            w, j, k = replay_fsm(d, s, ev)
        else:
            replay_react(d, s, ev)
            w, j, k = 1, 0, 0
        tot["worker"] += w
        tot["judge"] += j
        tot["judge_skipped_form_error"] += k
        tot["runs"] += 1
        by_arm[s["arm"]] += w
    tot["worker_by_arm"] = by_arm
    return tot


# =========================================================================== Rendern mit Markierung

def mark(x: str) -> str:
    return MO + x + MC


def unmark(t: str) -> str:
    return t.replace(MO, "").replace(MC, "")


def marked_du_report(rep):
    if not rep:
        return rep
    r = copy.deepcopy(rep)
    r["quality_issues"] = [mark(i) for i in (r.get("quality_issues") or [])]
    r["excluded_columns"] = [{"column": mark(e["column"]), "reason": mark(e.get("reason", ""))}
                             for e in (r.get("excluded_columns") or [])]
    return r


def elide_crisp(text: str, keys) -> str:
    """Ersetzt die CRISP-DM-Bloecke durch eine Platzhalter-Zeile (jeder genau einmal, Zeilenanfang)."""
    for k in keys:
        block = prompts.CRISP_TASKS[k]
        assert text.count(block) == 1, k
        i = text.index(block)
        assert i == 0 or text[i - 1] == "\n", k
        text = text.replace(block, PO + k + PC + "\n")
    return text


def mark_measurement(text: str) -> str:
    """Eval-Pruefer: die Zahlen der Harness-Messung sind Run-Werte (Zeile 'MEASUREMENT:')."""
    lines = text.split("\n")
    for n, line in enumerate(lines):
        if line.lstrip().startswith("MEASUREMENT:"):
            lines[n] = re.sub(r"[+-]?\d+\.\d{4}", lambda m: mark(m.group(0)), line)
    return "\n".join(lines)


# =========================================================================== LaTeX

# Nicht-ASCII-Zeichen: Textzeichen, die inputenc/T1 in Courier kennen, bleiben; Mathe-Zeichen ueber
# \AppCMath. Gedankenstriche sehen in Courier (T1) wie '--' bzw. '-' aus und stehen darum in der
# Grundschrift (\AppCDash, \AppCNDash); der Geviertstrich ist breiter als ein Courier-Zeichen und
# zaehlt beim Umbruch doppelt (WIDE).
TEXT_CHARS = set("\u2018\u2019\u201c\u201d\u2026") | {chr(i) for i in range(0xC0, 0x100)}
MATH_CHARS = {"\u2265": "geq", "\u2264": "leq", "\u2248": "approx", "\u2192": "rightarrow",
              "\u00d7": "times", "\u2212": "-", "\u2260": "neq", "\u00b1": "pm"}
DASHES = {"\u2014": "AppCDash", "\u2013": "AppCNDash"}
WIDE = {"\u2014"}


def latex_char(c: str) -> str:
    if c == "'":
        return ESC + "AppCQ" + BGRP + EGRP
    if c in DASHES:
        return ESC + DASHES[c] + BGRP + EGRP
    if c in MATH_CHARS:
        sym = MATH_CHARS[c]
        return ESC + "AppCMath" + BGRP + (sym if sym == "-" else ESC + sym) + EGRP
    assert c not in (ESC, BGRP, EGRP), repr(c)
    assert ord(c) < 128 or c in TEXT_CHARS, f"Zeichen ohne Satzregel: {c!r}"
    return c


def wrap_chars(chars):
    """chars: Liste (zeichen, markiert). Greedy-Umbruch am letzten Leerzeichen, so dass die Zeile
    hoechstens WRAP Courier-Zeichenbreiten belegt (WIDE-Zeichen zaehlen doppelt)."""
    def width(seq):
        return sum(2 if c in WIDE else 1 for c, _ in seq)

    segs = []
    while width(chars) > WRAP:
        fit = 0                                   # laengstes Praefix mit Breite <= WRAP
        while width(chars[:fit + 1]) <= WRAP:
            fit += 1
        cut = max((i for i in range(1, fit + 1) if chars[i][0] == " "), default=None)
        if cut is None:
            segs.append(chars[:fit])
            chars = chars[fit:]
        else:
            segs.append(chars[:cut])
            chars = chars[cut + 1:]
    segs.append(chars)
    assert all(width(s) <= WRAP for s in segs)
    return segs


def to_verbatim(text: str):
    """Klartext mit Markierungen -> Zeilen fuer die AppCPrompt-Umgebung."""
    if text.endswith("\n"):
        text = text[:-1]
    out, marked = [], False
    for line in text.split("\n"):
        m = re.fullmatch(PO + r"(\w+)" + PC, line)
        if m:
            if m.group(1) == "rest":
                out.append(f"{ESC}AppCRest{BGRP}{EGRP}")
            else:
                assert m.group(1) in prompts.CRISP_TASKS, m.group(1)
                out.append(f"{ESC}AppCCrisp{BGRP}{m.group(1)}{EGRP}")
            continue
        assert PO not in line and PC not in line, line
        chars = []
        for c in line:
            if c == MO:
                marked = True
            elif c == MC:
                marked = False
            else:
                chars.append((c, marked))
        for k, seg in enumerate(wrap_chars(chars)):
            s = (ESC + "AppCCont" + BGRP + EGRP) if k else ""
            run, cur = [], None
            for c, mk in seg + [(None, None)]:
                if mk != cur and run:
                    body = "".join(latex_char(x) for x in run)
                    s += (ESC + "AppCVal" + BGRP + body + EGRP) if cur else body
                    run = []
                cur = mk
                if c is not None:
                    run.append(c)
            out.append(s)
    return out


def verbatim_block(text: str) -> str:
    return "\\begin{AppCPrompt}\n" + "\n".join(to_verbatim(text)) + "\n\\end{AppCPrompt}\n"


def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


HEADER = ("%!TEX root = ../../thesis.tex\n"
          "% generiert von thesis/scripts/make_appendix_prompts.py — nicht von Hand aendern\n")


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".tex")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + body)
    print(f"  -> {os.path.relpath(path, REPO)}")


def thousands(k: int) -> str:
    return f"{k:,}".replace(",", "{,}")


def setup_tex(rep):
    cc = "".join("\\" + c for c in (ESC, BGRP, EGRP))
    return (
        "% Zahlen aus dem Replay (prompts_check.json) fuer den Einleitungstext von Anhang C.\n"
        f"\\providecommand{{\\AppCRuns}}{{{rep['runs']}}}\n"
        f"\\providecommand{{\\AppCWorkerCalls}}{{{thousands(rep['worker'])}}}\n"
        f"\\providecommand{{\\AppCJudgeCalls}}{{{thousands(rep['judge'])}}}\n"
        f"\\providecommand{{\\AppCJudgeSkipped}}{{{rep['judge_skipped_form_error']}}}\n"
        f"\\providecommand{{\\AppCWrap}}{{{WRAP}}}\n"
        "\\providecommand{\\AppCRest}{\\textrm{[\\,followed by the complete prompt of the phase "
        "as printed above\\,]}}\n"
        "% Satz der Prompts: Verbatim mit commandchars " + ESC + BGRP + EGRP + " (kommen in keinem Prompt vor).\n"
        f"\\DefineVerbatimEnvironment{{AppCPrompt}}{{Verbatim}}{{fontsize={FONTSIZE},commandchars={cc},"
        "frame=lines,framesep=1.5mm,rulecolor=\\color[gray]{0.6}}\n"
        "\\providecommand{\\AppCVal}[1]{\\textsl{#1}}\n"
        "\\providecommand{\\AppCQ}{\\textquotesingle}\n"
        "\\providecommand{\\AppCMath}[1]{\\ensuremath{#1}}\n"
        "\\providecommand{\\AppCDash}{\\textrm{\\textemdash}}\n"
        "\\providecommand{\\AppCNDash}{\\textrm{\\textendash}}\n"
        "\\providecommand{\\AppCCont}{\\makebox[0pt][r]{\\ensuremath{\\hookrightarrow}\\,}}\n"
        # {\ttfamily} statt \texttt: mit der Praeambel der Arbeit setzte \texttt die Groesse zurueck,
        # der Rest des Hinweises stand dann in Normalgroesse (Sichtpruefung 24.11., im Minimalbeispiel nachgestellt)
        "\\providecommand{\\AppCCrisp}[1]{{" + FONTSIZE + "\\rmfamily[\\,CRISP-DM 1.0 text of this phase "
        "(block {\\ttfamily #1}), quoted in \\autoref{app:C:CrispText}\\,]}}\n"
    )


def crisp_tex():
    parts = []
    for i, k in enumerate(PHASES, 1):
        parts.append(f"\\subsection*{{Phase {i}: {PHASE_TITLE[k]} (generic tasks {CRISP_RANGE[k]}, "
                     f"block {tt(k)})}}\n")
        parts.append(verbatim_block(prompts.CRISP_TASKS[k]))
    return "".join(parts)


def workers_tex(rec):
    parts = []
    funcs = {"bu": "bu_initial_prompt", "du": "du_initial_prompt", "prep": "initial_prompt",
             "model": "initial_prompt", "eval": "initial_prompt"}
    for i, ph in enumerate(PHASES, 1):
        r = rec[("worker", ph)]
        task_m = mark(r["task"])
        target_m = mark(r["target"])
        crit_m = [mark(c) for c in r["crit"]] if r["crit"] else r["crit"]
        rep_m = marked_du_report(r["du_report"])
        text = worker_prompt(ph, task_m, target_m, crit_m, rep_m, False, "", "own")
        assert unmark(text) == r["text"], ph
        parts.append(f"\\subsection*{{Phase {i}: {PHASE_TITLE[ph]} "
                     f"({tt('prompts.' + funcs[ph])})}}\n")
        parts.append(verbatim_block(elide_crisp(text, [ph])))
    return "".join(parts)


def resume_tex(rec):
    """Praeambeln fuer Wiederholungen: Platzhalter {feedback} bleibt sichtbar."""
    fb = mark("{feedback}")
    r = {ph: rec[("worker", ph)] for ph in PHASES}
    heads = {}
    for ph in PHASES:
        x = r[ph]
        init = worker_prompt(ph, x["task"], x["target"], x["crit"], x["du_report"], False, "", "own")
        for kind in prompts.RESUME_KINDS:
            full = worker_prompt(ph, x["task"], x["target"], x["crit"], x["du_report"], True, fb, kind)
            assert full.endswith(init), (ph, kind)
            heads[(ph, kind)] = full[: len(full) - len(init)]
    # 'later'/'earlier' identisch fuer alle Phasen; 'own' identisch fuer prep/model/eval
    for kind in ("later", "earlier"):
        assert len({heads[(ph, kind)] for ph in PHASES}) == 1, kind
    assert len({heads[(ph, "own")] for ph in ("prep", "model", "eval")}) == 1
    items = [
        ("Retry after the phase's own gate failed, Business Understanding", heads[("bu", "own")]),
        ("Retry after the phase's own gate failed, Data Understanding", heads[("du", "own")]),
        ("Retry after the phase's own gate failed, Data Preparation, Modeling and Evaluation",
         heads[("prep", "own")]),
        ("Re-entry after a later gate sent the process back (Evaluation to Modeling, "
         "specification mismatch to Business Understanding)", heads[("bu", "later")]),
        ("Re-entry after an earlier phase was revised (e.g., Evaluation after a new model)",
         heads[("bu", "earlier")]),
    ]
    parts = []
    for title, head in items:
        assert head.endswith("\n\n"), title          # Leerzeile, dann der vollstaendige Phasen-Prompt
        # bedingter Seitenumbruch: Ueberschrift nie allein am Seitenende (Sichtpruefung 24.11.)
        parts.append("\\par\\vspace{0pt plus 8\\baselineskip}\\penalty-300\\vspace{0pt plus -8\\baselineskip}\n")
        parts.append(f"\\paragraph{{{title}.}}\\leavevmode\n")
        parts.append(verbatim_block(head + PO + "rest" + PC + "\n"))
    return "".join(parts)


def judges_tex(rec):
    parts = []
    for i, ph in enumerate(PHASES, 1):
        r = rec[("judge", ph)]
        if ph == "eval":
            inp = judge_eval_inputs(r["gm"])
            text = mark_measurement(softgates.judge_prompt("eval", r["cfg"], score=inp[0], comparison=inp[1],
                                                           target=r["target"]))
        else:
            text = softgates.judge_prompt(ph, r["cfg"], target=mark(r["target"]))
        assert unmark(text) == r["text"], ph
        parts.append(f"\\subsection*{{Gate {i}: {PHASE_TITLE[ph]} "
                     f"({tt('softgates.judge_prompt')}, phase {tt(ph)})}}\n")
        parts.append(verbatim_block(text))
    return "".join(parts)


def react_tex(task):
    d_plain = prompts.react_prompt(task)
    c_plain = prompts.crisp_react_prompt(task)
    assert c_plain == d_plain + prompts.CRISP_GUIDANCE
    d_marked = prompts.react_prompt(mark(task))
    assert unmark(d_marked) == d_plain
    g = prompts.CRISP_GUIDANCE
    pos = [g.index(prompts.CRISP_TASKS[k]) for k in PHASES]
    assert pos == sorted(pos)
    parts = [f"\\subsection*{{Arm D: complete prompt ({tt('prompts.react_prompt')})}}\n",
             verbatim_block(d_marked),
             f"\\subsection*{{Arm C: block appended to the prompt of Arm D ({tt('prompts.CRISP_GUIDANCE')})}}\n",
             verbatim_block(elide_crisp(g, PHASES))]
    return "".join(parts)


def tasks_tex(core_man, post_man):
    from make_tables import CORE_ORDER, NAME, POST_ORDER   # gleiche Reihenfolge/Namen wie Kap. 6
    rows = []
    for ds in CORE_ORDER + POST_ORDER:
        t = (core_man.get(ds) or post_man[ds])["task_prompt"]
        assert not any(c in t for c in "#$%&_~^\\{}"), t      # Text geht unveraendert in LaTeX
        rows.append(f"    {NAME[ds]} & {t} \\\\")
        if ds == CORE_ORDER[-1]:
            rows.append("    \\midrule")
    # [!t] ohne p: sonst setzt LaTeX die Tabelle allein auf eine Float-Seite (2/3 leer)
    return ("\\begin{table}[!t]\n  \\centering\n  \\small\n"
            "  \\caption[Task descriptions]{Task descriptions inserted for \\texttt{\\{task\\_prompt\\}} "
            "into the prompts of all arms, from the dataset manifests; identical to the field "
            "\\texttt{task\\_prompt} in the summary of every run. Core suite in tercile order, then the "
            "post-cutoff datasets.}\n"
            "  \\label{tab:C:tasks}\n"
            "  \\begin{tabularx}{\\textwidth}{lL}\n    \\toprule\n"
            "    Dataset & Task description \\\\\n    \\midrule\n"
            + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabularx}\n\\end{table}\n")


# =========================================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(REPO), "thesis-crisp-dm-runs"))
    args = ap.parse_args()
    runs_dir = os.path.abspath(args.runs_dir)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    core_man, post_man = manifests()
    run_ids = clean_runs(runs_dir, set(core_man) | set(post_man))
    assert len(run_ids) == 360, len(run_ids)
    code = check_code_state(runs_dir, run_ids)
    print("Code-Stand:", code)
    assert ALLOWED_TOOLS == softgates.JUDGE_ALLOWED_TOOLS

    print("Replay aller Prompts ...")
    rep = replay_all(runs_dir, run_ids, core_man, post_man)
    rep["line_endings_in_transcripts"] = dict(EOL_SEEN)
    assert sum(EOL_SEEN.values()) == rep["worker"] + rep["judge"]
    # c-prompts.tex sagt: kein Eval-Pruefer sah einen Formfehler, alle Pruefer-Prompts sind geprueft
    assert rep["judge_skipped_form_error"] == 0 and rep["judge"] == rep["worker_by_arm"]["soft_fsm"], rep
    print("Replay:", rep)

    # Beispiel-Runs: Werte fuers Rendern
    rec = {}
    for arm in ("hard_fsm", "soft_fsm"):
        d = os.path.join(runs_dir, f"main-{EXAMPLE}-{arm}-r{EXAMPLE_REP}")
        s = jload(os.path.join(d, "summary.json"))
        assert all(v <= 1 for v in s["attempts"].values()), s["attempts"]   # nur Erst-Prompts
        r = {}
        replay_fsm(d, s, load_events(d), record=r)
        if arm == "hard_fsm":
            rec.update({k: v for k, v in r.items() if k[0] == "worker"})
        else:
            rec.update({k: v for k, v in r.items() if k[0] == "judge"})
            # Arm B bekommt dieselben Worker-Prompt-Funktionen (nur andere Run-Werte)
    task = rec[("worker", "bu")]["task"]
    for arm in ("crisp_react", "naked_react"):
        d = os.path.join(runs_dir, f"main-{EXAMPLE}-{arm}-r{EXAMPLE_REP}")
        replay_react(d, jload(os.path.join(d, "summary.json")), load_events(d))

    # Zeichensatz: commandchars und Markierungen kommen in keinem gerenderten Text vor
    texts = [v["text"] for v in rec.values()] + [prompts.crisp_react_prompt(task)]
    for t in texts:
        for c in (ESC, BGRP, EGRP, MO, MC, PO, PC):
            assert c not in t, repr(c)

    write("prompts_setup", setup_tex(rep))
    write("prompts_crisp", crisp_tex())
    write("prompts_workers", workers_tex(rec))
    write("prompts_resume", resume_tex(rec))
    write("prompts_judges", judges_tex(rec))
    write("prompts_react", react_tex(task))
    write("prompts_tasks", tasks_tex(core_man, post_man))
    summary = {"harness": HARNESS, "code": code, "replay": rep, "example_runs":
               [f"main-{EXAMPLE}-{a}-r{EXAMPLE_REP}" for a in ARMS], "wrap": WRAP}
    with open(os.path.join(OUT, "prompts_check.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    print("  -> thesis/appendix/generated/prompts_check.json")


if __name__ == "__main__":
    main()
