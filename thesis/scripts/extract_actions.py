#!/usr/bin/env python
"""Verdichtete Aktionslisten der Agenten-Laeufe (Grundlage der Prozess-Kodierung, Kap. 7.2).

Liest pro Run die Claude-Code-Transkripte (`<runs>/<run>/transcripts/*.jsonl`) und erzeugt
eine chronologische Liste: jede Tool-Aktion mit laufender Nummer, Typ, ausgefuehrtem Code
bzw. Datei-Inhalt (mit Zeilennummern) und gekuerzter Ausgabe; dazwischen der sichtbare
Assistant-Text. Thinking-Bloecke sind in den Transkripten leer (nur Signatur) und tauchen
deshalb nicht auf. Ausgabe: `thesis/facts/derived/actions/<run>.md` (gitignored, jederzeit
reproduzierbar).

Arme A/B: Die Transkripte enthalten pro Phase eine Worker-Session (Retries laufen in
derselben Session weiter) und bei B zusaetzlich die Sessions der LLM-Pruefer. Pruefer-
Sessions werden erkannt (erste Nachricht "You are the independent reviewer ...") und
weggelassen; die Phase einer Worker-Session kommt aus `events.jsonl` (session_id -> phase).
Sessions werden nach dem ersten Zeitstempel geordnet.

Nur saubere Runs (`main-*`). Archivierte Ordner (automem_*, gitctx_*, promptv1_*) werden nie
gelesen. Ins Runs-Repo wird nichts geschrieben.

Die Parse-Funktionen (`load_run`) nutzt auch `code_process.py`; beide sehen also exakt
dieselbe Aktionsfolge.

Aufruf aus dem Repo-Root:
    uv run python thesis/scripts/extract_actions.py [--runs-dir ../thesis-crisp-dm-runs]
        [--arms crisp_react naked_react hard_fsm soft_fsm] [--only <run_id> ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RUNS = os.path.join(os.path.dirname(REPO), "thesis-crisp-dm-runs")
OUT_DIR = os.path.join(REPO, "thesis", "facts", "derived", "actions")

ARMS = ["hard_fsm", "soft_fsm", "crisp_react", "naked_react"]
POST_CUTOFF = ("student_health", "ev_purchase")
RUN_RE = re.compile(r"^main-(?P<ds>.+)-(?P<arm>hard_fsm|soft_fsm|crisp_react|naked_react)-r(?P<rep>\d+)$")
JUDGE_PREFIX = "You are the independent reviewer"
OUT_HEAD, OUT_TAIL = 1500, 400      # Zeichen der Tool-Ausgabe im Markdown (Anfang / Ende)


@dataclass
class Entry:
    kind: str                 # "prompt" | "text" | "action"
    ts: str
    text: str = ""            # prompt/text
    n: int = 0                # laufende Aktionsnummer (nur action)
    tool: str = ""            # Bash | Write | Edit | Read
    inp: dict = field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    session: int = 0          # Index der Worker-Session (0-basiert, chronologisch)
    phase: str = "react"      # Phase der Session (A/B), sonst "react"

    @property
    def code(self) -> str:
        """Ausgefuehrter bzw. geschriebener Code/Text dieser Aktion (fuer den Kodierer)."""
        if self.tool == "Bash":
            return self.inp.get("command", "")
        if self.tool == "Write":
            return self.inp.get("content", "")
        if self.tool == "Edit":
            return self.inp.get("new_string", "")
        return ""

    @property
    def path(self) -> str:
        return self.inp.get("file_path", "")


@dataclass
class RunActions:
    run_id: str
    arm: str
    dataset: str
    rep: int
    summary: dict
    sessions: list            # [(session_id, phase)]
    entries: list             # [Entry] chronologisch

    @property
    def actions(self) -> list:
        return [e for e in self.entries if e.kind == "action"]

    @property
    def post_cutoff(self) -> bool:
        return self.dataset in POST_CUTOFF


def parse_run_id(run_id: str):
    m = RUN_RE.match(run_id)
    if not m:
        return None
    return m["ds"], m["arm"], int(m["rep"])


def list_runs(runs_dir: str, arms=ARMS) -> list[str]:
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "main-*"))):
        rid = os.path.basename(d)
        p = parse_run_id(rid)
        if p and p[1] in arms and os.path.isdir(os.path.join(d, "transcripts")):
            out.append(rid)
    return out


def _result_text(block: dict) -> str:
    c = block.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ""


def _read_session(path: str):
    """-> (first_ts, first_prompt, [raw entries]) einer Transkript-Datei."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    first_ts, first_prompt, raw = None, None, []
    for obj in rows:
        if obj.get("type") not in ("user", "assistant"):
            continue
        ts = obj.get("timestamp", "")
        first_ts = first_ts or ts
        msg = obj.get("message") or {}
        c = msg.get("content")
        if obj["type"] == "user" and isinstance(c, str):
            first_prompt = first_prompt if first_prompt is not None else c
            raw.append(("prompt", ts, c))
        elif isinstance(c, list):
            for b in c:
                bt = b.get("type")
                if bt == "text" and b.get("text", "").strip():
                    raw.append(("text", ts, b["text"]))
                elif bt == "tool_use":
                    raw.append(("tool_use", ts, b))
                elif bt == "tool_result":
                    raw.append(("tool_result", ts, b))
    return first_ts or "", first_prompt or "", raw


def load_run(runs_dir: str, run_id: str) -> RunActions:
    ds, arm, rep = parse_run_id(run_id)
    rdir = os.path.join(runs_dir, run_id)
    with open(os.path.join(rdir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    phase_of = {}
    ev_path = os.path.join(rdir, "events.jsonl")
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ev = json.loads(line)
                    if ev.get("session_id") and ev.get("phase"):
                        phase_of.setdefault(ev["session_id"], ev["phase"])
    sess = []
    for p in glob.glob(os.path.join(rdir, "transcripts", "*.jsonl")):
        sid = os.path.splitext(os.path.basename(p))[0]
        first_ts, first_prompt, raw = _read_session(p)
        if first_prompt.startswith(JUDGE_PREFIX):
            continue                                  # Pruefer-Session (Arm B)
        sess.append((first_ts, sid, raw))
    sess.sort(key=lambda t: (t[0], t[1]))
    entries, sessions, n = [], [], 0
    for si, (_, sid, raw) in enumerate(sess):
        phase = phase_of.get(sid, "react" if arm.endswith("react") else "?")
        sessions.append((sid, phase))
        pending = {}
        for kind, ts, payload in raw:
            if kind == "prompt":
                entries.append(Entry("prompt", ts, text=payload, session=si, phase=phase))
            elif kind == "text":
                entries.append(Entry("text", ts, text=payload, session=si, phase=phase))
            elif kind == "tool_use":
                n += 1
                e = Entry("action", ts, n=n, tool=payload.get("name", ""),
                          inp=payload.get("input") or {}, session=si, phase=phase)
                pending[payload.get("id")] = e
                entries.append(e)
            elif kind == "tool_result":
                e = pending.get(payload.get("tool_use_id"))
                if e is not None:
                    e.result = _result_text(payload)
                    e.is_error = bool(payload.get("is_error"))
    return RunActions(run_id, arm, ds, rep, summary, sessions, entries)


# ----------------------------------------------------------------------------- Markdown

def _numbered(code: str) -> str:
    lines = code.split("\n")
    w = len(str(len(lines)))
    return "\n".join(f"{i:>{w}}| {ln}" for i, ln in enumerate(lines, 1))


def _short(s: str) -> str:
    s = s.replace("\r\n", "\n")
    if len(s) <= OUT_HEAD + OUT_TAIL + 50:
        return s
    return s[:OUT_HEAD] + f"\n[... {len(s) - OUT_HEAD - OUT_TAIL} Zeichen gekuerzt ...]\n" + s[-OUT_TAIL:]


def _fence(s: str) -> str:
    """Zaun, der laenger ist als jede Backtick-Folge im Inhalt."""
    longest = max((len(m) for m in re.findall(r"`+", s)), default=0)
    return "`" * max(3, longest + 1)


def _lang(path: str) -> str:
    return {".py": "python", ".md": "markdown", ".json": "json", ".sh": "bash"}.get(
        os.path.splitext(path)[1].lower(), "")


def _rel(path: str, run_id: str) -> str:
    p = path.replace("\\", "/")
    key = f"{run_id}/work/"
    return p.split(key, 1)[1] if key in p else p


def render(ra: RunActions) -> str:
    s = ra.summary
    bc = s.get("baseline_comparison") or {}
    out = [f"# {ra.run_id}", ""]
    out.append(f"- Arm: {ra.arm} · Set: {ra.dataset} · r{ra.rep} · "
               f"{'Post-Cutoff' if ra.post_cutoff else 'Kern'}")
    f1 = s.get("best_f1_measured")
    out.append(f"- outcome: {s.get('outcome')} (fail_reason: {s.get('fail_reason')}) · "
               f"F1 {f1 if f1 is None else round(f1, 4)} · Baseline (Test) "
               f"{round(bc['baseline_f1'], 4) if bc.get('baseline_f1') is not None else '-'} · "
               f"Diff {round(bc['diff'], 4) if bc.get('diff') is not None else '-'}")
    out.append(f"- Worker-Sessions: {len(ra.sessions)} · Tool-Aktionen: {len(ra.actions)}")
    out.append("- Quelle: `transcripts/*.jsonl` (Pruefer-Sessions ausgelassen); Zeilennummern "
               "in Code-Bloecken = Zeile im geschriebenen Inhalt bzw. im Bash-Befehl.")
    out.append("")
    cur = None
    for e in ra.entries:
        if e.session != cur:
            cur = e.session
            sid, ph = ra.sessions[e.session]
            out += ["", f"## Session {e.session + 1} — Phase `{ph}` ({sid})", ""]
        if e.kind == "prompt":
            head = e.text.strip().split("\n", 1)[0]
            first = e.text.startswith("You are an ML engineer")
            label = "Auftrag (Prompt, gekuerzt)" if first else "Folge-Prompt"
            body = e.text[:300] + (" [...]" if len(e.text) > 300 else "") if first else e.text[:1200]
            out += [f"**{label}** ({e.ts}): {head[:120]}", "", "> " + body.replace("\n", "\n> "), ""]
        elif e.kind == "text":
            out += [f"**[Text]** {e.text.strip()}", ""]
        else:
            desc = e.inp.get("description", "")
            if e.tool == "Bash":
                out.append(f"### #{e.n} Bash — {desc}".rstrip(" —"))
                code = e.code
                f = _fence(code)
                out += [f"{f}bash", _numbered(code), f]
            elif e.tool == "Write":
                code = e.code
                out.append(f"### #{e.n} Write `{_rel(e.path, ra.run_id)}` ({len(code.splitlines())} Zeilen)")
                f = _fence(code)
                out += [f"{f}{_lang(e.path)}", _numbered(code), f]
            elif e.tool == "Edit":
                out.append(f"### #{e.n} Edit `{_rel(e.path, ra.run_id)}`")
                old, new = e.inp.get("old_string", ""), e.code
                f = _fence(old + new)
                out += ["alt:", f"{f}", old, f, "neu:", f"{f}{_lang(e.path)}", _numbered(new), f]
            else:
                out.append(f"### #{e.n} {e.tool} `{_rel(e.path, ra.run_id)}`")
            res = _short(e.result)
            f = _fence(res)
            out += [f"Ausgabe{' (FEHLER)' if e.is_error else ''}:", f"{f}text", res, f, ""]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--only", nargs="*", default=None, help="nur diese run_ids")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    runs = a.only if a.only else list_runs(a.runs_dir, a.arms)
    os.makedirs(a.out_dir, exist_ok=True)
    total = 0
    for rid in runs:
        ra = load_run(a.runs_dir, rid)
        with open(os.path.join(a.out_dir, f"{rid}.md"), "w", encoding="utf-8") as f:
            f.write(render(ra))
        total += len(ra.actions)
    print(f"{len(runs)} Runs, {total} Tool-Aktionen -> {os.path.relpath(a.out_dir, REPO)}")


if __name__ == "__main__":
    main()
