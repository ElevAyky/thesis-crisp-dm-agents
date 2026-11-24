"""Facts: Zugriff auf die Hold-out-Features nach einem Ruecksprung Evaluation -> Modeling
-> thesis/facts/derived/eval_return_access.json.

Hintergrund (Kap. 4.4): In Arm A/B (und A-Single) wird data/test_features.parquet erst zu
Beginn der Evaluation ins Arbeitsverzeichnis kopiert und danach nicht entfernt. Springt ein
Run von Evaluation zurueck zu Modeling, liegt die Datei in der wiederholten Modeling-Phase
also bereit. Dieses Skript prueft fuer alle Runs mit einem solchen Ruecksprung (Hauptlaeufe
und zweite Stufe), ob die wiederholte Modeling-Phase die Datei in einem Tool-Aufruf nennt.

Verfahren: Worker-Aufrufe aus events.jsonl (Eintraege mit session_id) in Reihenfolge; jeder
Model-Aufruf nach dem ersten Eval-Aufruf gehoert zur wiederholten Modeling-Phase. Pro Aufruf
wird im Transkript seiner Session das Prompt-Fenster bestimmt (k-ter Prompt der Session =
k-ter Aufruf dieser Session; Prompts = user-Nachrichten mit Text, nicht tool_result) und alle
tool_use-Eingaben dieses Fensters auf "test_features" durchsucht. In A-Single liegen alle
Phasen in einer Session; das Fenster trennt Modeling dort von der legitimen Eval-Nutzung.

Aufruf (Repo-Root):
    uv run python thesis/scripts/collect_facts_eval_return.py
"""
from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DER = os.path.join(REPO, "thesis", "facts", "derived")
NEEDLE = "test_features"


def worker_calls(run_dir):
    calls = []
    with open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8") as f:
        for ln in f:
            e = json.loads(ln)
            if e.get("phase") and e.get("session_id"):
                calls.append((e["phase"], e["session_id"]))
    return calls


def prompt_windows(transcript):
    """tool_use-Eingaben je Prompt-Index (1-basiert) einer Session."""
    windows, k = {}, 0
    with open(transcript, encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message") or {}
            content = msg.get("content")
            if rec.get("type") == "user":
                if isinstance(content, str) or (isinstance(content, list) and any(
                        isinstance(i, dict) and i.get("type") == "text" for i in content)):
                    k += 1
                    windows.setdefault(k, [])
                continue
            if isinstance(content, list):
                for i in content:
                    if isinstance(i, dict) and i.get("type") == "tool_use":
                        windows.setdefault(k, []).append(json.dumps(i.get("input"), ensure_ascii=False))
    return windows


def check_run(run_dir):
    calls = worker_calls(run_dir)
    phases = [p for p, _ in calls]
    if not any(a == "eval" and b == "model" for a, b in zip(phases, phases[1:])):
        return None
    first_eval = phases.index("eval")
    seen, windows_cache = {}, {}
    n_calls = n_mentions = 0
    mentions = []
    model_calls_after = 0
    for idx, (phase, sid) in enumerate(calls):
        seen[sid] = seen.get(sid, 0) + 1
        if phase != "model" or idx < first_eval:
            continue
        model_calls_after += 1
        if sid not in windows_cache:
            windows_cache[sid] = prompt_windows(os.path.join(run_dir, "transcripts", sid + ".jsonl"))
        win = windows_cache[sid].get(seen[sid], [])
        n_calls += len(win)
        for s in win:
            if NEEDLE in s:
                n_mentions += 1
                mentions.append(s[:300])
    # Plausibilitaet: Prompts je Session == Aufrufe je Session
    prompts_match = all(max(windows_cache[s].keys(), default=0) == seen[s] for s in windows_cache)
    return {"model_calls_after_return": model_calls_after, "tool_calls": n_calls,
            "mentions": n_mentions, "mention_examples": mentions[:3], "prompts_match_calls": prompts_match}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(REPO), "thesis-crisp-dm-runs"))
    args = ap.parse_args()
    out = {}
    for prefix in ("main", "h1"):
        rows = {}
        for d in sorted(os.listdir(args.runs_dir)):
            if not d.startswith(prefix + "-") or not any(a in d for a in ("-hard_fsm", "-soft_fsm")):
                continue
            rd = os.path.join(args.runs_dir, d)
            if not os.path.exists(os.path.join(rd, "events.jsonl")):
                continue
            r = check_run(rd)
            if r is not None:
                rows[d] = r
        out[prefix] = {"runs_with_eval_to_model": sorted(rows), "per_run": rows,
                       "runs_mentioning": sorted(k for k, v in rows.items() if v["mentions"]),
                       "tool_calls_min_max": [min((v["tool_calls"] for v in rows.values()), default=0),
                                              max((v["tool_calls"] for v in rows.values()), default=0)]}
    os.makedirs(DER, exist_ok=True)
    with open(os.path.join(DER, "eval_return_access.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, v in out.items():
        print(k, len(v["runs_with_eval_to_model"]), "runs; mentioning:", v["runs_mentioning"],
              "; tool calls min/max", v["tool_calls_min_max"],
              "; prompts match:", all(x["prompts_match_calls"] for x in v["per_run"].values()))


if __name__ == "__main__":
    main()
