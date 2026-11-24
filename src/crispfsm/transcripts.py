# src/crispfsm/transcripts.py
"""Worker-Transkripte sichern und auf Zugriffe ausserhalb des eigenen Runs pruefen
(Review 24.10., M2/M5).

M5: Claude Code legt pro Arbeitsordner Transkripte unter ~/.claude/projects/<key>/ ab und
raeumt sie nach ~30 Tagen weg — die Kern-C/D-Transkripte vom 23.09. waren am 24.10. schon
fort. Darum wandert jedes Transkript eines Runs in den Run-Ordner (-> runs-Repo).

M2: holdout_intact erkennt nur Schreib-, keine Lesezugriffe. Die Tool-Aufrufe der Worker
werden deshalb nach Mustern durchsucht, die auf Wissen ausserhalb des Runs zeigen:
Test-/Validierungs-Labels, andere Runs, das Harness-Repo, Archive, Ausbruch per ../.. ."""
from __future__ import annotations
import json
import os
import re
import shutil

PROJECTS_ROOT = os.path.join(os.path.expanduser("~"), ".claude", "projects")

PATTERNS = {
    "holdout_labels": re.compile(
        r"holdout[\\/]+(?:test|val)\.parquet|eval_index\.json|baseline_(?:test|val)\.parquet"),
    "harness_repo": re.compile(r"thesis-crisp-dm-agents|crispfsm"),
    "archives": re.compile(r"automem_|gitctx_|promptv1_|_auto_memory|memory\.automem"),
    "parent_escape": re.compile(r"\.\.[\\/]+\.\."),
    # Claude-Code-Zustand (Gedaechtnis, Transkripte anderer Runs) unter ~/.claude/projects
    "claude_state": re.compile(r"\.claude[\\/]+projects"),
}
# Pfad in einen Run-Ordner des runs-Repos; nicht der Projekt-Schluessel
# "C--projects-thesis-crisp-dm-runs" unter ~/.claude (dort steht ein '-' davor).
_RUN_PATH = re.compile(r"(?<![-\w])thesis-crisp-dm-runs[\\/]+([^\\/\"'\s]+)")


def project_key(path: str) -> str:
    """Name des Projektordners, den die CLI fuer ein cwd anlegt (jedes Nicht-Alnum -> '-')."""
    return "".join(c if c.isalnum() else "-" for c in os.path.abspath(path))


def collect_transcripts(work_dir: str, dest_dir: str, *, since: float,
                        projects_root: str | None = None) -> list[str]:
    """Kopiert die Transkripte dieses Runs (geaendert ab `since`) nach dest_dir. Aeltere
    Dateien im selben Projektordner stammen von archivierten Runs gleicher run_id."""
    src = os.path.join(projects_root or PROJECTS_ROOT, project_key(work_dir))
    if not os.path.isdir(src):
        return []
    names = sorted(n for n in os.listdir(src)
                   if n.endswith(".jsonl") and os.path.getmtime(os.path.join(src, n)) >= since)
    if names:
        os.makedirs(dest_dir, exist_ok=True)
    for n in names:
        shutil.copy2(os.path.join(src, n), os.path.join(dest_dir, n))
    return names


def _tool_inputs(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("type") != "assistant":
                continue
            content = (rec.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield json.dumps(block.get("input"), ensure_ascii=False)


def audit_access(transcript_paths: list[str], *, run_dir_name: str) -> dict:
    """Zaehlt Tool-Aufrufe mit Hinweis auf Zugriffe ausserhalb des eigenen Runs. Nur
    Tool-Eingaben, nicht Prompt/Systemprompt (dort stehen Pfade legitim)."""
    counts = {k: 0 for k in PATTERNS} | {"other_runs": 0}
    examples = []
    for path in transcript_paths:
        for text in _tool_inputs(path):
            hits = [k for k, rx in PATTERNS.items() if rx.search(text)]
            if any(m != run_dir_name for m in _RUN_PATH.findall(text)):
                hits.append("other_runs")
            for k in hits:
                counts[k] += 1
            if hits and len(examples) < 5:
                examples.append({"kinds": hits, "input": text[:300]})
    return {"counts": counts, "flagged": any(counts.values()), "examples": examples}
