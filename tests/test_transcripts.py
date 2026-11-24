# tests/test_transcripts.py
"""Worker-Transkripte sichern + Zugriffs-Audit (Review 24.10., M2/M5): Claude Code raeumt
Transkripte nach ~30 Tagen weg, und holdout_intact erkennt nur Schreib-, keine Lesezugriffe.
Synthetische Transkripte im Format von ~/.claude/projects/<key>/<session>.jsonl."""
import json
import os
import time

from crispfsm.transcripts import audit_access, collect_transcripts, project_key


def _tool_use(name, **inp):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": name, "input": inp}]}}


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_project_key_matches_claude_code_scheme():
    """Wie die CLI den Projektordner benennt (beobachtet: jedes Nicht-Alnum -> '-')."""
    assert (project_key(r"C:\projects\thesis-crisp-dm-runs\leak-adult-naked_react-r1\work")
            == "C--projects-thesis-crisp-dm-runs-leak-adult-naked-react-r1-work")


def test_collect_copies_only_transcripts_of_this_run(tmp_path):
    """Archivierte Runs mit derselben run_id teilen den Projektordner (Review G4) -> nur
    Transkripte ab Run-Start kopieren."""
    work = tmp_path / "runs" / "leak-x-r1" / "work"
    root = tmp_path / "projects"
    src = root / project_key(str(work))
    src.mkdir(parents=True)
    old = src / "old-session.jsonl"
    _write_jsonl(old, [{"type": "user"}])
    past = time.time() - 3600
    os.utime(old, (past, past))
    started = time.time() - 5
    _write_jsonl(src / "new-session.jsonl", [{"type": "user"}])

    dest = tmp_path / "runs" / "leak-x-r1" / "transcripts"
    names = collect_transcripts(str(work), str(dest), since=started, projects_root=str(root))
    assert names == ["new-session.jsonl"]
    assert (dest / "new-session.jsonl").exists() and not (dest / "old-session.jsonl").exists()


def test_collect_without_project_dir_returns_empty(tmp_path):
    assert collect_transcripts(str(tmp_path / "w"), str(tmp_path / "d"), since=0,
                               projects_root=str(tmp_path / "none")) == []


def test_audit_accepts_legit_access(tmp_path):
    t = tmp_path / "s.jsonl"
    _write_jsonl(t, [
        _tool_use("Bash", command="python -c \"import pandas as pd; pd.read_parquet('data/train.parquet')\""),
        _tool_use("Read", file_path=r"C:\projects\thesis-crisp-dm-runs\leak-x-r1\work\data\train.parquet"),
        _tool_use("Bash", command="ls ../holdout/test_features.parquet"),
        {"type": "user", "message": {"content": "holdout/test.parquet steht nur im Prompt"}},
    ])
    a = audit_access([str(t)], run_dir_name="leak-x-r1")
    assert a["flagged"] is False
    assert sum(a["counts"].values()) == 0


def test_audit_flags_reads_outside_the_run(tmp_path):
    t = tmp_path / "s.jsonl"
    _write_jsonl(t, [
        _tool_use("Bash", command="cat ../holdout/test.parquet"),
        _tool_use("Read", file_path=r"C:\projects\thesis-crisp-dm-runs\main-adult-hard_fsm-r1\work\notes.md"),
        _tool_use("Bash", command="python -c 'from crispfsm.datasets import prepare_from_manifest'"),
        _tool_use("Bash", command="ls ../../automem_main-adult-hard_fsm-r1"),
    ])
    a = audit_access([str(t)], run_dir_name="leak-x-r1")
    assert a["flagged"] is True
    c = a["counts"]
    assert c["holdout_labels"] == 1 and c["other_runs"] == 1
    assert c["harness_repo"] == 1 and c["archives"] == 1 and c["parent_escape"] == 1
    assert a["examples"]


def test_audit_classifies_claude_state_not_as_other_run(tmp_path):
    """Echte Transkripte (automem-Runs): Worker schrieben ins Projekt-Gedaechtnis unter
    ~/.claude/projects/C--projects-thesis-crisp-dm-runs/memory — eigene Kategorie, kein
    'anderer Run' (der Projekt-Schluessel enthaelt den Repo-Namen)."""
    t = tmp_path / "s.jsonl"
    _write_jsonl(t, [_tool_use("Write", file_path=(
        r"C:\Users\A\.claude\projects\C--projects-thesis-crisp-dm-runs\memory\x.md"))])
    a = audit_access([str(t)], run_dir_name="leak-x-r1")
    assert a["counts"]["claude_state"] == 1
    assert a["counts"]["other_runs"] == 0
