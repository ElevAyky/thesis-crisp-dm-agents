# tests/test_events.py
import json
import os

from crispfsm.events import log_event


def _read_lines(run_workdir):
    path = os.path.join(run_workdir, "events.jsonl")
    with open(path) as f:
        return f.read().splitlines()


def test_log_event_creates_missing_workdir(tmp_path):
    """run_workdir (including nested, non-existent dirs) is created by makedirs."""
    run_workdir = os.path.join(str(tmp_path), "runs", "abc123", "work")
    assert not os.path.exists(run_workdir)

    event = {"phase": "prep", "ok": True}
    log_event(run_workdir, event)

    assert os.path.isdir(run_workdir)
    assert os.path.isfile(os.path.join(run_workdir, "events.jsonl"))


def test_single_call_writes_exactly_one_valid_line(tmp_path):
    run_workdir = os.path.join(str(tmp_path), "run")
    event = {"phase": "model", "attempt": 1, "nested": {"a": [1, 2, 3]}}

    log_event(run_workdir, event)

    lines = _read_lines(run_workdir)
    assert len(lines) == 1
    assert json.loads(lines[0]) == event


def test_n_sequential_calls_write_n_ordered_lines(tmp_path):
    run_workdir = os.path.join(str(tmp_path), "run")
    events = [{"i": i, "phase": p} for i, p in enumerate(["prep", "model", "eval", "prep"])]

    for event in events:
        log_event(run_workdir, event)

    lines = _read_lines(run_workdir)
    assert len(lines) == len(events)
    for line, expected in zip(lines, events):
        assert json.loads(line) == expected


def test_append_does_not_clobber_prior_lines(tmp_path):
    run_workdir = os.path.join(str(tmp_path), "run")
    first = {"step": "first"}
    second = {"step": "second"}

    log_event(run_workdir, first)
    log_event(run_workdir, second)

    lines = _read_lines(run_workdir)
    assert len(lines) == 2
    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second


def test_append_to_preexisting_file_keeps_prior_content(tmp_path):
    """Even if the file already has content from another writer, we only append."""
    run_workdir = os.path.join(str(tmp_path), "run")
    os.makedirs(run_workdir, exist_ok=True)
    path = os.path.join(run_workdir, "events.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"step": "pre-existing"}) + "\n")

    log_event(run_workdir, {"step": "new"})

    lines = _read_lines(run_workdir)
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": "pre-existing"}
    assert json.loads(lines[1]) == {"step": "new"}
