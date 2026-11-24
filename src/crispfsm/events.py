# src/crispfsm/events.py
from __future__ import annotations
import os, json


def log_event(run_workdir: str, event: dict) -> None:
    """Hängt eine JSONL-Zeile an runs/<run_id>/events.jsonl an."""
    path = os.path.join(run_workdir, "events.jsonl")
    os.makedirs(run_workdir, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")
