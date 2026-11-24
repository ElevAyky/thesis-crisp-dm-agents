# src/crispfsm/dashboard/launcher.py
"""Run-Launcher des Dashboards: startet/abbricht FSM-Subprozesse und hält den
Status in der Registry aktuell.

Sicherheit: der Befehl wird IMMER als Liste gebaut und ohne ``shell=True``
gestartet — keine Shell, keine Injektion. Inputs (dataset/model/threshold)
werden vorher gegen Whitelists validiert."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

from crispfsm.pricing import PRICING

# Whitelist der erlaubten Datasets (analog zur Server-Validierung im Design-Doc).
ALLOWED_DATASETS = {"breast_cancer", "wine"}

# Repo-Root = drei Ebenen über dieser Datei (src/crispfsm/dashboard/launcher.py).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def launch_run(runs_dir, dataset, model, eval_f1_threshold, registry, popen_table):
    """Validiert Inputs, startet ``python -m crispfsm.run`` als Hintergrund-
    Subprozess und registriert den Run. Liefert die generierte ``run_id``.

    Raises ``ValueError`` bei ungültigen Inputs (die API mappt das auf HTTP 400).
    """
    # ---- Validierung (Whitelists, kein freier Text in den Befehl) -----------
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; allowed: {sorted(ALLOWED_DATASETS)}"
        )
    if model not in PRICING:
        raise ValueError(
            f"unknown model {model!r}; allowed: {sorted(PRICING.keys())}"
        )
    # None = keine Schwelle mitgeben -> run.py nimmt k x baseline_f1 aus dem Manifest.
    threshold = None
    if eval_f1_threshold is not None:
        try:
            threshold = float(eval_f1_threshold)
        except (TypeError, ValueError):
            raise ValueError(f"eval_f1_threshold must be a number, got {eval_f1_threshold!r}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"eval_f1_threshold must be in [0.0, 1.0], got {threshold}"
            )

    run_id = f"{dataset}-{uuid4().hex[:8]}"

    # Run-Ordner + Logdatei vorab anlegen (run.py legt work/ selbst an).
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")

    # Befehl als LISTE — niemals shell=True.
    cmd = [
        sys.executable, "-m", "crispfsm.run",
        "--run-id", run_id,
        "--dataset", dataset,
        "--model", model,
        "--runs-dir", runs_dir,
    ]
    if threshold is not None:
        cmd += ["--eval-f1-threshold", str(threshold)]

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=_REPO_ROOT,
    )

    registry.add(run_id, {
        "pid": proc.pid,
        "status": "running",
        "started_at": _now_iso(),
        "ended_at": None,
        "cmd": cmd,
        "dataset": dataset,
        "model": model,
    })
    popen_table[run_id] = proc
    return run_id


def _terminal_status_from_summary(runs_dir, run_id) -> str:
    """Liest ``summary.json`` des beendeten Runs und liefert
    ``success``/``failed``. Fehlt das summary, gilt der Run als ``failed``
    (Crash/Abbruch vor Abschluss)."""
    summary_path = os.path.join(runs_dir, run_id, "summary.json")
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "failed"
    return "success" if summary.get("success") else "failed"


def reconcile(registry, popen_table, runs_dir) -> None:
    """Pollt alle bekannten Subprozesse. Beendete werden terminalisiert
    (success/failed je nach ``summary.json``) und aus ``popen_table`` entfernt.
    Idempotent — gefahrlos vor jeder Liste/Detail aufrufbar."""
    for run_id in list(popen_table.keys()):
        proc = popen_table[run_id]
        if proc.poll() is None:
            continue  # läuft noch
        status = _terminal_status_from_summary(runs_dir, run_id)
        registry.mark_finished(run_id, status)
        popen_table.pop(run_id, None)


def cancel_run(run_id, registry, popen_table) -> str:
    """Bricht einen laufenden Run ab: Subprozess terminieren (falls noch am
    Leben), Status in der Registry auf ``cancelled`` setzen. Liefert den
    gesetzten Status (``cancelled``)."""
    proc = popen_table.get(run_id)
    if proc is not None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        popen_table.pop(run_id, None)
    registry.mark_finished(run_id, "cancelled")
    return "cancelled"
