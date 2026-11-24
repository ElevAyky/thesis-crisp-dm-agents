# src/crispfsm/dashboard/runs.py
"""Reine, FastAPI-freie Lese-Logik des Dashboards (unit-testbar ohne HTTP).

Quellen pro Run (``<runs_dir>/<run_id>/``):
- ``summary.json`` — existiert nur nach Abschluss.
- ``events.jsonl`` — wächst während des Laufs (eine Zeile pro Worker-Call).

Status wird aus Registry + Ordner-Inhalt nach der Tabelle aus dem Design-Doc
(Abschnitt 5) abgeleitet."""
from __future__ import annotations

import json
import os

# Name des Sidecar-Ordners; wird beim Scan übersprungen (keine Run-Daten).
DASHBOARD_DIR = "_dashboard"

# Caps gegen riesige Responses (Debug/Verbose-Mode):
# - run.log nur die letzten N Zeichen (Tail), damit lange Läufe die Detail-Response
#   nicht aufblähen.
# - Datei-Viewer cappt Text-Inhalt auf N Zeichen.
RUN_LOG_TAIL_CHARS = 40_000
FILE_VIEW_MAX_CHARS = 200_000

# Erweiterungen, die immer als binär gelten (auch wenn sie zufällig dekodierbar wären).
BINARY_EXTENSIONS = {".pkl", ".parquet", ".png", ".jpg", ".jpeg", ".bin"}


def _read_run_log_tail(path: str, max_chars: int = RUN_LOG_TAIL_CHARS) -> str | None:
    """Liest ``run.log`` und gibt die LETZTEN ``max_chars`` Zeichen zurück;
    ``None``, wenn die Datei fehlt."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        return None
    if len(content) > max_chars:
        return content[-max_chars:]
    return content


def _read_json(path: str):
    """Liest eine JSON-Datei; gibt ``None`` zurück, wenn sie fehlt/kaputt ist."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_events(path: str) -> list[dict]:
    """Parst ``events.jsonl`` zeilenweise. Kaputte Zeilen werden übersprungen,
    fehlende Datei ergibt eine leere Liste."""
    events: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return events


def _safe_segment(run_id: str) -> bool:
    """True, wenn ``run_id`` ein einzelnes, sicheres Pfadsegment ist (kein
    Traversal, kein Separator, kein '.'/'..')."""
    if not run_id or run_id in (".", ".."):
        return False
    if "/" in run_id or "\\" in run_id or "\x00" in run_id:
        return False
    # os.path.basename darf den Wert nicht verändern (sonst steckt ein sep drin)
    return os.path.basename(run_id) == run_id


def _derive_status(run_id: str, summary, n_events: int, registry) -> str:
    """Status nach der Design-Doc-Tabelle (Abschnitt 5).

    Priorität: lebender Subprozess (running) > terminales summary > Registry-Tod
    > extern (unknown). ``reconcile()`` aktualisiert die Registry vorher, daher
    spiegelt ein Registry-Status ``running`` hier einen wirklich lebenden Prozess
    wider."""
    reg = registry.get(run_id) if registry is not None else None

    if reg is not None and reg.get("status") == "running":
        return "running"

    if isinstance(summary, dict) and "success" in summary:
        return "success" if summary.get("success") else "failed"

    if reg is not None:
        # Registry-Eintrag, aber kein erfolgreiches summary -> Status aus Registry
        # (cancelled/failed) bzw. failed als Fallback (Crash/Abbruch).
        st = reg.get("status")
        return st if st in ("success", "failed", "cancelled", "unknown") else "failed"

    # Ordner ohne Registry-Eintrag und ohne (terminales) summary -> extern.
    return "unknown"


def discover_runs(runs_dir: str, registry) -> list[dict]:
    """Scannt die Run-Ordner unter ``runs_dir`` und mergt sie mit der Registry.

    Liefert eine Zeile pro Run mit:
    run_id, dataset, model, status, success, final_f1, total_calls,
    total_cost_usd, started_at, n_events, last_phase, last_gate_passed.
    """
    rows: list[dict] = []
    if not os.path.isdir(runs_dir):
        return rows

    for name in os.listdir(runs_dir):
        if name == DASHBOARD_DIR:
            continue
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue

        summary = _read_json(os.path.join(run_dir, "summary.json"))
        events = _read_events(os.path.join(run_dir, "events.jsonl"))
        n_events = len(events)
        last = events[-1] if events else None
        reg = registry.get(name) if registry is not None else None

        status = _derive_status(name, summary, n_events, registry)

        # dataset/model bevorzugt aus summary, sonst aus Registry, sonst None.
        dataset = None
        model = None
        if isinstance(summary, dict):
            dataset = summary.get("dataset")
            model = summary.get("model")
        if dataset is None and reg is not None:
            dataset = reg.get("dataset")
        if model is None and reg is not None:
            model = reg.get("model")

        # started_at: Registry zuerst (Launch-Zeit), sonst Ordner-mtime als ISO.
        started_at = reg.get("started_at") if reg is not None else None
        try:
            mtime = os.path.getmtime(run_dir)
        except OSError:
            mtime = 0.0

        rows.append({
            "run_id": name,
            "dataset": dataset,
            "model": model,
            "status": status,
            "success": (summary.get("success") if isinstance(summary, dict) else None),
            "final_f1": (summary.get("final_f1") if isinstance(summary, dict) else None),
            "total_calls": (summary.get("total_calls") if isinstance(summary, dict) else None),
            "total_cost_usd": (summary.get("total_cost_usd") if isinstance(summary, dict) else None),
            "started_at": started_at,
            "n_events": n_events,
            "last_phase": (last.get("phase") if isinstance(last, dict) else None),
            "last_gate_passed": (last.get("gate_passed") if isinstance(last, dict) else None),
            # interner Sortier-Schlüssel (nicht Teil des API-Vertrags, aber harmlos)
            "_mtime": mtime,
        })

    # Neueste zuerst: nach started_at (ISO sortierbar) bzw. mtime als Fallback.
    rows.sort(key=lambda r: (r["started_at"] or "", r["_mtime"]), reverse=True)
    return rows


def load_run_detail(runs_dir: str, run_id: str) -> dict | None:
    """Detail eines Runs: summary (oder None), Events-Liste, Artefakt-Dateiliste
    (relative Pfade unter dem Run-Ordner) und der Registry-Status.

    Pfad-Traversal-geschützt: ``run_id`` muss ein einzelnes sicheres Segment
    sein und der aufgelöste Ordner unter ``runs_dir`` liegen. Gibt ``None``
    zurück, wenn der Run-Ordner nicht existiert (oder die ID unsicher ist) —
    die API macht daraus ein 404."""
    if not _safe_segment(run_id):
        return None

    runs_dir_abs = os.path.abspath(runs_dir)
    run_dir = os.path.abspath(os.path.join(runs_dir_abs, run_id))

    # Defense-in-depth: der aufgelöste Pfad muss echtes Kind von runs_dir sein.
    if os.path.commonpath([runs_dir_abs, run_dir]) != runs_dir_abs:
        return None
    if run_dir == runs_dir_abs or not os.path.isdir(run_dir):
        return None

    summary = _read_json(os.path.join(run_dir, "summary.json"))
    events = _read_events(os.path.join(run_dir, "events.jsonl"))
    run_log = _read_run_log_tail(os.path.join(run_dir, "run.log"))

    # Artefakt-Liste: alle Dateien (rekursiv) unter dem Run-Ordner, relativ.
    artifacts: list[str] = []
    for root, _dirs, files in os.walk(run_dir):
        for fname in files:
            full = os.path.join(root, fname)
            artifacts.append(os.path.relpath(full, run_dir))
    artifacts.sort()

    return {
        "run_id": run_id,
        "summary": summary,
        "events": events,
        "artifacts": artifacts,
        "run_log": run_log,
    }


def read_run_file(runs_dir: str, run_id: str, relpath: str) -> dict:
    """Liest eine Artefakt-Datei eines Runs für den UI-Viewer (read-only, sandboxed).

    Sicherheit: ``run_id`` muss ein sicheres Segment sein, der Run-Ordner muss
    existieren, und ``relpath`` wird strikt UNTER den Run-Ordner aufgelöst —
    jeder Versuch, daraus auszubrechen (``..``, absolute Pfade, Symlink-Eskape),
    wird über ``realpath`` + ``commonpath`` abgewiesen.

    Raises:
        ValueError: bei unsicherer ``run_id``, fehlendem Run-Ordner oder
            Pfad-Traversal (die API mappt das auf HTTP 400).
        FileNotFoundError: wenn die Ziel-Datei nicht existiert (-> HTTP 404).

    Returns ein dict:
        - binär: ``{"path", "binary": True, "size", "content": None}``
        - text:  ``{"path", "binary": False, "size", "content", "truncated"}``
    """
    if not _safe_segment(run_id):
        raise ValueError(f"unsafe run_id: {run_id!r}")

    runs_dir_abs = os.path.realpath(runs_dir)
    run_dir = os.path.realpath(os.path.join(runs_dir_abs, run_id))
    if os.path.commonpath([runs_dir_abs, run_dir]) != runs_dir_abs:
        raise ValueError(f"run_id escapes runs_dir: {run_id!r}")
    if run_dir == runs_dir_abs or not os.path.isdir(run_dir):
        raise ValueError(f"run dir not found: {run_id!r}")

    # relpath darf nicht absolut sein und keine '..'-Segmente enthalten.
    if not relpath or os.path.isabs(relpath):
        raise ValueError(f"invalid path: {relpath!r}")
    norm = os.path.normpath(relpath)
    if norm == ".." or norm.startswith(".." + os.sep) or os.path.isabs(norm):
        raise ValueError(f"path traversal rejected: {relpath!r}")

    # Auflösen UNTER dem Run-Ordner; realpath fängt auch Symlink-Eskapes ab.
    target = os.path.realpath(os.path.join(run_dir, norm))
    if os.path.commonpath([run_dir, target]) != run_dir or target == run_dir:
        raise ValueError(f"path traversal rejected: {relpath!r}")

    if not os.path.isfile(target):
        raise FileNotFoundError(relpath)

    size = os.path.getsize(target)
    ext = os.path.splitext(target)[1].lower()

    # Bekannte Binär-Endungen direkt als binär melden (kein Dekodierversuch).
    if ext in BINARY_EXTENSIONS:
        return {"path": relpath, "binary": True, "size": size, "content": None}

    # Sonst: UTF-8-Dekodierung versuchen. Schlägt sie fehl -> binär.
    with open(target, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"path": relpath, "binary": True, "size": size, "content": None}

    truncated = len(text) > FILE_VIEW_MAX_CHARS
    if truncated:
        text = text[:FILE_VIEW_MAX_CHARS]
    return {
        "path": relpath,
        "binary": False,
        "size": size,
        "content": text,
        "truncated": truncated,
    }
