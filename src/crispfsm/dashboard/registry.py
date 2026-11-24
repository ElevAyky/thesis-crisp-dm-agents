# src/crispfsm/dashboard/registry.py
"""Sidecar-Registry des Dashboards: hält pro Run die Orchestrierungs-Metadaten
(PID, Status, Start/Ende, Befehl, Dataset, Modell), persistiert nach
``<runs_dir>/_dashboard/registry.json``.

Das ist *normaler Laufzeit-Code* (kein Workflow-Skript), daher sind hier
``datetime.now()`` und ISO-Zeitstempel ausdrücklich erlaubt. Keine externen
Abhängigkeiten — nur stdlib, damit die Core-Suite ohne die ``dashboard``-Gruppe
weiterhin importierbar bleibt, falls jemand das Modul direkt anfasst."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def _now_iso() -> str:
    """Aktuelle Zeit als ISO-8601-String (UTC, sekundengenau, ohne Mikros)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Registry:
    """In-Memory-Map ``{run_id: meta}`` mit Persistenz nach JSON.

    ``meta`` ist ein dict mit den Schlüsseln
    ``{pid, status, started_at, ended_at, cmd, dataset, model}``.
    """

    def __init__(self, runs_dir: str):
        self.runs_dir = runs_dir
        self.path = os.path.join(runs_dir, "_dashboard", "registry.json")
        self._runs: dict[str, dict] = {}

    # ------------------------------------------------------------------ I/O
    def load(self) -> "Registry":
        """Liest die Registry von der Platte (falls vorhanden). Fehlt die Datei
        oder ist sie kaputt, startet man mit einer leeren Map."""
        try:
            with open(self.path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._runs = {str(k): dict(v) for k, v in data.items()}
            else:
                self._runs = {}
        except (FileNotFoundError, json.JSONDecodeError):
            self._runs = {}
        return self

    def save(self) -> None:
        """Schreibt die Registry atomar (tmp + rename) nach ``self.path``."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._runs, f, indent=2)
        os.replace(tmp, self.path)

    # -------------------------------------------------------------- Mutators
    def add(self, run_id: str, meta: dict) -> dict:
        """Registriert einen Run. Fehlende Standardfelder werden ergänzt;
        ``started_at`` wird gesetzt, falls nicht mitgegeben. Persistiert."""
        entry = {
            "pid": None,
            "status": "running",
            "started_at": _now_iso(),
            "ended_at": None,
            "cmd": None,
            "dataset": None,
            "model": None,
        }
        entry.update(meta)
        self._runs[run_id] = entry
        self.save()
        return entry

    def mark_finished(self, run_id: str, status: str) -> None:
        """Setzt Status (z.B. success/failed/cancelled) und ``ended_at``.
        No-op, wenn der Run unbekannt ist. Persistiert."""
        entry = self._runs.get(run_id)
        if entry is None:
            return
        entry["status"] = status
        entry["ended_at"] = _now_iso()
        self.save()

    # --------------------------------------------------------------- Readers
    def get(self, run_id: str) -> dict | None:
        """Liefert das Meta-dict für ``run_id`` (oder ``None``)."""
        return self._runs.get(run_id)

    def all(self) -> dict[str, dict]:
        """Liefert eine flache Kopie der gesamten Map ``{run_id: meta}``."""
        return {k: dict(v) for k, v in self._runs.items()}
