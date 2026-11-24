# src/crispfsm/dashboard/app.py
"""Dünne FastAPI-Schicht über der unit-getesteten Logik in
``runs.py`` / ``registry.py`` / ``launcher.py``.

Die App hält pro Instanz eine in-process ``popen_table`` (run_id -> Popen) und
eine ``Registry`` (persistiert nach ``<runs_dir>/_dashboard/registry.json``).
Vor jeder Liste/Detail wird ``reconcile()`` aufgerufen, damit Status frisch
sind. Logik bleibt bewusst in den anderen Modulen — hier nur HTTP-Glue."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from crispfsm.dashboard.registry import Registry
from crispfsm.dashboard import launcher as _launcher
from crispfsm.dashboard.runs import discover_runs, load_run_detail, read_run_file

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_INDEX_HTML = os.path.join(_STATIC_DIR, "index.html")


class NewRunRequest(BaseModel):
    """Body für ``POST /api/runs``."""
    dataset: str
    model: str
    eval_f1_threshold: float | None = None   # None = Manifest-Baseline (k = 1)


def create_app(runs_dir: str) -> FastAPI:
    """Baut eine FastAPI-App, die Runs unter ``runs_dir`` inventarisiert,
    startet und abbricht. Jede App-Instanz hat ihre eigene Registry +
    popen_table (praktisch für Tests mit ``tmp_path``)."""
    runs_dir = str(runs_dir)
    app = FastAPI(title="crispfsm dashboard")

    registry = Registry(runs_dir).load()
    popen_table: dict[str, object] = {}

    # Beim Start tote PIDs aus einer früheren Dashboard-Session rekonzilieren:
    # die popen_table ist leer, daher tut reconcile() hier nichts — aber als
    # running markierte Runs ohne lebenden Prozess sind aus dieser Sicht
    # extern. Der Ordner-Scan/_derive_status behandelt sie korrekt.

    # für Tests/Introspektion am App-Objekt erreichbar machen
    app.state.registry = registry
    app.state.popen_table = popen_table
    app.state.runs_dir = runs_dir

    @app.get("/")
    def index():
        return FileResponse(_INDEX_HTML)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/runs")
    def list_runs():
        _launcher.reconcile(registry, popen_table, runs_dir)
        return {"runs": discover_runs(runs_dir, registry)}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        _launcher.reconcile(registry, popen_table, runs_dir)
        detail = load_run_detail(runs_dir, run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="run not found")
        # Registry-Status anhängen (Detail-Logik selbst ist registry-frei).
        reg = registry.get(run_id)
        detail["registry_status"] = reg.get("status") if reg else None
        return detail

    @app.get("/api/runs/{run_id}/file")
    def run_file(run_id: str, path: str):
        # Read-only Artefakt-Viewer (Debug-Mode), sandboxed auf den Run-Ordner.
        try:
            return read_run_file(runs_dir, run_id, path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="file not found")

    @app.post("/api/runs")
    def create_run(body: NewRunRequest):
        try:
            run_id = _launcher.launch_run(
                runs_dir, body.dataset, body.model, body.eval_f1_threshold,
                registry, popen_table,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"run_id": run_id}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        # Run muss existieren (Ordner ODER Registry-Eintrag), sonst 404.
        run_dir = os.path.join(runs_dir, run_id)
        known = registry.get(run_id) is not None or os.path.isdir(run_dir)
        if not known:
            raise HTTPException(status_code=404, detail="run not found")
        _launcher.cancel_run(run_id, registry, popen_table)
        return {"status": "cancelled"}

    return app
