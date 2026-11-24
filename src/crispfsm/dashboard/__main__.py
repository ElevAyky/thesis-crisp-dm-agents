# src/crispfsm/dashboard/__main__.py
"""CLI-Entrypoint des Dashboards.

    uv run --group dashboard python -m crispfsm.dashboard --runs-dir runs --port 8000
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from crispfsm.dashboard.app import create_app


def main():
    ap = argparse.ArgumentParser(description="crispfsm dashboard server")
    ap.add_argument(
        "--runs-dir",
        default=os.environ.get("CRISPFSM_RUNS_DIR", "runs"),
        help="Verzeichnis mit den Run-Ordnern (default: $CRISPFSM_RUNS_DIR oder 'runs')",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = create_app(args.runs_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
