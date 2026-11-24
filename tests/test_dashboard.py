# tests/test_dashboard.py
"""Tests für das Dashboard-Backend (Inventar, Status, Launch/Cancel-API).

Übersprungen, wenn die optionale ``dashboard``-Gruppe (fastapi) nicht
installiert ist — so bleibt die Core-Suite ohne diese Gruppe grün.

NIEMALS echter claude-/Subprozess-Aufruf: der POST-Pfad wird gemonkeypatcht,
sodass ``launch_run`` nur einen Fake-Run-Ordner + Registry-Eintrag anlegt
statt einen echten Popen zu starten."""
import json
import os

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from crispfsm.dashboard.app import create_app  # noqa: E402
from crispfsm.dashboard.registry import Registry  # noqa: E402
from crispfsm.dashboard.runs import (  # noqa: E402
    FILE_VIEW_MAX_CHARS,
    discover_runs,
    load_run_detail,
    read_run_file,
)


# --------------------------------------------------------------------- seed helpers
def _write_summary(run_dir, **fields):
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "run_id": os.path.basename(run_dir),
        "success": True,
        "final_f1": 0.95,
        "visited_phases": ["prep", "model", "eval"],
        "attempts": {"prep": 1, "model": 1, "eval": 1},
        "total_calls": 3,
        "usage_total": {"input_tokens": 30, "output_tokens": 6,
                        "cache_read_tokens": 0, "cache_creation_tokens": 0},
        "total_cost_usd": 0.0001,
    }
    summary.update(fields)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _write_events(run_dir, events):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _seed_three_runs(runs_dir):
    """Legt drei Runs an: success (summary+events), failed (summary success=false),
    running (nur Registry-Eintrag, kein summary)."""
    # 1) completed-success
    ok_dir = os.path.join(runs_dir, "ok-run")
    _write_summary(ok_dir, run_id="ok-run", success=True, final_f1=0.95,
                   dataset="breast_cancer", model="claude-haiku-4-5-20251001")
    _write_events(ok_dir, [
        {"phase": "prep", "attempt": 1, "gate_passed": True},
        {"phase": "model", "attempt": 1, "gate_passed": True},
        {"phase": "eval", "attempt": 1, "gate_passed": True},
    ])

    # 2) completed-failed
    fail_dir = os.path.join(runs_dir, "fail-run")
    _write_summary(fail_dir, run_id="fail-run", success=False, final_f1=0.10,
                   dataset="wine", model="claude-sonnet-4-6")
    _write_events(fail_dir, [
        {"phase": "eval", "attempt": 1, "gate_passed": False},
        {"phase": "model", "attempt": 2, "gate_passed": True},
        {"phase": "eval", "attempt": 2, "gate_passed": False},
    ])

    # 3) in-registry "running" (Ordner mit events, kein summary)
    run_dir = os.path.join(runs_dir, "live-run")
    _write_events(run_dir, [{"phase": "prep", "attempt": 1, "gate_passed": True}])

    reg = Registry(runs_dir).load()
    reg.add("live-run", {
        "pid": 999999, "status": "running",
        "started_at": "2026-06-05T10:00:00+00:00",
        "cmd": ["x"], "dataset": "breast_cancer",
        "model": "claude-haiku-4-5-20251001",
    })
    return reg


# --------------------------------------------------------------------- health
def test_health_ok(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# --------------------------------------------------------------------- discover/status
def test_discover_runs_three_statuses(tmp_path):
    runs_dir = str(tmp_path)
    _seed_three_runs(runs_dir)
    reg = Registry(runs_dir).load()

    rows = discover_runs(runs_dir, reg)
    by_id = {r["run_id"]: r for r in rows}

    assert by_id["ok-run"]["status"] == "success"
    assert by_id["ok-run"]["success"] is True
    assert by_id["ok-run"]["final_f1"] == 0.95
    assert by_id["ok-run"]["n_events"] == 3

    assert by_id["fail-run"]["status"] == "failed"
    assert by_id["fail-run"]["success"] is False

    assert by_id["live-run"]["status"] == "running"
    # running run: kein summary -> success None, dataset/model aus Registry
    assert by_id["live-run"]["success"] is None
    assert by_id["live-run"]["dataset"] == "breast_cancer"
    assert by_id["live-run"]["last_phase"] == "prep"

    # _dashboard-Ordner taucht NICHT als Run auf
    assert "_dashboard" not in by_id


def test_api_runs_three_statuses(tmp_path):
    runs_dir = str(tmp_path)
    _seed_three_runs(runs_dir)

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs")
    assert r.status_code == 200
    rows = r.json()["runs"]
    by_id = {row["run_id"]: row for row in rows}
    assert by_id["ok-run"]["status"] == "success"
    assert by_id["fail-run"]["status"] == "failed"
    assert by_id["live-run"]["status"] == "running"


def test_run_with_summary_but_no_registry_is_terminal(tmp_path):
    """Extern gestarteter Run (Ordner + summary, kein Registry-Eintrag) ->
    Status terminal aus summary, nicht 'unknown'."""
    runs_dir = str(tmp_path)
    _write_summary(os.path.join(runs_dir, "ext-run"), run_id="ext-run", success=True)
    reg = Registry(runs_dir).load()
    rows = {r["run_id"]: r for r in discover_runs(runs_dir, reg)}
    assert rows["ext-run"]["status"] == "success"


def test_run_without_summary_without_registry_is_unknown(tmp_path):
    runs_dir = str(tmp_path)
    _write_events(os.path.join(runs_dir, "orphan"), [{"phase": "prep"}])
    reg = Registry(runs_dir).load()
    rows = {r["run_id"]: r for r in discover_runs(runs_dir, reg)}
    assert rows["orphan"]["status"] == "unknown"


# --------------------------------------------------------------------- detail
def test_detail_returns_summary_events_artifacts(tmp_path):
    runs_dir = str(tmp_path)
    _seed_three_runs(runs_dir)
    # zusätzliches Artefakt anlegen
    art_dir = os.path.join(runs_dir, "ok-run", "eval")
    os.makedirs(art_dir, exist_ok=True)
    with open(os.path.join(art_dir, "eval_metrics.json"), "w") as f:
        json.dump({"f1": 0.95}, f)

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/ok-run")
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["success"] is True
    assert len(body["events"]) == 3
    assert any("eval_metrics.json" in a for a in body["artifacts"])
    assert "summary.json" in body["artifacts"]


def test_detail_404_unknown(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/api/runs/does-not-exist")
    assert r.status_code == 404


def test_detail_path_traversal_rejected(tmp_path):
    """Path-Traversal-IDs dürfen niemals aus runs_dir ausbrechen."""
    runs_dir = str(tmp_path)
    _seed_three_runs(runs_dir)

    # pure Funktion: '..' / Separatoren werden abgelehnt
    assert load_run_detail(runs_dir, "..") is None
    assert load_run_detail(runs_dir, ".") is None
    assert load_run_detail(runs_dir, "foo/bar") is None

    client = TestClient(create_app(runs_dir))
    # '..' als Segment -> 404 oder 400, jedenfalls kein 200/Escape
    for bad in ["..", "%2e%2e"]:
        r = client.get(f"/api/runs/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)


# --------------------------------------------------------------------- detail: run_log
def test_detail_includes_run_log(tmp_path):
    runs_dir = str(tmp_path)
    run_dir = os.path.join(runs_dir, "log-run")
    _write_events(run_dir, [{"phase": "prep", "attempt": 1, "gate_passed": True}])
    with open(os.path.join(run_dir, "run.log"), "w") as f:
        f.write("hello worker stdout\n")

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/log-run")
    assert r.status_code == 200
    assert "hello worker stdout" in r.json()["run_log"]


def test_detail_run_log_none_when_absent(tmp_path):
    runs_dir = str(tmp_path)
    run_dir = os.path.join(runs_dir, "nolog-run")
    _write_events(run_dir, [{"phase": "prep", "attempt": 1, "gate_passed": True}])

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/nolog-run")
    assert r.status_code == 200
    assert r.json()["run_log"] is None


def test_detail_passes_through_worker_text(tmp_path):
    """events.jsonl mit worker_text/worker_num_turns -> Felder erscheinen im Detail."""
    runs_dir = str(tmp_path)
    run_dir = os.path.join(runs_dir, "verbose-run")
    _write_events(run_dir, [
        {"phase": "prep", "attempt": 1, "gate_passed": True,
         "worker_text": "I wrote prepare.py and ran it.", "worker_num_turns": 7},
    ])

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/verbose-run")
    assert r.status_code == 200
    events = r.json()["events"]
    assert events[0]["worker_text"] == "I wrote prepare.py and ran it."
    assert events[0]["worker_num_turns"] == 7


# --------------------------------------------------------------------- file viewer
def test_file_viewer_happy_path(tmp_path):
    runs_dir = str(tmp_path)
    work_dir = os.path.join(runs_dir, "fv-run", "work", "prep")
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "prepare.py"), "w") as f:
        f.write("print('hello prep')\n")

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/fv-run/file", params={"path": "work/prep/prepare.py"})
    assert r.status_code == 200
    body = r.json()
    assert body["binary"] is False
    assert "hello prep" in body["content"]
    assert body["truncated"] is False
    assert body["path"] == "work/prep/prepare.py"


def test_file_viewer_traversal_rejected(tmp_path):
    runs_dir = str(tmp_path)
    run_dir = os.path.join(runs_dir, "fv-run")
    os.makedirs(run_dir, exist_ok=True)
    # eine Datei AUSSERHALB des Run-Ordners, die niemals erreichbar sein darf
    secret = os.path.join(runs_dir, "secret.txt")
    with open(secret, "w") as f:
        f.write("TOP SECRET")

    client = TestClient(create_app(runs_dir))
    for bad in ["../secret.txt", "../../etc/passwd", "/etc/passwd"]:
        r = client.get("/api/runs/fv-run/file", params={"path": bad})
        assert r.status_code == 400, (bad, r.status_code)
        assert "TOP SECRET" not in r.text

    # pure-Funktion: Traversal -> ValueError, absoluter Pfad -> ValueError
    with pytest.raises(ValueError):
        read_run_file(runs_dir, "fv-run", "../secret.txt")
    with pytest.raises(ValueError):
        read_run_file(runs_dir, "fv-run", "/etc/passwd")


def test_file_viewer_binary_flagged(tmp_path):
    runs_dir = str(tmp_path)
    model_dir = os.path.join(runs_dir, "fv-run", "model")
    os.makedirs(model_dir, exist_ok=True)
    # Bytes, die kein gültiges UTF-8 sind, und eine .pkl-Endung (binary set)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        f.write(b"\x80\x81\x82\xff\xfe garbage")

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/fv-run/file", params={"path": "model/model.pkl"})
    assert r.status_code == 200
    body = r.json()
    assert body["binary"] is True
    assert body["content"] is None
    assert body["size"] > 0


def test_file_viewer_truncation(tmp_path):
    runs_dir = str(tmp_path)
    run_dir = os.path.join(runs_dir, "fv-run")
    os.makedirs(run_dir, exist_ok=True)
    big = "x" * (FILE_VIEW_MAX_CHARS + 100)
    with open(os.path.join(run_dir, "big.txt"), "w") as f:
        f.write(big)

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/fv-run/file", params={"path": "big.txt"})
    assert r.status_code == 200
    body = r.json()
    assert body["truncated"] is True
    assert len(body["content"]) == FILE_VIEW_MAX_CHARS


def test_file_viewer_missing_file_404(tmp_path):
    runs_dir = str(tmp_path)
    run_dir = os.path.join(runs_dir, "fv-run")
    os.makedirs(run_dir, exist_ok=True)

    client = TestClient(create_app(runs_dir))
    r = client.get("/api/runs/fv-run/file", params={"path": "work/nope.py"})
    assert r.status_code == 404


# --------------------------------------------------------------------- POST validation
def test_post_validation_bad_dataset(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.post("/api/runs", json={
        "dataset": "iris", "model": "claude-haiku-4-5-20251001",
        "eval_f1_threshold": 0.5})
    assert r.status_code == 400


def test_post_validation_bad_model(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.post("/api/runs", json={
        "dataset": "wine", "model": "not-a-model", "eval_f1_threshold": 0.5})
    assert r.status_code == 400


def test_post_validation_threshold_out_of_range(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.post("/api/runs", json={
        "dataset": "wine", "model": "claude-haiku-4-5-20251001",
        "eval_f1_threshold": 1.5})
    assert r.status_code == 400


def test_post_valid_launch_monkeypatched(tmp_path, monkeypatch):
    """Valider POST mit gemonkeypatchtem Launcher: kein echter Subprozess,
    stattdessen Fake-Run-Ordner + Registry-Eintrag. Run taucht danach in
    /api/runs auf."""
    runs_dir = str(tmp_path)

    def fake_launch(runs_dir_, dataset, model, threshold, registry, popen_table):
        # Whitelist-Validierung beibehalten? Hier reicht ein Fake; die echte
        # Validierung ist über die bad-* Tests abgedeckt.
        run_id = f"{dataset}-fake1234"
        run_dir = os.path.join(runs_dir_, run_id)
        os.makedirs(run_dir, exist_ok=True)
        registry.add(run_id, {
            "pid": 4242, "status": "running", "cmd": ["fake"],
            "dataset": dataset, "model": model,
        })
        return run_id

    # Launcher im app-Modul patchen (app ruft _launcher.launch_run).
    monkeypatch.setattr("crispfsm.dashboard.launcher.launch_run", fake_launch)

    client = TestClient(create_app(runs_dir))
    r = client.post("/api/runs", json={
        "dataset": "wine", "model": "claude-haiku-4-5-20251001",
        "eval_f1_threshold": 0.5})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert run_id == "wine-fake1234"

    # Run erscheint in der Liste (Status running, da Registry-Eintrag + kein summary).
    r2 = client.get("/api/runs")
    ids = {row["run_id"]: row for row in r2.json()["runs"]}
    assert run_id in ids
    assert ids[run_id]["status"] == "running"


# --------------------------------------------------------------------- cancel
def test_cancel_marks_cancelled(tmp_path):
    runs_dir = str(tmp_path)
    reg = _seed_three_runs(runs_dir)  # live-run is "running" in registry

    client = TestClient(create_app(runs_dir))
    r = client.post("/api/runs/live-run/cancel")
    assert r.status_code == 200
    assert r.json() == {"status": "cancelled"}

    # Registry frisch laden: Status persistiert als cancelled
    reg2 = Registry(runs_dir).load()
    assert reg2.get("live-run")["status"] == "cancelled"


def test_cancel_404_unknown(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.post("/api/runs/nope/cancel")
    assert r.status_code == 404


# --------------------------------------------------------------------- registry persistence
def test_registry_roundtrip(tmp_path):
    runs_dir = str(tmp_path)
    reg = Registry(runs_dir).load()
    reg.add("r1", {"pid": 1, "status": "running", "dataset": "wine",
                   "model": "claude-haiku-4-5-20251001", "cmd": ["x"]})
    reg.mark_finished("r1", "success")

    reg2 = Registry(runs_dir).load()
    entry = reg2.get("r1")
    assert entry is not None
    assert entry["status"] == "success"
    assert entry["dataset"] == "wine"
    assert entry["ended_at"] is not None
    assert reg2.all().keys() == {"r1"}


# --------------------------------------------------------------------- root serves the SPA
def test_root_serves_index_html(tmp_path):
    client = TestClient(create_app(str(tmp_path)))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    # serves the real control-center SPA (stable markers, not the old placeholder)
    assert "schaltzentrale" in r.text.lower()
    assert 'data-testid="runs-table"' in r.text
