# tests/test_phases.py
import os, json, pickle, uuid
import pandas as pd
from sklearn.dummy import DummyClassifier
from crispfsm.config import RunConfig
from crispfsm.state import new_state
from crispfsm.phases import make_phase_node
from crispfsm.backend import WorkerResult, TokenUsage


class FakeBackend:
    """Schreibt deterministisch ein gültiges prepared.parquet und meldet Usage."""
    name = "fake"

    def __init__(self, model="m"):
        self.model = model

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        os.makedirs(os.path.join(cwd, "prep"), exist_ok=True)
        df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
        df.to_parquet(os.path.join(cwd, "prep", "prepared.parquet"), index=False)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=5.0,
                            session_id=session_id, raw={})


def test_prep_node_passes_and_updates_state(tmp_path):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)
    state = new_state(cfg)
    node = make_phase_node("prep", FakeBackend(), cfg)

    update = node(state)
    assert update["gate_passed"] is True
    assert "prep" in update["visited_phases"]
    assert update["attempts"]["prep"] == 1
    assert update["total_calls"] == 1
    assert update["usage_total"]["input_tokens"] == 10
    # Event geloggt
    events = open(os.path.join(tmp_path, "events.jsonl")).read().strip().splitlines()
    assert len(events) == 1
    ev = json.loads(events[0])
    assert ev["phase"] == "prep"
    # Debug-Mode: die LLM-Ausgabe wird mitgeloggt (FakeBackend liefert text="ok").
    assert ev["worker_text"] == "ok"
    # num_turns ist im Event vorhanden, aber None für Fakes (raw={} -> .get None).
    assert "worker_num_turns" in ev
    assert ev["worker_num_turns"] is None


# ---------------------------------------------------------------------------
# Shared fake backends for the additional coverage below
# ---------------------------------------------------------------------------

def _write_prep(cwd):
    os.makedirs(os.path.join(cwd, "prep"), exist_ok=True)
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    df.to_parquet(os.path.join(cwd, "prep", "prepared.parquet"), index=False)


class RecordingFakeBackend:
    """Captures the prompt/resume/session_id it was called with and still
    writes a valid prep artifact so the gate passes."""
    name = "recording"

    def __init__(self):
        self.calls = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self.calls.append({
            "prompt": prompt, "resume": resume, "session_id": session_id,
            "cwd": cwd, "max_turns": max_turns, "budget_usd": budget_usd,
            "allowed_tools": allowed_tools,
        })
        _write_prep(cwd)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=5.0,
                            session_id=session_id, raw={})


class ModelFakeBackend:
    """Writes a real pickled, fitted sklearn model + metrics.json so check_model passes."""
    name = "model_fake"

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        os.makedirs(os.path.join(cwd, "model"), exist_ok=True)
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1])
        with open(os.path.join(cwd, "model", "model.pkl"), "wb") as f:
            pickle.dump(clf, f)
        with open(os.path.join(cwd, "model", "metrics.json"), "w") as f:
            json.dump({"metric": "f1", "val_score": 0.9, "dummy_score": 0.6}, f)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=5.0,
                            session_id=session_id, raw={})


class EvalFakeBackend:
    """Schreibt eval/predictions.parquet. perfect=True kopiert die Wahrheit aus
    ../holdout (Test-Fake-Cheat), perfect=False invertiert die 0/1-Labels."""
    name = "eval_fake"

    def __init__(self, perfect=True):
        self.perfect = perfect

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        os.makedirs(os.path.join(cwd, "eval"), exist_ok=True)
        truth = pd.read_parquet(
            os.path.join(cwd, "..", "holdout", "test.parquet"))["target"]
        preds = truth if self.perfect else truth.map({0: 1, 1: 0})
        pd.DataFrame({"prediction": preds}).to_parquet(
            os.path.join(cwd, "eval", "predictions.parquet"), index=False)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=5.0,
                            session_id=session_id, raw={})


class SilentFakeBackend:
    """Writes NOTHING — every gate must fail."""
    name = "silent"

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        return WorkerResult(text="", usage=TokenUsage(10, 2), latency_ms=5.0,
                            session_id=session_id, raw={})


# ---------------------------------------------------------------------------
# RESUME path
# ---------------------------------------------------------------------------

def test_prep_resume_uses_resume_prompt_and_stable_session(tmp_path):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)

    stored_sid = str(uuid.uuid4())
    feedback = "prep/prepared.parquet fehlt — Phase hat kein Artefakt erzeugt."
    state = new_state(cfg)
    state["attempts"]["prep"] = 1                 # > 0 → resume path
    state["current_phase"] = "prep"               # eigenes Gate gerade gescheitert (kind "own")
    state["gate_passed"] = False
    state["gate_feedback"] = feedback
    state["session_ids"]["prep"] = stored_sid     # stable reuse

    backend = RecordingFakeBackend()
    node = make_phase_node("prep", backend, cfg)
    update = node(state)

    assert len(backend.calls) == 1
    call = backend.calls[0]
    # resume=True passed to backend
    assert call["resume"] is True
    # resume_prompt was used: contains the Quality-Gate framing + the feedback text
    assert "Quality Gate" in call["prompt"]
    assert feedback in call["prompt"]
    # stable session reuse: backend got the stored id, and it round-trips into state
    assert call["session_id"] == stored_sid
    assert update["session_ids"]["prep"] == stored_sid
    # gate still passes (artifact written) and this was a second attempt
    assert update["gate_passed"] is True
    assert update["attempts"]["prep"] == 2


# ---------------------------------------------------------------------------
# MODEL node
# ---------------------------------------------------------------------------

def test_model_node_passes_with_real_pickled_model(tmp_path):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)
    state = new_state(cfg)
    node = make_phase_node("model", ModelFakeBackend(), cfg)

    update = node(state)
    assert update["gate_passed"] is True
    assert "model" in update["visited_phases"]
    assert update["attempts"]["model"] == 1
    # model gate is not eval → no success flag set
    assert "success" not in update
    # metrics surfaced from the gate
    assert update["gate_metrics"]["val_score"] == 0.9


# ---------------------------------------------------------------------------
# EVAL node
# ---------------------------------------------------------------------------

def test_eval_node_copies_test_features_and_sets_success(tmp_path):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    work = os.path.join(tmp_path, "work")
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "holdout"), exist_ok=True)
    test_df = pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 1]})
    test_df.to_parquet(os.path.join(tmp_path, "holdout", "test.parquet"), index=False)
    test_df.drop(columns=["target"]).to_parquet(
        os.path.join(tmp_path, "holdout", "test_features.parquet"), index=False)

    state = new_state(cfg)
    node = make_phase_node("eval", EvalFakeBackend(), cfg)
    update = node(state)

    # Node kopierte NUR die Features ins Work-Verzeichnis — nie die Labels
    copied = os.path.join(work, "data", "test_features.parquet")
    assert os.path.exists(copied)
    assert "target" not in pd.read_parquet(copied).columns
    assert not os.path.exists(os.path.join(work, "data", "test.parquet"))

    assert update["gate_passed"] is True
    assert update["success"] is True
    assert update["final_f1"] == 1.0


def test_eval_node_threshold_fail_reports_metrics_without_form_error(tmp_path):
    """Schwellen-Fail (invertierte Labels -> F1 0.0): das Gate liefert Metriken,
    aber KEIN form_error-Flag — der Router muss das als Modellqualitätsproblem
    (Rücksprung zu model) einordnen, nicht als Eval-Kontraktfehler."""
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "holdout"), exist_ok=True)
    test_df = pd.DataFrame({"f1": [1.0, 2.0], "target": [0, 1]})
    test_df.to_parquet(os.path.join(tmp_path, "holdout", "test.parquet"), index=False)
    test_df.drop(columns=["target"]).to_parquet(
        os.path.join(tmp_path, "holdout", "test_features.parquet"), index=False)

    state = new_state(cfg)
    node = make_phase_node("eval", EvalFakeBackend(perfect=False), cfg)
    update = node(state)

    assert update["gate_passed"] is False
    assert update["gate_metrics"]["f1"] == 0.0
    assert "form_error" not in update["gate_metrics"]
    assert "success" not in update


# ---------------------------------------------------------------------------
# GATE-FAIL path
# ---------------------------------------------------------------------------

def test_prep_node_gate_fail_increments_and_no_success(tmp_path):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)
    state = new_state(cfg)
    node = make_phase_node("prep", SilentFakeBackend(), cfg)

    update = node(state)
    assert update["gate_passed"] is False
    assert update["gate_feedback"]                       # non-empty failure reason
    assert "success" not in update                       # never set on failure
    assert update["attempts"]["prep"] == 1               # incremented from 0
    assert update["total_calls"] == 1                    # call still counted
    # the phase is still recorded as visited even on failure
    assert "prep" in update["visited_phases"]


# ---------------------------------------------------------------------------
# USAGE accumulation across two calls (manual reducer merge)
# ---------------------------------------------------------------------------

def test_usage_accumulates_across_two_calls(tmp_path):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)
    node = make_phase_node("prep", FakeBackend(), cfg)

    state = new_state(cfg)
    update1 = node(state)
    # mimic LangGraph reducer: merge partial update back into state
    state = {**state, **update1}
    update2 = node(state)
    state = {**state, **update2}

    # usage doubled (10 input tokens per call)
    assert state["usage_total"]["input_tokens"] == 20
    assert state["usage_total"]["output_tokens"] == 4
    assert state["usage_per_phase"]["prep"]["input_tokens"] == 20
    # two worker calls counted
    assert state["total_calls"] == 2
    # attempts incremented twice
    assert state["attempts"]["prep"] == 2
    # visited deduped — phase recorded once, not twice
    assert state["visited_phases"] == ["prep"]


# ---------------------------------------------------------------------------
# BU node
# ---------------------------------------------------------------------------

class BuSpecBackend:
    """Schreibt eine gültige bzw. abstain-Spec, je nach Konstruktor."""
    name = "bu_fake"
    def __init__(self, spec):
        self.model = "m"; self.spec = spec
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        os.makedirs(f"{cwd}/bu", exist_ok=True)
        json.dump(self.spec, open(f"{cwd}/bu/task_spec.json", "w"))
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})

def _bu_state(tmp_path, target="col_2"):
    cfg = RunConfig(run_id="r", workdir=str(tmp_path), target=target,
                    task_prompt="Bestimme die Klasse.")
    work = os.path.join(str(tmp_path), "work", "data")
    os.makedirs(work, exist_ok=True)
    pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0],
                  target: ["a", "b"]}).to_parquet(os.path.join(work, "train.parquet"), index=False)
    return cfg, new_state(cfg)

def test_bu_node_pass_sets_spec_and_correctness(tmp_path):
    cfg, state = _bu_state(tmp_path)
    backend = BuSpecBackend({"task_type": "classification", "target_column": "col_2", "metric": "f1"})
    update = make_phase_node("bu", backend, cfg)(state)
    assert update["gate_passed"] is True
    assert update["task_spec"]["target_column"] == "col_2"
    assert update["bu_correct"] is True

def test_bu_node_wrong_target_is_measured_not_gated(tmp_path):
    # f1 existiert im Schema -> Gate PASST, aber bu_correct ist False (Messung, kein Zwang)
    cfg, state = _bu_state(tmp_path)
    backend = BuSpecBackend({"task_type": "classification", "target_column": "f1", "metric": "accuracy"})
    update = make_phase_node("bu", backend, cfg)(state)
    assert update["gate_passed"] is True
    assert update["bu_correct"] is False

def test_bu_node_abstain(tmp_path):
    cfg, state = _bu_state(tmp_path)
    backend = BuSpecBackend({"cannot_determine": True, "reason": "kein Ziel", "missing_info": []})
    update = make_phase_node("bu", backend, cfg)(state)
    assert update["gate_passed"] is False
    assert update["abstained"] is True


# ---------------------------------------------------------------------------
# DU node
# ---------------------------------------------------------------------------

def _write_faithful_du_report(cwd):
    """Rechnet den Report ehrlich aus data/train.parquet — Gate muss passen."""
    df = pd.read_parquet(os.path.join(cwd, "data", "train.parquet"))
    target = df.columns[-1]
    os.makedirs(os.path.join(cwd, "du"), exist_ok=True)
    kinds = {c: ("bool" if pd.api.types.is_bool_dtype(df[c]) else
                 "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical")
             for c in df.columns}
    json.dump({
        "n_rows": len(df),
        "columns": {c: {"dtype": kinds[c], "n_missing": int(df[c].isna().sum())}
                    for c in df.columns},
        "class_balance": {str(k): int(v) for k, v in df[target].value_counts().items()},
        "quality_issues": [],
        "excluded_columns": [],
    }, open(os.path.join(cwd, "du", "data_report.json"), "w"))


class DuReportBackend:
    name = "du_fake"
    def __init__(self):
        self.model = "m"
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        _write_faithful_du_report(cwd)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})


def test_du_node_pass_sets_du_report(tmp_path):
    cfg, state = _bu_state(tmp_path)               # legt work/data/train.parquet an
    update = make_phase_node("du", DuReportBackend(), cfg)(state)
    assert update["gate_passed"] is True
    assert update["du_report"]["n_rows"] == 2
    assert update["attempts"]["du"] == 1
    assert "du" in update["visited_phases"]


def test_du_prompt_contains_critical_columns_from_bu_spec(tmp_path):
    cfg, state = _bu_state(tmp_path)
    state["task_spec"] = {"task_type": "classification", "target_column": "col_2",
                          "metric": "f1", "critical_columns": ["f2"]}
    backend = RecordingFakeBackend()
    make_phase_node("du", backend, cfg)(state)
    prompt = backend.calls[0]["prompt"]
    assert "Data Understanding" in prompt
    assert "f2" in prompt


def test_du_node_gate_fail_on_missing_report(tmp_path):
    cfg, state = _bu_state(tmp_path)
    update = make_phase_node("du", SilentFakeBackend(), cfg)(state)
    assert update["gate_passed"] is False
    assert "data_report.json" in update["gate_feedback"]
    assert "du_report" not in update


def test_prep_prompt_receives_du_findings_from_state(tmp_path):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)
    state = new_state(cfg)
    state["du_report"] = {"quality_issues": ["f1 hat Ausreißer > 3 IQR"],
                          "excluded_columns": [{"column": "f9", "reason": "Leakage-Verdacht"}]}
    backend = RecordingFakeBackend()
    make_phase_node("prep", backend, cfg)(state)
    prompt = backend.calls[0]["prompt"]
    assert "f1 hat Ausreißer > 3 IQR" in prompt
    assert "f9" in prompt and "Leakage-Verdacht" in prompt


def test_resume_kind_distinguishes_own_later_earlier(tmp_path):
    """R2-3: eigener Fehlversuch / Ruecksprung einer spaeteren Phase / erneuter Lauf nach
    Ueberarbeitung einer frueheren Phase bekommen verschiedene Resume-Saetze."""
    from crispfsm.phases import resume_kind
    state = new_state(RunConfig(run_id="r", workdir=str(tmp_path)))
    state.update(current_phase="model", gate_passed=False)
    assert resume_kind(state, "model") == "own"
    state.update(current_phase="eval", gate_passed=False)      # eval -> model
    assert resume_kind(state, "model") == "later"
    state.update(current_phase="du", gate_passed=False)        # spec_mismatch du -> bu
    assert resume_kind(state, "bu") == "later"
    state.update(current_phase="model", gate_passed=True)      # model bestanden -> eval erneut
    assert resume_kind(state, "eval") == "earlier"


def test_eval_rerun_after_model_retrain_gets_no_stale_feedback(tmp_path):
    """R2-3: EVAL laeuft nach bestandenem MODEL erneut — kein 'did NOT pass', kein Feedback
    des bestandenen MODEL-Gates im Prompt."""
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    os.makedirs(os.path.join(tmp_path, "work"), exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "holdout"), exist_ok=True)
    open(os.path.join(tmp_path, "holdout", "test_features.parquet"), "wb").close()
    state = new_state(cfg)
    state["attempts"]["eval"] = 1
    state.update(current_phase="model", gate_passed=True, gate_feedback="STALE-MODEL-FEEDBACK")
    backend = RecordingFakeBackend()
    make_phase_node("eval", backend, cfg)(state)
    prompt = backend.calls[0]["prompt"]
    assert "did NOT pass" not in prompt and "STALE-MODEL-FEEDBACK" not in prompt
    assert prompt.startswith("An earlier step of the pipeline has been revised")
