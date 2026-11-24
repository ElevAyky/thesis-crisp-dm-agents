# tests/test_single_session.py
"""Arm A-Single (hard_fsm_single, H1-Nachtrag 13.11., docs/h1-single-session.md).

Der Arm darf sich von Arm A in GENAU einem Punkt unterscheiden: eine Worker-Session
fuer den ganzen Run (CLI --resume ueber die Phasengrenzen) statt einer frischen
Session pro Phase. Prompts, Gates, Router und Zaehler muessen byte-gleich bleiben —
und Arm A selbst darf sich durch den Umbau nicht veraendern."""
import json
import os

import pytest

from crispfsm.config import RunConfig
from crispfsm.datasets import prepare_dataset
from crispfsm.graph import build_graph
from crispfsm.phases import SINGLE_SESSION_KEY
from crispfsm.state import new_state
from tests.test_graph import HallucinatingDuBackend, ScriptedBackend, SpecMismatchOnceBackend


class Recorder:
    """Mixin: schneidet jeden Worker-Call mit (Prompt, Session, CLI-Resume)."""

    def _record(self, prompt, session_id, resume):
        self.calls.append({"prompt": prompt, "session_id": session_id, "resume": resume})


class RecordingScripted(ScriptedBackend, Recorder):
    def __init__(self, fail_eval_once=False):
        super().__init__(fail_eval_once=fail_eval_once)
        self.calls = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self._record(prompt, session_id, resume)
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


class RecordingSpecMismatch(SpecMismatchOnceBackend, Recorder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self._record(prompt, session_id, resume)
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


class RecordingDuRetry(HallucinatingDuBackend, Recorder):
    """DU faellt beim 1. Versuch durchs eigene Gate (Self-Retry, haeufigster Resume-Fall:
    26 von 90 A-Runs) und besteht beim 2."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self._record(prompt, session_id, resume)
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def _run(tmp_path, backend, arm):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), arm=arm)
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    graph = build_graph(backend, cfg)
    final = graph.invoke(new_state(cfg), config={"recursion_limit": cfg.recursion_limit})
    events = [json.loads(line) for line in
              open(os.path.join(tmp_path, "events.jsonl")).read().splitlines() if line.strip()]
    return final, events


def _pair(tmp_path, make_backend):
    """Faehrt denselben Skript-Backend-Ablauf einmal als A, einmal als A-Single."""
    a, s = make_backend(), make_backend()
    fa, ea = _run(tmp_path / "a", a, "hard_fsm")
    fs, es = _run(tmp_path / "s", s, "hard_fsm_single")
    return (a, fa, ea), (s, fs, es)


SCENARIOS = {
    "clean": lambda: RecordingScripted(),
    "eval_backjump": lambda: RecordingScripted(fail_eval_once=True),
    "spec_mismatch": lambda: RecordingSpecMismatch(),
    "du_self_retry": lambda: RecordingDuRetry(),
}


def test_config_accepts_single_arm():
    assert RunConfig(run_id="r", workdir="w", arm="hard_fsm_single").arm == "hard_fsm_single"
    with pytest.raises(ValueError):
        RunConfig(run_id="r", workdir="w", arm="hard_fsm_multi")


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_single_uses_one_session_and_resumes_after_first_call(tmp_path, scenario):
    _, (s, final, _) = _pair(tmp_path, SCENARIOS[scenario])
    sids = {c["session_id"] for c in s.calls}
    assert len(sids) == 1                                   # EIN Agent fuer alle Phasen
    assert [c["resume"] for c in s.calls] == [False] + [True] * (len(s.calls) - 1)
    assert final["session_ids"] == {SINGLE_SESSION_KEY: sids.pop()}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_single_sends_byte_identical_prompts_and_same_flow_as_a(tmp_path, scenario):
    """Der einzige Unterschied zu A ist die Session — Prompt-Folge, Phasenfolge,
    Versuchszaehler und Ergebnis muessen gleich sein."""
    (a, fa, ea), (s, fs, es) = _pair(tmp_path, SCENARIOS[scenario])
    assert [c["prompt"] for c in s.calls] == [c["prompt"] for c in a.calls]
    for key in ("success", "final_f1", "attempts", "visited_phases", "total_calls",
                "current_phase", "gate_passed"):
        assert fs[key] == fa[key], key
    # Prompt-Resume (Versuch > 1 einer Phase) bleibt wie in A im Event-Feld "resume"
    assert [e["resume"] for e in es] == [e["resume"] for e in ea]
    assert [e["phase"] for e in es] == [e["phase"] for e in ea]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_arm_a_session_handling_unchanged(tmp_path, scenario):
    """Regression: Arm A behaelt eine Session PRO PHASE; die CLI setzt nur bei einem
    erneuten Versuch derselben Phase fort, und Events tragen kein cli_resume."""
    (a, fa, ea), _ = _pair(tmp_path, SCENARIOS[scenario])
    by_phase = {}
    for c, e in zip(a.calls, ea):
        by_phase.setdefault(e["phase"], set()).add(c["session_id"])
        assert c["resume"] == e["resume"]                 # CLI-Resume == Prompt-Resume
        assert "cli_resume" not in e and "cli_total_cost_usd" not in e
    sessions = [sid for sids in by_phase.values() for sid in sids]
    assert len(sessions) == len(set(sessions))            # keine Session ueber Phasen geteilt
    # nur DU nach spec_mismatch bekommt eine zweite Session, alle anderen Phasen genau eine
    for phase, sids in by_phase.items():
        assert len(sids) == (2 if phase == "du" and scenario == "spec_mismatch" else 1), phase
    assert SINGLE_SESSION_KEY not in fa["session_ids"]


def test_single_du_self_retry_resumes_with_feedback_in_same_session(tmp_path):
    (a, _, ea), (s, _, es) = _pair(tmp_path, SCENARIOS["du_self_retry"])
    du_a = [c for c, e in zip(a.calls, ea) if e["phase"] == "du"]
    du_s = [c for c, e in zip(s.calls, es) if e["phase"] == "du"]
    assert [c["resume"] for c in du_a] == [False, True]   # A: frische DU-Session, dann Resume
    assert [c["resume"] for c in du_s] == [True, True]    # A-Single: laeuft seit BU weiter
    assert du_s[1]["prompt"] == du_a[1]["prompt"] and "Quality Gate" in du_s[1]["prompt"]


def test_single_budget_exhausted_before_first_call_sets_no_session(tmp_path):
    """Leeres Run-Kontingent: kein Worker-Call, also auch kein Session-Schluessel —
    sonst schickte ein spaeterer Call --resume fuer eine nie angelegte Session."""
    from crispfsm.phases import make_phase_node
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), arm="hard_fsm_single", run_budget_usd=0.0)
    os.makedirs(tmp_path / "work", exist_ok=True)
    backend = RecordingScripted()
    update = make_phase_node("bu", backend, cfg)(new_state(cfg))
    assert update["budget_exhausted"] is True
    assert backend.calls == [] and "session_ids" not in update


def test_single_keeps_session_after_spec_mismatch(tmp_path):
    """A verwirft die DU-Session nach spec_mismatch (frische Diagnose); A-Single
    behaelt die eine Session — der zweite DU-Lauf bekommt trotzdem den Initial-Prompt."""
    (a, _, _), (s, _, es) = _pair(tmp_path, SCENARIOS["spec_mismatch"])
    assert a.du_sessions[0] != a.du_sessions[1]
    assert s.du_sessions[0] == s.du_sessions[1]
    assert "did NOT pass" not in s.du_prompts[1]
    du_events = [e for e in es if e["phase"] == "du"]
    assert [e["cli_resume"] for e in du_events] == [True, True]


def test_single_events_log_cli_resume(tmp_path):
    _, (s, _, es) = _pair(tmp_path, SCENARIOS["eval_backjump"])
    assert [e["cli_resume"] for e in es] == [False] + [True] * (len(es) - 1)
    assert len({e["session_id"] for e in es}) == 1
