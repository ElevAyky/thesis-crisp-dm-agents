# src/crispfsm/state.py
from __future__ import annotations
from typing import TypedDict
from crispfsm.config import FSM_PHASES, RunConfig


class RunState(TypedDict):
    run_id: str
    dataset: str
    target: str
    workdir: str
    current_phase: str
    visited_phases: list[str]
    session_ids: dict[str, str]
    attempts: dict[str, int]
    task_spec: dict | None             # von der BU-Phase gefundene ML-Task-Spec
    du_report: dict | None             # Diagnose-Report der DU-Phase (Faktentreue-geprüft)
    abstained: bool                    # BU hat sich enthalten (ABSTAIN)
    bu_correct: bool | None            # Messung: BU-Zielspalte == kanonische Zielspalte
    bu_correct_first: bool | None      # dasselbe fuer die ERSTE BU-Entscheidung (B3)
    worker_cost_usd: float             # S4: verbrauchtes Run-Budget (nur Worker)
    worker_turns: int                  # S4: verbrauchte Turns (nur Worker)
    budget_exhausted: bool             # S4: Run-Kontingent leer -> FSM stoppt
    target_correct: bool | None        # Arm D: result.json.target_column == kanonisches Target
    gate_passed: bool
    gate_metrics: dict
    gate_feedback: str
    artifacts: dict
    usage_total: dict
    usage_per_phase: dict
    total_calls: int                   # Worker-Calls (Router-Limits lesen NUR diesen)
    gate_calls: int                    # Arm B: LLM-Prüfer-Calls (separat, kein Router-Input)
    success: bool
    final_f1: float | None             # nur bei SUCCESS (F1-auf-Erfolg)
    best_f1_measured: float | None     # bester gemessener Eval-F1 unabhängig von der
                                       # Schwelle (volle F1-Verteilung für Friedman;
                                       # Probe-Befund 18.09.: FAIL verlor sonst den Wert)


def new_state(cfg: RunConfig) -> RunState:
    return RunState(
        run_id=cfg.run_id,
        dataset=cfg.dataset,
        target=cfg.target,
        workdir=cfg.workdir,
        current_phase="",
        visited_phases=[],
        session_ids={},
        attempts={p: 0 for p in (*FSM_PHASES, "react")},
        task_spec=None,
        du_report=None,
        abstained=False,
        bu_correct=None,
        bu_correct_first=None,
        worker_cost_usd=0.0,
        worker_turns=0,
        budget_exhausted=False,
        target_correct=None,
        gate_passed=False,
        gate_metrics={},
        gate_feedback="",
        artifacts={},
        usage_total={"input_tokens": 0, "output_tokens": 0,
                     "cache_read_tokens": 0, "cache_creation_tokens": 0},
        usage_per_phase={},
        total_calls=0,
        gate_calls=0,
        success=False,
        final_f1=None,
        best_f1_measured=None,
    )


def best_f1(state: RunState, measured: float | None) -> float | None:
    """Neuer best_f1_measured nach einer Messung (None = nichts gemessen, z. B. Formfehler)."""
    prev = state.get("best_f1_measured")
    if measured is None:
        return prev
    return measured if prev is None else max(prev, measured)


_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_creation_tokens": 0}


def accumulate_usage(state: RunState, phase: str, usage: dict) -> tuple[dict, dict]:
    """Akkumuliert Worker-Usage in usage_total + usage_per_phase (immutable-Style,
    wie LangGraph-Updates es brauchen). Von phases.py und react.py genutzt."""
    usage_total = dict(state["usage_total"])
    for k, v in usage.items():
        usage_total[k] += v
    usage_per_phase = dict(state["usage_per_phase"])
    pu = dict(usage_per_phase.get(phase, _ZERO_USAGE))
    for k, v in usage.items():
        pu[k] += v
    usage_per_phase[phase] = pu
    return usage_total, usage_per_phase
