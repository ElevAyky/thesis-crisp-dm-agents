# src/crispfsm/graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from crispfsm.state import RunState
from crispfsm.config import RunConfig
from crispfsm.phases import make_phase_node


def _make_router(phase: str, cfg: RunConfig):
    """conditional_edge: liest nur gate_passed + Limits, gibt ein Routing-Label
    ('next' | 'retry' | 'stop') zurück. Kein State-Write."""
    def route(state: RunState) -> str:
        if state["gate_passed"]:
            return "next"
        if state.get("budget_exhausted"):   # S4: Run-Kontingent leer -> kein Retry
            return "stop"
        if (state["attempts"][phase] >= cfg.max_retries
                or state["total_calls"] >= cfg.safety_max_calls):
            return "stop"
        return "retry"
    return route


def _make_eval_router(cfg: RunConfig):
    """Wie _make_router, plus 'retry_self': Das Eval-Gate kennt zwei Fail-Arten,
    und das Feedback muss den VERURSACHENDEN Worker erreichen.
    - Formfehler (gate_metrics['form_error']: predictions.parquet fehlt/kaputt/
      falsche Spalte/Zeilenzahl/NaN) = Fehler des Eval-Deliverables. Der Model-
      Worker kann daran nichts fixen -> Self-Loop zu eval, das Feedback landet
      via Resume-Prompt beim Eval-Worker.
    - Schwellen-Fail (F1/Accuracy unter Schwelle) = Modellqualität -> Rücksprung
      zu model wie bisher.
    Limits (max_retries auf attempts['eval'], safety_max_calls) gelten für beide
    Retry-Arten gleichermaßen."""
    def route(state: RunState) -> str:
        if state["gate_passed"]:
            return "next"
        if state.get("budget_exhausted"):   # S4: Run-Kontingent leer -> kein Retry
            return "stop"
        if (state["attempts"]["eval"] >= cfg.max_retries
                or state["total_calls"] >= cfg.safety_max_calls):
            return "stop"
        if state["gate_metrics"].get("form_error"):
            return "retry_self"
        return "retry"
    return route


def _make_bu_router(cfg: RunConfig):
    """Wie _make_router, plus 'abstain': legitimer dritter Terminal-Zustand
    (bu-gate.md §3) — terminiert sofort, ohne Retry."""
    def route(state: RunState) -> str:
        if state["gate_passed"]:
            return "next"
        if state.get("budget_exhausted"):   # S4: Run-Kontingent leer -> kein Retry
            return "stop"
        if state["abstained"]:
            return "abstain"
        if (state["attempts"]["bu"] >= cfg.max_retries
                or state["total_calls"] >= cfg.safety_max_calls):
            return "stop"
        return "retry"
    return route


def _make_du_router(cfg: RunConfig):
    """Wie _make_router, plus 'back_bu': DU meldet spec_mismatch (Daten
    widersprechen der abgeleiteten Aufgabe) → Rücksprung zu bu (Deckers
    „Loop-Engineering", Meeting 20.07.), solange bu Retry-Budget hat.
    spec_mismatch zählt NICHT als DU-Retry — es ist kein Formfehler des
    Reports, sondern ein inhaltlicher Einwand gegen die BU-Spec."""
    def route(state: RunState) -> str:
        if state["gate_passed"]:
            return "next"
        if state.get("budget_exhausted"):   # S4: Run-Kontingent leer -> kein Retry
            return "stop"
        if state["total_calls"] >= cfg.safety_max_calls:
            return "stop"
        if state["gate_metrics"].get("spec_mismatch"):
            return "back_bu" if state["attempts"]["bu"] < cfg.max_retries else "stop"
        if state["attempts"]["du"] >= cfg.max_retries:
            return "stop"
        return "retry"
    return route


def build_graph(backend, cfg: RunConfig):
    b = StateGraph(RunState)
    b.add_node("bu", make_phase_node("bu", backend, cfg))
    b.add_node("du", make_phase_node("du", backend, cfg))
    b.add_node("prep", make_phase_node("prep", backend, cfg))
    b.add_node("model", make_phase_node("model", backend, cfg))
    b.add_node("eval", make_phase_node("eval", backend, cfg))

    b.add_edge(START, "bu")
    b.add_conditional_edges("bu", _make_bu_router(cfg),
                            {"next": "du", "retry": "bu", "stop": END, "abstain": END})
    # DU: 'back_bu' = spec_mismatch-Rücksprung (Loop-Engineering); sonst wie üblich.
    b.add_conditional_edges("du", _make_du_router(cfg),
                            {"next": "prep", "retry": "du", "stop": END, "back_bu": "bu"})
    # path_map: Label → Ziel-Node. 'retry' = Self-Loop (Resume), 'stop' = Abbruch.
    b.add_conditional_edges("prep", _make_router("prep", cfg),
                            {"next": "model", "retry": "prep", "stop": END})
    b.add_conditional_edges("model", _make_router("model", cfg),
                            {"next": "eval", "retry": "model", "stop": END})
    # Eval: 'next' (bestanden) = END; 'retry' (zu niedrige F1) springt zurück zu
    # Modeling; 'retry_self' (Formfehler im Predictions-Kontrakt) loopt zu eval selbst.
    b.add_conditional_edges("eval", _make_eval_router(cfg),
                            {"next": END, "retry_self": "eval", "retry": "model", "stop": END})
    return b.compile()
