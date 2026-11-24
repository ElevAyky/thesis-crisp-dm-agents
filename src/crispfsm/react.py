"""Freie Arme C + D (Spec 2026-07-18 §2, roadmap §1): EINE agentische
claude-p-Session im selben Harness wie Arm A. Kein Retry, kein Gate-Feedback —
die Retries mit deterministischem Feedback SIND das Treatment von Arm A.
Deliverable-Auswertung (Scoring, result.json) ist Messung, kein Gate.
C und D unterscheiden sich in genau einem Faktor: Arm C (crisp_react) bekommt
die CRISP-Anleitung in den Prompt angehängt, Arm D (naked_react) nicht —
alles andere (Budget, Turns, Tools, Datenzugang, Scoring) ist identisch."""
from __future__ import annotations
import json
import os
import shutil
import uuid

from langgraph.graph import StateGraph, START, END

from crispfsm import prompts
from crispfsm.config import RunConfig
from crispfsm.events import log_event
from crispfsm.phases import ALLOWED_TOOLS
from crispfsm.scoring import compare_part_to_baseline, score_part
from crispfsm.state import RunState, accumulate_usage, best_f1


def _read_result_json(work: str) -> tuple[dict | None, str | None]:
    """result.json des Workers -> (declared, diagnose). Unparsebar zählt wie
    fehlend (Spec §4), aber die Diagnose geht ins Event-Log: „Worker schrieb
    Müll" muss von „Worker schrieb nichts" unterscheidbar bleiben
    (Fabrikation vs. Unterlassung). OSError (z. B. result.json ist ein
    Verzeichnis) darf den Run nicht crashen — No-Retry-Posture: FAIL/Diagnose
    statt Exception."""
    path = os.path.join(work, "result.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            declared = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        return None, f"result.json ist kein valides JSON: {e}"
    if not isinstance(declared, dict):
        return None, "result.json ist kein JSON-Objekt (Array o. ä.)"
    return declared, None


def make_react_node(backend, cfg: RunConfig):
    def node(state: RunState) -> dict:
        work = os.path.join(state["workdir"], "work")
        # Einmalige Session: Features von Anfang an in Worker-Reichweite
        # (gestaffelter Datenzugang ist Arm-A-Treatment, Spec §2.2).
        os.makedirs(os.path.join(work, "data"), exist_ok=True)
        shutil.copy(os.path.join(state["workdir"], "holdout", "test_features.parquet"),
                    os.path.join(work, "data", "test_features.parquet"))

        session_id = str(uuid.uuid4())
        # Budget-Paritaet (S4, 23.09.): dasselbe Gesamtkontingent pro Run wie A/B
        # (run_budget_usd / run_max_turns), hier in einer einzigen Session.
        budget_usd = cfg.run_budget_usd
        # Einziger C↔D-Unterschied: CRISP-Anleitung im Prompt (Purity-Tests
        # sichern beide Richtungen — D ohne CRISP-Vokabular, C ohne Enforcement).
        prompt = (prompts.crisp_react_prompt(cfg.task_prompt)
                  if cfg.arm == "crisp_react"
                  else prompts.react_prompt(cfg.task_prompt))
        result = backend.run(prompt, cwd=work,
                             session_id=session_id, resume=False,
                             max_turns=cfg.react_max_turns,
                             budget_usd=budget_usd,
                             allowed_tools=ALLOWED_TOOLS)

        declared, result_json_error = _read_result_json(work)
        abstained = bool(declared and declared.get("cannot_determine"))
        target_correct = None
        if not abstained and declared and declared.get("target_column"):
            target_correct = declared["target_column"] == cfg.target

        # Finale Bewertung auf dem TEST-Anteil (S1): die Feature-Datei enthaelt auch
        # Validierungszeilen, die fuer C/D keine Rolle spielen (nur A/B-Gates nutzen sie).
        holdout_dir = os.path.join(state["workdir"], "holdout")
        score = score_part(os.path.join(work, "predictions.parquet"), holdout_dir, "test",
                           target=cfg.target)
        # Paritäts-Fix (Scan-Befund 08.08.): die FSM-Arme gaten SUCCESS auf
        # metric >= threshold (gates.check_eval) — die freien Arme brauchen
        # dieselbe SUCCESS-Definition, sonst ist die AV arm-abhängig definiert
        # (vorher: reiner Form-Check, SUCCESS auch bei F1 0.0).
        metric_val = {"f1": score.f1, "accuracy": score.accuracy}.get(cfg.eval_metric)
        gate_metrics = {"f1": score.f1, "accuracy": score.accuracy,
                        "form_error": not score.ok}
        if score.ok and cfg.baseline_pred_path and cfg.eval_metric == "f1":
            # Erfolgsregel seit 23.09. (Paritaet zu gates.check_eval): nicht
            # signifikant schlechter als die Default-Baseline, gepaarter Bootstrap.
            cmp = compare_part_to_baseline(
                os.path.join(work, "predictions.parquet"), holdout_dir, "test",
                cfg.baseline_pred_path, target=cfg.target)
            gate_metrics.update({"baseline_f1": cmp.baseline_f1, "diff": cmp.diff,
                                 "ci_lo": cmp.ci_lo, "ci_hi": cmp.ci_hi,
                                 "noninferior": cmp.noninferior})
            passes = cmp.noninferior
        else:
            passes = metric_val is not None and metric_val >= cfg.eval_f1_threshold
        success = score.ok and not abstained and passes    # ABSTAIN gewinnt (Spec §4)

        usage_total, usage_per_phase = accumulate_usage(state, "react",
                                                        result.usage.as_dict())
        log_event(state["workdir"], {
            "phase": "react", "attempt": 1, "session_id": session_id,
            "resume": False, "usage": result.usage.as_dict(),
            "latency_ms": result.latency_ms,
            "worker_text": result.text,
            "worker_num_turns": result.raw.get("num_turns"),
            "gate_passed": success, "gate_metrics": gate_metrics,
            "gate_feedback": score.feedback, "abstain": abstained,
            "result_json_error": result_json_error,
        })

        return {
            "current_phase": "react",
            "visited_phases": ["react"],
            "session_ids": {**state["session_ids"], "react": session_id},
            "attempts": {**state["attempts"], "react": 1},
            "gate_passed": success,
            # form_error: Predictions-Kontrakt gebrochen (fehlt/Spalte/Zeilenzahl/
            # Label-Raum) — Parität zu phases.py, Basis für fail_reason in run.py
            "gate_metrics": gate_metrics,
            "gate_feedback": score.feedback,
            "usage_total": usage_total,
            "usage_per_phase": usage_per_phase,
            "total_calls": state["total_calls"] + 1,
            "abstained": abstained,
            "target_correct": target_correct,
            "success": success,
            "final_f1": score.f1 if success else None,
            "best_f1_measured": best_f1(state, score.f1),
        }
    return node


def build_react_graph(backend, cfg: RunConfig):
    b = StateGraph(RunState)
    b.add_node("react", make_react_node(backend, cfg))
    b.add_edge(START, "react")
    b.add_edge("react", END)
    return b.compile()
