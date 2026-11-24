# src/crispfsm/phases.py
from __future__ import annotations
import os, uuid
from crispfsm import gates, prompts, scoring, softgates
from crispfsm.events import log_event
from crispfsm.config import RunConfig
from crispfsm.pricing import cost_usd
from crispfsm.state import RunState, accumulate_usage, best_f1


def gate_part_and_baseline(cfg: RunConfig) -> tuple[str, str | None]:
    """Anteil + passende Baseline fuer das Evaluation-Gate (S1): im Kern 'val' mit der
    Validierungs-Baseline — das Gate-Feedback stammt nie aus dem finalen Test-Set."""
    part = scoring.gate_part(os.path.join(cfg.workdir, "holdout"))
    if part == "val":
        if cfg.baseline_pred_path and not cfg.baseline_val_path:
            raise ValueError("Holdout mit Validierungsanteil, aber keine Validierungs-Baseline "
                             "(cfg.baseline_val_path) — Gate wuerde gegen die falsche Baseline messen")
        return part, cfg.baseline_val_path
    return part, cfg.baseline_pred_path


def _evaluation_gate(work: str, cfg: RunConfig, work_target: str):
    # Misst gegen die ECHTE Zielspalte (cfg.target): das Gate ist die Qualitaetspruefung
    # gegen verborgene Labels — work_target (BU-Wahl) spielt hier bewusst keine Rolle.
    part, baseline = gate_part_and_baseline(cfg)
    check = gates.check_eval
    return check(work, holdout_dir=os.path.join(cfg.workdir, "holdout"),
                 target=cfg.target, metric=cfg.eval_metric, threshold=cfg.eval_f1_threshold,
                 baseline_pred_path=baseline, part=part)


# Signatur (work, cfg, work_target): DU/Prep pruefen gegen die Zielspalte, mit der der
# Prozess arbeitet (B3: die der eigenen BU-Phase).
_GATE_FUNCS = {
    "bu": lambda work, cfg, t: gates.check_bu(work),
    "du": lambda work, cfg, t: gates.check_du(work, target=t),
    "prep": lambda work, cfg, t: gates.check_prep(work, target=t),
    "model": lambda work, cfg, t: gates.check_model(work),
    "eval": _evaluation_gate,
}
ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit"]
# Arm A-Single (H1, 13.11.): Schluessel der EINEN Worker-Session in state["session_ids"].
SINGLE_SESSION_KEY = "single"


def resume_kind(state: RunState, phase: str) -> str:
    """R2-3: Woher kommt ein erneuter Lauf dieser Phase? (prompts._resume_head)
    own = das eigene Gate ist gerade gescheitert; later = eine spaetere Phase ist
    gescheitert und springt zurueck (eval -> model, du -> bu); earlier = eine
    fruehere Phase hat bestanden und die Kette laeuft erneut hierher."""
    if state["current_phase"] == phase and not state["gate_passed"]:
        return "own"
    return "earlier" if state["gate_passed"] else "later"


def make_phase_node(phase: str, backend, cfg: RunConfig):
    # Arm B (soft_fsm): identischer Node-Ablauf und identische Router — nur der
    # Gate-MECHANISMUS ist getauscht (LLM-Prüfer statt _GATE_FUNCS). Worker-
    # Prompts, Session-Handling und Datenzugänge bleiben byte-gleich zu Arm A.
    soft = cfg.arm == "soft_fsm"
    # Arm A-Single (hard_fsm_single, H1, docs/h1-single-session.md): identisch zu Arm A —
    # gleiche Prompts, Gates, Router, Budgets —, nur arbeitet EINE Worker-Session alle
    # Phasen ab (claude -p --resume ueber die Phasengrenzen), statt einer frischen
    # Session pro Phase. Getrennt bleiben darum zwei Dinge, die in Arm A zusammenfallen:
    #   resume     = Prompt-Wahl (Resume-Prompt ab dem 2. Versuch einer Phase) — wie A
    #   cli_resume = ob die CLI eine bestehende Session fortsetzt
    single = cfg.arm == "hard_fsm_single"

    def node(state: RunState) -> dict:
        work = os.path.join(state["workdir"], "work")
        attempt = state["attempts"][phase]            # 0 = erster Versuch
        resume = attempt > 0

        # Output-Dir der Phase vorab anlegen (deterministisch; spart eine Worker-Retry-Runde)
        os.makedirs(os.path.join(work, phase), exist_ok=True)

        # Eval: Test-FEATURES erst jetzt in Worker-Reichweite kopieren.
        # Labels bleiben in holdout/ (Spec 2026-07-18 §1.1) — das Gate scored selbst.
        if phase == "eval":
            import shutil
            os.makedirs(os.path.join(work, "data"), exist_ok=True)
            shutil.copy(os.path.join(state["workdir"], "holdout", "test_features.parquet"),
                        os.path.join(work, "data", "test_features.parquet"))

        # Session-ID pro Phase (für Resume stabil halten); A-Single: eine fuer den ganzen Run
        if single:
            session_id = state["session_ids"].get(SINGLE_SESSION_KEY) or str(uuid.uuid4())
            cli_resume = SINGLE_SESSION_KEY in state["session_ids"]
        else:
            session_id = state["session_ids"].get(phase) or str(uuid.uuid4())
            cli_resume = resume

        # B3 (23.09.): ab DU arbeitet der Prozess mit der Zielspalte, die SEINE BU-Phase
        # bestimmt hat — nicht mit der kanonischen. Wie bei C/D traegt der Run die Folgen
        # einer falschen BU-Entscheidung, sofern DU sie nicht per spec_mismatch zurueck
        # an BU gibt. Nur das Evaluation-Gate und das Endurteil kennen die echte Spalte.
        work_target = (state["task_spec"] or {}).get("target_column") or cfg.target
        kind = resume_kind(state, phase)

        if phase == "bu":
            prompt = (prompts.bu_resume_prompt(cfg.task_prompt, state["gate_feedback"], kind=kind)
                      if resume else prompts.bu_initial_prompt(cfg.task_prompt))
        elif phase == "du":
            # critical_columns: BU-Hinweis an DU (roadmap §1 — Info, nicht pass-kritisch)
            crit = (state["task_spec"] or {}).get("critical_columns")
            prompt = (prompts.du_resume_prompt(work_target, state["gate_feedback"], crit,
                                               task_prompt=cfg.task_prompt, kind=kind)
                      if resume else prompts.du_initial_prompt(work_target, crit,
                                                               task_prompt=cfg.task_prompt))
        elif resume:
            prompt = prompts.resume_prompt(phase, work_target, state["gate_feedback"],
                                           du_report=state["du_report"],
                                           task_prompt=cfg.task_prompt, kind=kind)
        else:
            prompt = prompts.initial_prompt(phase, work_target, du_report=state["du_report"],
                                            task_prompt=cfg.task_prompt)

        # S4: gemeinsames Run-Kontingent. Ist es leer, laeuft kein Worker mehr; der
        # Router stoppt (budget_exhausted). Sonst bekommt der Call hoechstens den Rest.
        remaining_usd = (None if cfg.run_budget_usd is None
                         else cfg.run_budget_usd - state["worker_cost_usd"])
        remaining_turns = cfg.run_max_turns - state["worker_turns"]
        if (remaining_usd is not None and remaining_usd <= 0) or remaining_turns <= 0:
            return {"current_phase": phase, "gate_passed": False, "budget_exhausted": True,
                    "gate_feedback": "run budget exhausted"}
        call_budget = (remaining_usd if cfg.budget_usd_per_call is None
                       else cfg.budget_usd_per_call if remaining_usd is None
                       else min(cfg.budget_usd_per_call, remaining_usd))
        result = backend.run(prompt, cwd=work, session_id=session_id, resume=cli_resume,
                             max_turns=min(cfg.max_turns, remaining_turns),
                             budget_usd=call_budget, allowed_tools=ALLOWED_TOOLS)
        try:
            call_cost = cost_usd(result.usage, cfg.model)
        except KeyError:                    # unbekanntes Modell (Debug) -> nicht verbuchbar
            call_cost = 0.0
        worker_cost = state["worker_cost_usd"] + call_cost
        worker_turns = state["worker_turns"] + int(result.raw.get("num_turns") or 0)
        budget_exhausted = ((cfg.run_budget_usd is not None and worker_cost >= cfg.run_budget_usd)
                            or worker_turns >= cfg.run_max_turns)

        if soft:
            gate, judge = softgates.run_soft_gate(phase, backend, cfg, work, target=work_target)
        else:
            gate, judge = _GATE_FUNCS[phase](work, cfg, work_target), None

        # Usage akkumulieren; der Prüfer-Call (Arm B) wird separat unter
        # gate_<phase> verbucht — Gate-Kosten sind AV, kein Rauschen im Worker.
        usage_total, usage_per_phase = accumulate_usage(state, phase, result.usage.as_dict())
        gate_calls = state["gate_calls"]
        if judge is not None:
            usage_total, usage_per_phase = accumulate_usage(
                {**state, "usage_total": usage_total, "usage_per_phase": usage_per_phase},
                f"gate_{phase}", judge.usage.as_dict())
            gate_calls += 1

        # spec_mismatch (du): kein Fehlversuch der Phase — DU hat korrekt einen
        # Widerspruch zur BU-Spec gemeldet. Zählt darum NICHT ins DU-Retry-Budget
        # (den Rücksprung zahlt das bu-Budget, s. _make_du_router) und die Session
        # wird verworfen: nach neuer Spec soll DU frisch diagnostizieren, nicht als
        # "Resume" die alte Einschätzung fortschreiben. total_calls zählt trotzdem.
        spec_mismatch = phase == "du" and bool(gate.metrics.get("spec_mismatch"))
        attempts = dict(state["attempts"])
        if not spec_mismatch:
            attempts[phase] += 1
        if single:
            # Ein Agent vergisst nichts: auch nach spec_mismatch laeuft DIESELBE Session
            # weiter (DU bekommt wie in A den Initial-Prompt, aber mit Vorgeschichte).
            session_ids = {**state["session_ids"], SINGLE_SESSION_KEY: session_id}
        else:
            session_ids = {**state["session_ids"], phase: session_id}
            if spec_mismatch:
                session_ids.pop(phase)
        visited = state["visited_phases"] + ([phase] if phase not in state["visited_phases"] else [])

        event = {
            # attempt = Ausführungs-Nummer dieses Calls (bei spec_mismatch bewusst
            # ungleich dem persistierten Retry-Zähler)
            "phase": phase, "attempt": state["attempts"][phase] + 1, "session_id": session_id,
            "resume": resume, "usage": result.usage.as_dict(),
            "latency_ms": result.latency_ms,
            "worker_text": result.text,                        # die eigentliche LLM-Ausgabe (result message)
            "worker_num_turns": result.raw.get("num_turns"),   # agentische Schritte (None bei Fakes)
            "gate_passed": gate.passed, "gate_metrics": gate.metrics,
            "gate_feedback": gate.feedback, "abstain": gate.abstain,
        }
        if single:
            # nur A-Single; A-Events bleiben unveraendert. cli_total_cost_usd = Kostenstand
            # der CLI-Session (kumulativ ueber --resume?) — Pruefpunkt, ob --max-budget-usd
            # pro Call oder pro Session greift (Review 13.11.).
            event["cli_resume"] = cli_resume
            event["cli_total_cost_usd"] = result.raw.get("total_cost_usd")
        if judge is not None:
            event.update({
                "gate_session_id": judge.session_id,
                "gate_usage": judge.usage.as_dict(),
                "gate_latency_ms": judge.latency_ms,
                "judge_text": judge.text,
                "judge_num_turns": judge.raw.get("num_turns"),
            })
        log_event(state["workdir"], event)

        update = {
            "current_phase": phase,
            "visited_phases": visited,
            "session_ids": session_ids,
            "attempts": attempts,
            "gate_passed": gate.passed,
            "gate_metrics": gate.metrics,
            "gate_feedback": gate.feedback,
            "usage_total": usage_total,
            "usage_per_phase": usage_per_phase,
            "total_calls": state["total_calls"] + 1,
            # total_calls zählt bewusst NUR Worker-Calls (Router-Parität zu Arm A);
            # Prüfer-Calls laufen separat in gate_calls.
            "gate_calls": gate_calls,
            "worker_cost_usd": worker_cost, "worker_turns": worker_turns,
            "budget_exhausted": budget_exhausted,
        }
        if phase == "bu":
            update["abstained"] = bool(gate.abstain)
            # .get statt [] — ein LLM-Prüfer (Arm B) kann ein formal lückenhaftes
            # oder unlesbares Artefakt durchwinken; dann fehlt eben die Spec
            # (Soft-Arm-Befund, kein Harness-Crash).
            spec = gate.metrics.get("spec") if gate.passed else None
            if spec is not None:
                update["task_spec"] = spec
                # Messung, kein Gate-Kriterium (bu-gate.md §4): stimmt die gefundene
                # Zielspalte mit der kanonischen (neutralisierten) überein?
                update["bu_correct"] = spec.get("target_column") == cfg.target
                # B3: erste BU-Entscheidung festhalten — zusammen mit bu_correct (letzte)
                # zeigt sie, ob ein spec_mismatch-Ruecksprung eine falsche Zielspalte
                # korrigiert hat.
                if state.get("bu_correct_first") is None:
                    update["bu_correct_first"] = update["bu_correct"]
        if phase == "du" and gate.passed and gate.metrics.get("report") is not None:
            update["du_report"] = gate.metrics["report"]
        if phase == "eval":
            # Jede Messung zählt für die F1-Verteilung, auch unter der Schwelle
            # (best_f1_measured); final_f1 bleibt SUCCESS-gebunden.
            update["best_f1_measured"] = best_f1(state, gate.metrics.get("f1"))
        if phase == "eval" and gate.passed:
            # scored_ok fehlt bei Arm A (check_eval scored selbst => Pass impliziert
            # scorbar). Arm B: Judge-Pass allein reicht nicht — success nur, wenn
            # die Harness-Messung die Predictions auch scoren konnte.
            scored_ok = bool(gate.metrics.get("scored_ok", True))
            update["success"] = scored_ok
            update["final_f1"] = gate.metrics.get("f1") if scored_ok else None
        return update
    return node
