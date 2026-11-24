# tests/test_softgates.py
"""Arm B (Soft CRISP, roadmap §1): gleiche FSM-Topologie/Router wie Arm A, aber
das Quality Gate entscheidet ein LLM-Prüfer (softgates.py). Getestet werden:
Routing-Semantik über LLM-Verdicts (abstain / spec_mismatch / form_error),
judge_error als regulärer Gate-Fail, fabrikationssichere Erfolgsmessung
(Judge-Pass allein reicht nicht) sowie Paritäts-/Purity-Auflagen analog Arm D
(Wording-, Tool-, Budget-Parität; kein Label-Zugang für den Prüfer)."""
import json
import os
import re

import pandas as pd
import pytest

from crispfsm import phases, prompts, softgates
from crispfsm.backend import TokenUsage, WorkerResult
from crispfsm.config import RunConfig
from crispfsm.datasets import prepare_dataset
from crispfsm.graph import build_graph
from crispfsm.scoring import ScoreResult
from crispfsm.state import new_state
from tests.test_graph import ScriptedBackend

JUDGE_MARKER = "GATE CHECK"
WORKER_PHASE_KEYWORDS = ["Business Understanding", "Data Understanding",
                         "Data Preparation", "Modeling", "Evaluation"]


class SoftBackend(ScriptedBackend):
    """Bedient Worker- UND Prüfer-Calls im Soft-Arm. Worker-Calls delegieren an
    ScriptedBackend (valide Artefakte). Prüfer-Calls (erkannt am JUDGE_MARKER)
    schreiben ein Verdict: per Default „ehrlich" (mit den deterministischen
    Gates aus gates.py nachgerechnet — Test-Cheat: idealer Prüfer), per
    judge_overrides pro Phase skriptbar:
      dict  -> immer dieses Verdict
      list  -> Verdicts der Reihe nach verbrauchen, danach ehrlich
      "SILENT" (auch in Listen) -> kein Verdict schreiben (judge_error)."""
    name = "soft_scripted"

    def __init__(self, judge_overrides=None, fail_eval_once=False):
        super().__init__(fail_eval_once=fail_eval_once)
        self.judge_overrides = {k: (list(v) if isinstance(v, list) else v)
                                for k, v in (judge_overrides or {}).items()}
        self.judge_calls = []
        self.worker_calls = []

    def _honest_verdict(self, phase, cwd):
        from crispfsm import gates, scoring
        target = pd.read_parquet(os.path.join(cwd, "data", "train.parquet")).columns[-1]
        if phase == "eval":
            s = scoring.score_predictions(
                os.path.join(cwd, "eval", "predictions.parquet"),
                os.path.join(cwd, "..", "holdout", "test.parquet"), target=target)
            return {"passed": s.ok, "feedback": s.feedback, "form_error": not s.ok}
        g = {"bu": lambda: gates.check_bu(cwd),
             "du": lambda: gates.check_du(cwd, target=target),
             "prep": lambda: gates.check_prep(cwd, target=target),
             "model": lambda: gates.check_model(cwd)}[phase]()
        v = {"passed": g.passed, "feedback": g.feedback}
        if phase == "bu":
            v["abstain"] = g.abstain
        if phase == "du" and g.metrics.get("spec_mismatch"):
            v["spec_mismatch"] = True
        return v

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        m = re.search(r"GATE CHECK '(\w+)'", prompt)
        if m:
            phase = m.group(1)
            self.judge_calls.append({
                "phase": phase, "prompt": prompt, "session_id": session_id,
                "resume": resume, "max_turns": max_turns, "budget_usd": budget_usd,
                "allowed_tools": allowed_tools, "cwd": cwd,
            })
            override = self.judge_overrides.get(phase)
            if isinstance(override, list):
                verdict = override.pop(0) if override else self._honest_verdict(phase, cwd)
            elif override is not None:
                verdict = override
            else:
                verdict = self._honest_verdict(phase, cwd)
            if verdict != "SILENT":
                os.makedirs(os.path.join(cwd, "gate"), exist_ok=True)
                json.dump(verdict, open(os.path.join(cwd, "gate", "verdict.json"), "w"))
            return WorkerResult(text="verdict", usage=TokenUsage(5, 1), latency_ms=1.0,
                                session_id=session_id, raw={})
        self.worker_calls.append({"prompt": prompt, "session_id": session_id,
                                  "resume": resume, "budget_usd": budget_usd})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def _cfg(tmp_path, **kwargs):
    return RunConfig(run_id="r1", workdir=str(tmp_path), arm="soft_fsm", **kwargs)


def _run(tmp_path, backend, **kwargs):
    cfg = _cfg(tmp_path, **kwargs)
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    graph = build_graph(backend, cfg)
    return graph.invoke(new_state(cfg), config={"recursion_limit": cfg.recursion_limit})


def _read_events(tmp_path):
    return [json.loads(l) for l in
            open(os.path.join(tmp_path, "events.jsonl")).read().strip().splitlines()]


# ---------------------------------------------------------------------------
# End-to-End & Accounting
# ---------------------------------------------------------------------------

def test_soft_graph_end_to_end_success(tmp_path):
    backend = SoftBackend()
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["final_f1"] == 1.0
    assert final["visited_phases"] == ["bu", "du", "prep", "model", "eval"]
    assert final["bu_correct"] is True
    assert final["du_report"]["n_rows"] > 0            # Feed-Forward geladen
    # 5 Worker- + 5 Prüfer-Calls; Router-Zähler sieht nur Worker
    assert final["total_calls"] == 5
    assert final["gate_calls"] == 5
    assert [c["phase"] for c in backend.judge_calls] == ["bu", "du", "prep", "model", "eval"]


def test_soft_gate_usage_tracked_separately(tmp_path):
    final = _run(tmp_path, SoftBackend())
    # Worker: 10 in / 2 out je Call, Prüfer: 5 in / 1 out je Call
    assert final["usage_per_phase"]["bu"]["input_tokens"] == 10
    assert final["usage_per_phase"]["gate_bu"]["input_tokens"] == 5
    assert final["usage_total"]["input_tokens"] == 5 * 10 + 5 * 5
    assert final["usage_total"]["output_tokens"] == 5 * 2 + 5 * 1


def test_soft_events_carry_gate_call_data(tmp_path):
    _run(tmp_path, SoftBackend())
    events = _read_events(tmp_path)
    assert len(events) == 5
    for ev in events:
        assert ev["gate_usage"] == {"input_tokens": 5, "output_tokens": 1,
                                    "cache_read_tokens": 0, "cache_creation_tokens": 0}
        assert ev["gate_session_id"] != ev["session_id"]
        assert ev["judge_text"] == "verdict"


# ---------------------------------------------------------------------------
# Routing über LLM-Verdicts
# ---------------------------------------------------------------------------

def test_soft_judge_fail_feeds_back_to_worker_and_retries(tmp_path):
    backend = SoftBackend(judge_overrides={
        "prep": [{"passed": False, "feedback": "Sentinel-Mangel: NaN in f9"}]})
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["attempts"]["prep"] == 2
    prep_prompts = [c["prompt"] for c in backend.worker_calls
                    if "Data Preparation" in c["prompt"]]
    assert len(prep_prompts) == 2
    assert "Quality Gate" in prep_prompts[1]
    assert "Sentinel-Mangel: NaN in f9" in prep_prompts[1]   # LLM-Feedback kam an


def test_soft_bu_abstain_verdict_terminates(tmp_path):
    backend = SoftBackend(judge_overrides={
        "bu": {"passed": False, "abstain": True, "feedback": "kein Ziel ableitbar"}})
    final = _run(tmp_path, backend)
    assert final["abstained"] is True
    assert final["success"] is False
    assert final["visited_phases"] == ["bu"]
    assert final["attempts"]["bu"] == 1                # ABSTAIN retryt NICHT


def test_soft_du_spec_mismatch_verdict_loops_back_to_bu(tmp_path):
    backend = SoftBackend(judge_overrides={
        "du": [{"passed": False, "spec_mismatch": True,
                "feedback": "Daten widersprechen der Spec"}]})
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["attempts"]["bu"] == 2                # Rücksprung lief über bu
    assert final["attempts"]["du"] == 1                # kein DU-Retry-Budget verbraucht
    du_workers = [c for c in backend.worker_calls if "Data Understanding" in c["prompt"]]
    assert len(du_workers) == 2
    # 2. DU-Lauf frisch: neue Session, Initial-Prompt (Semantik wie Arm A)
    assert du_workers[1]["session_id"] != du_workers[0]["session_id"]
    assert "did NOT pass" not in du_workers[1]["prompt"]


def test_soft_eval_form_error_verdict_retries_eval_itself(tmp_path):
    class WrongColumnOnceBackend(SoftBackend):
        def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
            if JUDGE_MARKER not in prompt and "Evaluation" in prompt:
                self.eval_calls += 1
                self.worker_calls.append({"prompt": prompt, "session_id": session_id,
                                          "resume": resume, "budget_usd": budget_usd})
                os.makedirs(os.path.join(cwd, "eval"), exist_ok=True)
                truth = pd.read_parquet(os.path.join(cwd, "..", "holdout", "test.parquet"))["target"]
                col = "pred" if self.eval_calls == 1 else "prediction"
                pd.DataFrame({col: truth}).to_parquet(
                    os.path.join(cwd, "eval", "predictions.parquet"), index=False)
                return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                    session_id=session_id, raw={})
            return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                               max_turns=max_turns, budget_usd=budget_usd,
                               allowed_tools=allowed_tools)

    final = _run(tmp_path, WrongColumnOnceBackend())
    assert final["success"] is True
    assert final["attempts"]["eval"] == 2
    assert final["attempts"]["model"] == 1             # Self-Loop, kein Umweg über model


def test_soft_eval_quality_fail_verdict_routes_to_model(tmp_path):
    backend = SoftBackend(judge_overrides={
        "eval": [{"passed": False, "form_error": False,
                  "feedback": "Güte unzureichend, Modell verbessern"}]})
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["attempts"]["model"] == 2             # Rücksprung lief über model
    assert final["attempts"]["eval"] == 2
    model_prompts = [c["prompt"] for c in backend.worker_calls if "Modeling" in c["prompt"]]
    assert "Güte unzureichend" in model_prompts[1]     # Feedback erreichte den model-Worker


# ---------------------------------------------------------------------------
# judge_error & Fabrikationssicherheit
# ---------------------------------------------------------------------------

def test_soft_missing_verdict_is_regular_gate_fail(tmp_path):
    backend = SoftBackend(judge_overrides={"prep": ["SILENT"]})
    final = _run(tmp_path, backend)
    assert final["success"] is True                    # 2. Prüfung ehrlich -> weiter
    assert final["attempts"]["prep"] == 2              # judge_error verbraucht Budget
    ev = [e for e in _read_events(tmp_path) if e["phase"] == "prep"][0]
    assert ev["gate_passed"] is False
    assert ev["gate_metrics"]["judge_error"] is True
    prep_prompts = [c["prompt"] for c in backend.worker_calls
                    if "Data Preparation" in c["prompt"]]
    assert "without a usable verdict" in prep_prompts[1]


def test_soft_corrupt_verdict_is_judge_error(tmp_path, monkeypatch):
    class CorruptVerdictBackend(SoftBackend):
        def run(self, prompt, *, cwd, **kwargs):
            if JUDGE_MARKER in prompt and "'model'" in prompt:
                self.judge_calls.append({"phase": "model", "prompt": prompt})
                os.makedirs(os.path.join(cwd, "gate"), exist_ok=True)
                open(os.path.join(cwd, "gate", "verdict.json"), "w").write("kein json{{")
                return WorkerResult(text="?", usage=TokenUsage(5, 1), latency_ms=1.0,
                                    session_id=kwargs["session_id"], raw={})
            return super().run(prompt, cwd=cwd, **kwargs)

    final = _run(tmp_path, CorruptVerdictBackend())
    assert final["success"] is False
    assert final["attempts"]["model"] == 3             # bis max_retries, dann stop
    ev = [e for e in _read_events(tmp_path) if e["phase"] == "model"][0]
    assert ev["gate_metrics"]["judge_error"] is True
    assert "not valid JSON" in ev["gate_feedback"]


def test_soft_verdict_without_bool_passed_is_judge_error(tmp_path):
    backend = SoftBackend(judge_overrides={"prep": [{"passed": "ja", "feedback": "x"}]})
    final = _run(tmp_path, backend)
    assert final["attempts"]["prep"] == 2
    ev = [e for e in _read_events(tmp_path) if e["phase"] == "prep"][0]
    assert ev["gate_metrics"]["judge_error"] is True


def test_soft_judged_pass_with_unscorable_predictions_is_not_success(tmp_path):
    """Fabrikationssicherheit: Der Prüfer winkt kaputte Predictions durch —
    success bleibt False, weil die Harness-Messung nicht scoren kann."""
    class AlwaysWrongColumnBackend(SoftBackend):
        def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
            if JUDGE_MARKER not in prompt and "Evaluation" in prompt:
                os.makedirs(os.path.join(cwd, "eval"), exist_ok=True)
                truth = pd.read_parquet(os.path.join(cwd, "..", "holdout", "test.parquet"))["target"]
                pd.DataFrame({"pred": truth}).to_parquet(
                    os.path.join(cwd, "eval", "predictions.parquet"), index=False)
                return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                    session_id=session_id, raw={})
            return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                               max_turns=max_turns, budget_usd=budget_usd,
                               allowed_tools=allowed_tools)

    backend = AlwaysWrongColumnBackend(judge_overrides={
        "eval": {"passed": True, "feedback": "sieht gut aus"}})
    final = _run(tmp_path, backend)
    assert final["success"] is False
    assert final["final_f1"] is None
    ev = [e for e in _read_events(tmp_path) if e["phase"] == "eval"][0]
    assert ev["gate_metrics"]["scored_ok"] is False
    assert ev["gate_passed"] is True                   # Judge-Pass ist dokumentiert


def test_soft_preplanted_verdict_is_deleted_not_read(tmp_path):
    """Ein Worker, der gate/verdict.json vorpflanzt, kann das Gate nicht gamen:
    der Harness löscht die Datei vor dem Prüfer-Call."""
    class PlantingPrepBackend(SoftBackend):
        def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
            if JUDGE_MARKER not in prompt and "Data Preparation" in prompt:
                os.makedirs(os.path.join(cwd, "gate"), exist_ok=True)
                json.dump({"passed": True, "feedback": "gamed"},
                          open(os.path.join(cwd, "gate", "verdict.json"), "w"))
                # KEIN prepared.parquet — nur das geplante Verdict
                self.worker_calls.append({"prompt": prompt, "session_id": session_id,
                                          "resume": resume, "budget_usd": budget_usd})
                return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                    session_id=session_id, raw={})
            return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                               max_turns=max_turns, budget_usd=budget_usd,
                               allowed_tools=allowed_tools)

    backend = PlantingPrepBackend(judge_overrides={"prep": ["SILENT"]})
    final = _run(tmp_path, backend)
    ev = [e for e in _read_events(tmp_path) if e["phase"] == "prep"][0]
    # Wäre das geplante Verdict gelesen worden, stünde hier passed=True
    assert ev["gate_passed"] is False
    assert ev["gate_metrics"]["judge_error"] is True
    assert final["attempts"]["prep"] >= 2


def test_soft_verdict_file_cleaned_up_after_run(tmp_path):
    _run(tmp_path, SoftBackend())
    assert not os.path.exists(os.path.join(tmp_path, "work", "gate", "verdict.json"))


# ---------------------------------------------------------------------------
# Parität & Purity (roadmap §1: analog Arm D)
# ---------------------------------------------------------------------------

def test_soft_worker_prompts_identical_to_arm_a(tmp_path):
    """Wording-Parität: Die Worker sehen im Soft-Arm exakt dieselben Prompts wie
    in Arm A — das Treatment steckt AUSSCHLIESSLICH im Gate-Mechanismus."""
    backend = SoftBackend()
    task = "Bestimme die Klasse."                      # S2: Aufgabe in jedem Phasen-Prompt
    final = _run(tmp_path, backend, task_prompt=task)
    assert final["success"] is True
    cfg = _cfg(tmp_path, task_prompt=task)
    expected = [
        prompts.bu_initial_prompt(cfg.task_prompt),
        prompts.du_initial_prompt(cfg.target, None, task_prompt=task),
        prompts.initial_prompt("prep", cfg.target, du_report=final["du_report"],
                               task_prompt=task),
        prompts.initial_prompt("model", cfg.target, task_prompt=task),
        prompts.initial_prompt("eval", cfg.target, task_prompt=task),
    ]
    assert [c["prompt"] for c in backend.worker_calls] == expected
    for c in backend.worker_calls:                     # kein Prüfer-Vokabular
        assert JUDGE_MARKER not in c["prompt"]
        assert "verdict" not in c["prompt"].lower()


def test_soft_judge_call_parity(tmp_path):
    """Budget-/Tool-Parität: Prüfer-Calls laufen mit demselben Call-Budget,
    denselben max_turns und derselben Tool-Liste wie Worker-Calls — jeweils als
    frische Session (Prüfer akkumulieren keinen Kontext)."""
    backend = SoftBackend()
    _run(tmp_path, backend)
    cfg = _cfg(tmp_path)
    sessions = set()
    for c in backend.judge_calls:
        assert c["resume"] is False
        assert c["budget_usd"] == cfg.budget_usd_per_call
        assert c["max_turns"] == cfg.max_turns
        assert c["allowed_tools"] == softgates.JUDGE_ALLOWED_TOOLS
        sessions.add(c["session_id"])
    assert len(sessions) == len(backend.judge_calls)   # jede Prüfung frisch
    worker_sessions = {c["session_id"] for c in backend.worker_calls}
    assert sessions.isdisjoint(worker_sessions)


def test_judge_allowed_tools_match_worker_tools():
    assert softgates.JUDGE_ALLOWED_TOOLS == phases.ALLOWED_TOOLS


def _all_judge_prompts(cfg):
    ok_score = ScoreResult(True, f1=0.9, accuracy=0.91)
    return {phase: softgates.judge_prompt(phase, cfg,
                                          score=ok_score if phase == "eval" else None)
            for phase in ("bu", "du", "prep", "model", "eval")}


def test_judge_prompts_never_reference_labels_or_worker_keywords():
    """Purity: Prüfer-Prompts zeigen nie auf holdout/Test-Labels und enthalten
    keine Worker-Phasen-Keywords (Fakes/Log-Analyse routen darüber)."""
    cfg = RunConfig(run_id="x", workdir="w", arm="soft_fsm")
    for phase, p in _all_judge_prompts(cfg).items():
        assert "holdout" not in p, phase
        assert "test.parquet" not in p, phase          # test_features ist erlaubt
        for kw in WORKER_PHASE_KEYWORDS:
            assert kw not in p, (phase, kw)


def test_judge_prompts_mirror_gate_contracts():
    """Wording-Parität der Kriterien: Jeder Prüfer-Prompt beschreibt denselben
    Kontrakt, den das deterministische Gate in Arm A prüft."""
    cfg = RunConfig(run_id="x", workdir="w", arm="soft_fsm", target="col_7")
    p = _all_judge_prompts(cfg)
    assert "bu/task_spec.json" in p["bu"] and "cannot_determine" in p["bu"]
    assert "classification" in p["bu"] and "feature_columns" in p["bu"]
    assert "du/data_report.json" in p["du"] and "spec_mismatch" in p["du"]
    for field in ("n_rows", "columns", "class_balance", "excluded_columns"):
        assert field in p["du"]
    assert "col_7" in p["du"] and "col_7" in p["prep"]
    assert "prep/prepared.parquet" in p["prep"] and "NaN" in p["prep"]
    assert "numeric" in p["prep"]
    assert "model/model.pkl" in p["model"] and "dummy_score" in p["model"]
    assert "as reported in 'model/metrics.json'" in p["model"]      # S6
    assert "do not re-split or retrain" in p["model"]
    assert "eval/predictions.parquet" in p["eval"] and "'prediction'" in p["eval"]
    assert "data/test_features.parquet" in p["eval"]
    for prompt in p.values():
        assert "{" in prompt and "{{" not in prompt    # Braces sauber gerendert


def test_judge_eval_prompt_carries_measurement_and_threshold():
    cfg = RunConfig(run_id="x", workdir="w", arm="soft_fsm",
                    eval_metric="f1", eval_f1_threshold=0.5)
    p_ok = softgates.judge_prompt("eval", cfg, score=ScoreResult(True, f1=0.8765, accuracy=0.9))
    assert "f1=0.8765" in p_ok and "0.5" in p_ok
    p_err = softgates.judge_prompt("eval", cfg,
                                   score=ScoreResult(False, feedback="Spalte 'prediction' fehlt"))
    assert "Form error" in p_err and "Spalte 'prediction' fehlt" in p_err


def test_soft_arm_config_accepted_and_unknown_rejected(tmp_path):
    RunConfig(run_id="r", workdir=str(tmp_path), arm="soft_fsm")
    with pytest.raises(ValueError):
        RunConfig(run_id="r", workdir=str(tmp_path), arm="medium_fsm")


def test_judge_eval_prompt_carries_baseline_comparison():
    """Informations-Paritaet zu Arm A: der Pruefer bekommt dieselbe Messung wie
    das deterministische Gate (Baseline-F1, Differenz, 95-%-KI) + dieselbe Regel."""
    from crispfsm.scoring import BaselineComparison
    cfg = RunConfig(run_id="x", workdir="w", arm="soft_fsm", eval_metric="f1",
                    eval_f1_threshold=0.878, baseline_pred_path="holdout/b.parquet")
    cmp = BaselineComparison(f1=0.8765, baseline_f1=0.878, diff=-0.0015,
                             ci_lo=-0.0102, ci_hi=0.0071, noninferior=True)
    p = softgates.judge_prompt("eval", cfg, score=ScoreResult(True, f1=0.8765, accuracy=0.9),
                               comparison=cmp)
    assert "reference_baseline_f1=0.8780" in p and "[-0.0102, +0.0071]" in p
    assert "not be significantly below the reference baseline" in p
    assert "upper bound of the 95% confidence interval" in p and "must be ≥ 0" in p
    assert "at least 0.878" not in p            # alte Punkt-Regel nicht mehr im Kontrakt
