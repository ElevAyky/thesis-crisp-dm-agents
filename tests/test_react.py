"""Arm D (Naked ReAct, Spec 2026-07-18 §2): EIN Worker-Call, kein Retry, kein
Gate-Feedback. Outcome aus Deliverables: predictions.parquet + result.json."""
import json
import os

import pandas as pd

from crispfsm.backend import TokenUsage, WorkerResult
from crispfsm.config import RunConfig
from crispfsm.datasets import prepare_dataset
from crispfsm.react import build_react_graph
from crispfsm.state import new_state
from tests.test_prompts import FORBIDDEN_IN_REACT, _has_word


class ReactFake:
    """Konfigurierbarer Arm-D-Fake: schreibt predictions/result.json je nach Modus.
    Liest die Wahrheit aus ../holdout (Test-Cheat, echte Worker sehen das nie)."""
    name = "react_fake"
    model = "m"

    def __init__(self, mode="success"):
        # success | abstain | silent | no_result_json | wrong_target
        # | corrupt_result | array_result | wrong_labels | label_space_mismatch
        self.mode = mode
        self.calls = 0

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self.calls += 1
        self.last_prompt = prompt
        self.last_max_turns = max_turns
        self.last_budget_usd = budget_usd
        truth_df = pd.read_parquet(os.path.join(cwd, "..", "holdout", "test.parquet"))
        target = truth_df.columns[-1]
        result_path = os.path.join(cwd, "result.json")
        if self.mode == "abstain":
            json.dump({"cannot_determine": True, "reason": "kein Ziel erkennbar"},
                      open(result_path, "w"))
        elif self.mode != "silent":
            preds = truth_df[target]
            if self.mode == "wrong_labels":
                # valide Labels, aber zyklisch vertauscht -> F1 ~0 bei ok=True
                labels = sorted(preds.unique())
                shift = {a: labels[(i + 1) % len(labels)] for i, a in enumerate(labels)}
                preds = preds.map(shift)
            elif self.mode == "label_space_mismatch":
                # Fremd-Codes statt Klassen-Labels (Scan-Befund jm1): disjunkter
                # Label-Raum -> Scorer muss ok=False melden, nicht F1=0.0
                preds = pd.Series([f"X{c}" for c in pd.factorize(preds)[0]])
            pd.DataFrame({"prediction": preds}).to_parquet(
                os.path.join(cwd, "predictions.parquet"), index=False)
            if self.mode == "wrong_target":
                json.dump({"target_column": "ghost", "metric": "f1"},
                          open(result_path, "w"))
            elif self.mode == "corrupt_result":
                open(result_path, "w").write("kein json{{")
            elif self.mode == "array_result":
                json.dump([1, 2, 3], open(result_path, "w"))
            elif self.mode != "no_result_json":
                json.dump({"target_column": target, "metric": "f1"},
                          open(result_path, "w"))
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})


def _run(tmp_path, backend):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    graph = build_react_graph(backend, cfg)
    return graph.invoke(new_state(cfg), config={"recursion_limit": cfg.recursion_limit})


def test_react_success_scores_and_measures_target(tmp_path):
    backend = ReactFake("success")
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["final_f1"] == 1.0
    assert final["target_correct"] is True
    assert final["visited_phases"] == ["react"]
    assert final["attempts"]["react"] == 1
    assert backend.calls == 1                       # kein Retry (Spec §0 Nr. 6)
    assert final["usage_per_phase"]["react"]["input_tokens"] == 10


def test_react_workdir_has_features_but_never_labels(tmp_path):
    _run(tmp_path, ReactFake("success"))
    work = os.path.join(tmp_path, "work")
    assert os.path.exists(os.path.join(work, "data", "test_features.parquet"))
    assert not os.path.exists(os.path.join(work, "data", "test.parquet"))


def test_react_worker_prompt_is_pure_and_turn_budget_applied(tmp_path):
    backend = ReactFake("success")
    _run(tmp_path, backend)
    assert "predictions.parquet" in backend.last_prompt
    for word in FORBIDDEN_IN_REACT:                 # Review-Auflage Task 6
        assert not _has_word(word, backend.last_prompt), word
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    assert backend.last_max_turns == cfg.react_max_turns
    # Budget-Paritaet (S4): das gemeinsame Run-Kontingent, in einer Session
    assert backend.last_budget_usd == cfg.run_budget_usd == 20.0


def test_react_no_budget_cap_stays_none(tmp_path):
    """run_budget_usd=None (kein Cap) wird als None an die CLI weitergegeben."""
    backend = ReactFake("success")
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), run_budget_usd=None)
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    build_react_graph(backend, cfg).invoke(
        new_state(cfg), config={"recursion_limit": cfg.recursion_limit})
    assert backend.last_budget_usd is None


def test_react_low_f1_is_not_success(tmp_path):
    """Paritäts-Fix (Scan-Befund 08.08.): SUCCESS erfordert metric >= threshold
    wie im FSM-Eval-Gate — wohlgeformte, aber schlechte Predictions sind FAIL."""
    final = _run(tmp_path, ReactFake("wrong_labels"))
    assert final["success"] is False
    assert final["final_f1"] is None                # Parität zu phases.py (nur bei Pass)
    assert final["gate_metrics"]["f1"] < 0.5        # gemessen wurde trotzdem


def test_react_label_space_mismatch_is_form_error_not_zero_f1(tmp_path):
    """Scan-Befund jm1/bank_marketing: Int-Codes/Wahrscheinlichkeiten statt
    Klassen-Labels liefen still als F1=0.0 mit SUCCESS durch."""
    final = _run(tmp_path, ReactFake("label_space_mismatch"))
    assert final["success"] is False
    assert final["final_f1"] is None
    assert "target label space" in final["gate_feedback"]
    # Parität zu phases.py: Kontraktbruch ist als form_error markiert, damit
    # fail_reason (run.py) ihn von below_threshold trennen kann.
    assert final["gate_metrics"]["form_error"] is True
    assert final["best_f1_measured"] is None      # Formfehler = nichts gemessen


def test_react_low_f1_is_below_threshold_not_form_error(tmp_path):
    final = _run(tmp_path, ReactFake("wrong_labels"))
    assert final["success"] is False
    assert final["gate_metrics"]["form_error"] is False
    # gemessener F1 bleibt für die Verteilung erhalten, final_f1 nicht
    assert final["best_f1_measured"] == final["gate_metrics"]["f1"]
    assert final["final_f1"] is None


def test_react_abstain_wins_and_no_success(tmp_path):
    final = _run(tmp_path, ReactFake("abstain"))
    assert final["abstained"] is True
    assert final["success"] is False
    assert final["target_correct"] is None


def test_react_silent_is_fail(tmp_path):
    final = _run(tmp_path, ReactFake("silent"))
    assert final["success"] is False
    assert final["abstained"] is False
    assert final["final_f1"] is None


def test_react_missing_result_json_is_success_with_unknown_target(tmp_path):
    final = _run(tmp_path, ReactFake("no_result_json"))
    assert final["success"] is True
    assert final["target_correct"] is None          # Spec §4: fehlend => null


def test_react_wrong_target_is_measured_not_gated(tmp_path):
    final = _run(tmp_path, ReactFake("wrong_target"))
    assert final["success"] is True                 # Predictions zählen, kein Gate
    assert final["target_correct"] is False


def _read_events(tmp_path):
    return [json.loads(l) for l in
            open(os.path.join(tmp_path, "events.jsonl")).read().strip().splitlines()]


def test_react_logs_event_with_phase_react(tmp_path):
    _run(tmp_path, ReactFake("success"))
    events = _read_events(tmp_path)
    assert len(events) == 1
    assert events[0]["phase"] == "react"
    assert events[0]["gate_passed"] is True
    assert events[0]["result_json_error"] is None   # valides result.json => keine Diagnose


# --- Arm C: CRISP-ReAct (roadmap §1) — gleicher Node, nur der Prompt anders --


def _run_crisp(tmp_path, backend):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), arm="crisp_react")
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    graph = build_react_graph(backend, cfg)
    return graph.invoke(new_state(cfg), config={"recursion_limit": cfg.recursion_limit})


def test_crisp_react_success_scores_like_arm_d(tmp_path):
    backend = ReactFake("success")
    final = _run_crisp(tmp_path, backend)
    assert final["success"] is True
    assert final["final_f1"] == 1.0
    assert final["target_correct"] is True
    assert final["visited_phases"] == ["react"]
    assert backend.calls == 1                       # frei bleibt frei: kein Retry


def test_crisp_react_worker_gets_guidance_appended_to_d_prompt(tmp_path):
    """C↔D-Parität am echten Worker-Call: exakt der D-Prompt plus Anleitung."""
    from crispfsm import prompts
    backend = ReactFake("success")
    _run_crisp(tmp_path, backend)
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), arm="crisp_react")
    assert backend.last_prompt == prompts.crisp_react_prompt(cfg.task_prompt)
    assert backend.last_prompt.startswith(prompts.react_prompt(cfg.task_prompt))
    assert "CRISP-DM" in backend.last_prompt


def test_crisp_react_budget_and_turn_parity_match_arm_d(tmp_path):
    """Budget-/Turn-Parität identisch zu D (Spec §2.5) — C unterscheidet sich
    ausschließlich im Prompt."""
    from crispfsm.config import FSM_PHASES
    backend = ReactFake("success")
    _run_crisp(tmp_path, backend)
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), arm="crisp_react")
    assert backend.last_max_turns == cfg.react_max_turns
    assert backend.last_budget_usd == cfg.run_budget_usd


def test_crisp_react_workdir_has_features_but_never_labels(tmp_path):
    _run_crisp(tmp_path, ReactFake("success"))
    work = os.path.join(tmp_path, "work")
    assert os.path.exists(os.path.join(work, "data", "test_features.parquet"))
    assert not os.path.exists(os.path.join(work, "data", "test.parquet"))


def test_crisp_react_abstain_wins_like_arm_d(tmp_path):
    final = _run_crisp(tmp_path, ReactFake("abstain"))
    assert final["abstained"] is True
    assert final["success"] is False


def test_naked_react_prompt_stays_pure_after_crisp_arm_exists(tmp_path):
    """Regression: Der Arm-C-Zweig darf D nicht kontaminieren — D-Worker sehen
    weiterhin null CRISP-Vokabular."""
    backend = ReactFake("success")
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), arm="naked_react")
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    build_react_graph(backend, cfg).invoke(
        new_state(cfg), config={"recursion_limit": cfg.recursion_limit})
    for word in FORBIDDEN_IN_REACT:
        assert not _has_word(word, backend.last_prompt), word


def test_react_corrupt_result_json_is_diagnosed_not_crashed(tmp_path):
    """„Worker schrieb Müll" != „Worker schrieb nichts" (Fabrikation vs.
    Unterlassung) — Diagnose ins Event-Log, Run läuft weiter (Spec §4)."""
    final = _run(tmp_path, ReactFake("corrupt_result"))
    assert final["success"] is True                 # Predictions zählen
    assert final["target_correct"] is None          # unparsebar == fehlend
    events = _read_events(tmp_path)
    assert "kein valides JSON" in events[0]["result_json_error"]


def test_react_array_result_json_is_diagnosed_not_crashed(tmp_path):
    final = _run(tmp_path, ReactFake("array_result"))
    assert final["success"] is True
    assert final["target_correct"] is None
    events = _read_events(tmp_path)
    assert "kein JSON-Objekt" in events[0]["result_json_error"]


# --- Erfolgsregel gepaarter Bootstrap (23.09.) -------------------------------

def _run_vs_baseline(tmp_path, backend, baseline="perfect"):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    truth = pd.read_parquet(os.path.join(cfg.workdir, "holdout", "test.parquet"))["target"]
    base = truth.astype(str)
    path = os.path.join(cfg.workdir, "holdout", "baseline_predictions.parquet")
    pd.DataFrame({"prediction": base}).to_parquet(path, index=False)
    cfg.baseline_pred_path = path
    return build_react_graph(backend, cfg).invoke(
        new_state(cfg), config={"recursion_limit": cfg.recursion_limit})


def test_react_success_decided_by_paired_bootstrap(tmp_path):
    final = _run_vs_baseline(tmp_path, ReactFake("success"))
    assert final["success"] is True
    assert final["gate_metrics"]["noninferior"] is True
    assert final["gate_metrics"]["baseline_f1"] == 1.0


def test_react_worse_than_baseline_is_below_threshold(tmp_path):
    final = _run_vs_baseline(tmp_path, ReactFake("wrong_labels"))
    assert final["success"] is False
    assert final["gate_metrics"]["form_error"] is False
    assert final["gate_metrics"]["noninferior"] is False
    assert final["best_f1_measured"] == final["gate_metrics"]["f1"]
