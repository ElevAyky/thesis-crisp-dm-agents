# tests/test_graph.py
import os
import pandas as pd
from crispfsm.config import RunConfig
from crispfsm.state import new_state
from crispfsm.graph import build_graph
from crispfsm.datasets import prepare_dataset
from crispfsm.backend import WorkerResult, TokenUsage


def _write_du_report(cwd):
    """Rechnet den DU-Report ehrlich aus data/train.parquet — check_du muss passen."""
    df = pd.read_parquet(f"{cwd}/data/train.parquet")
    target = df.columns[-1]
    os.makedirs(f"{cwd}/du", exist_ok=True)
    import json
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
    }, open(f"{cwd}/du/data_report.json", "w"))


class ScriptedBackend:
    """Erzeugt für jede Phase gültige Artefakte. Optional: lässt eval beim ersten
    Versuch durchfallen (zu niedrige F1), um den Rücksprung zu testen."""
    name = "scripted"
    def __init__(self, fail_eval_once=False):
        self.model = "m"; self.fail_eval_once = fail_eval_once; self.eval_calls = 0
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        if "Business Understanding" in prompt:
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            import json
            json.dump({"task_type": "classification", "target_column": "target",
                       "metric": "f1", "feature_columns": None},
                      open(f"{cwd}/bu/task_spec.json", "w"))
        elif "Data Understanding" in prompt:
            _write_du_report(cwd)
        elif "Data Preparation" in prompt:
            os.makedirs(f"{cwd}/prep", exist_ok=True)
            pd.DataFrame({"f1":[1,2,3,4],"target":[0,1,0,1]}).to_parquet(f"{cwd}/prep/prepared.parquet", index=False)
        elif "Modeling" in prompt:
            os.makedirs(f"{cwd}/model", exist_ok=True)
            import pickle
            from sklearn.dummy import DummyClassifier
            pickle.dump(DummyClassifier().fit([[0]],[0]), open(f"{cwd}/model/model.pkl","wb"))
            import json; json.dump({"metric":"f1","val_score":0.9,"dummy_score":0.6}, open(f"{cwd}/model/metrics.json","w"))
        elif "Evaluation" in prompt:
            self.eval_calls += 1
            os.makedirs(f"{cwd}/eval", exist_ok=True)
            truth = pd.read_parquet(f"{cwd}/../holdout/test.parquet")["target"]
            fail = self.fail_eval_once and self.eval_calls == 1
            preds = truth.map({0: 1, 1: 0}) if fail else truth
            pd.DataFrame({"prediction": preds}).to_parquet(
                f"{cwd}/eval/predictions.parquet", index=False)
        return WorkerResult(text="ok", usage=TokenUsage(10,2), latency_ms=1.0, session_id=session_id, raw={})

def _run(tmp_path, backend):
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    graph = build_graph(backend, cfg)
    final = graph.invoke(new_state(cfg), config={"recursion_limit": cfg.recursion_limit})
    return final

def test_graph_runs_end_to_end(tmp_path):
    final = _run(tmp_path, ScriptedBackend())
    assert final["success"] is True
    assert final["final_f1"] == 1.0
    assert final["visited_phases"] == ["bu", "du", "prep", "model", "eval"]

def test_graph_eval_failure_loops_back_to_model(tmp_path):
    backend = ScriptedBackend(fail_eval_once=True)
    final = _run(tmp_path, backend)
    assert final["success"] is True            # zweiter Eval-Versuch klappt
    assert backend.eval_calls == 2             # Rücksprung hat stattgefunden
    assert final["attempts"]["eval"] == 2
    assert final["attempts"]["model"] == 2     # Rücksprung lief WIRKLICH über Modeling


class BadPrepBackend:
    """Schreibt bei JEDEM prep-Call ein ungültiges Artefakt (NaN in Feature-Spalte),
    sodass check_prep IMMER durchfällt. Modeling/Evaluation werden gültig bedient,
    falls die FSM je so weit käme (was sie hier nicht tut)."""
    name = "bad_prep"
    def __init__(self):
        self.model = "m"
        self.prep_calls = 0
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        if "Business Understanding" in prompt:
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            import json
            json.dump({"task_type": "classification", "target_column": "target",
                       "metric": "f1", "feature_columns": None},
                      open(f"{cwd}/bu/task_spec.json", "w"))
        elif "Data Understanding" in prompt:
            _write_du_report(cwd)
        elif "Data Preparation" in prompt:
            self.prep_calls += 1
            os.makedirs(f"{cwd}/prep", exist_ok=True)
            # NaN in der Feature-Spalte -> check_prep findet nan_cols -> Gate failt
            pd.DataFrame({"f1": [1, None, 3, 4], "target": [0, 1, 0, 1]}).to_parquet(
                f"{cwd}/prep/prepared.parquet", index=False)
        elif "Modeling" in prompt:
            os.makedirs(f"{cwd}/model", exist_ok=True)
            import pickle
            from sklearn.dummy import DummyClassifier
            pickle.dump(DummyClassifier().fit([[0]], [0]), open(f"{cwd}/model/model.pkl", "wb"))
            import json
            json.dump({"metric": "f1", "val_score": 0.9, "dummy_score": 0.6}, open(f"{cwd}/model/metrics.json", "w"))
        elif "Evaluation" in prompt:
            os.makedirs(f"{cwd}/eval", exist_ok=True)
            truth = pd.read_parquet(f"{cwd}/../holdout/test.parquet")["target"]
            pd.DataFrame({"prediction": truth}).to_parquet(
                f"{cwd}/eval/predictions.parquet", index=False)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})


class FlakyPrepBackend:
    """Schreibt beim 1. prep-Call ein kaputtes Artefakt und ab dem 2. ein gültiges.
    Modeling/Evaluation danach gültig -> Run läuft am Ende erfolgreich durch."""
    name = "flaky_prep"
    def __init__(self):
        self.model = "m"
        self.prep_calls = 0
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        if "Business Understanding" in prompt:
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            import json
            json.dump({"task_type": "classification", "target_column": "target",
                       "metric": "f1", "feature_columns": None},
                      open(f"{cwd}/bu/task_spec.json", "w"))
        elif "Data Understanding" in prompt:
            _write_du_report(cwd)
        elif "Data Preparation" in prompt:
            self.prep_calls += 1
            os.makedirs(f"{cwd}/prep", exist_ok=True)
            if self.prep_calls == 1:
                # 1. Versuch: NaN -> Gate failt
                pd.DataFrame({"f1": [1, None, 3, 4], "target": [0, 1, 0, 1]}).to_parquet(
                    f"{cwd}/prep/prepared.parquet", index=False)
            else:
                # ab 2. Versuch: sauber -> Gate passt
                pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_parquet(
                    f"{cwd}/prep/prepared.parquet", index=False)
        elif "Modeling" in prompt:
            os.makedirs(f"{cwd}/model", exist_ok=True)
            import pickle
            from sklearn.dummy import DummyClassifier
            pickle.dump(DummyClassifier().fit([[0]], [0]), open(f"{cwd}/model/model.pkl", "wb"))
            import json
            json.dump({"metric": "f1", "val_score": 0.9, "dummy_score": 0.6}, open(f"{cwd}/model/metrics.json", "w"))
        elif "Evaluation" in prompt:
            os.makedirs(f"{cwd}/eval", exist_ok=True)
            truth = pd.read_parquet(f"{cwd}/../holdout/test.parquet")["target"]
            pd.DataFrame({"prediction": truth}).to_parquet(
                f"{cwd}/eval/predictions.parquet", index=False)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})


class FailEvalForeverBackend:
    """Gültiges prep + model, aber eval liefert dauerhaft eine f1 unter der Schwelle,
    sodass die FSM eval->model->eval-Schleife läuft, bis ein Limit greift."""
    name = "fail_eval_forever"
    def __init__(self):
        self.model = "m"
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        if "Business Understanding" in prompt:
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            import json
            json.dump({"task_type": "classification", "target_column": "target",
                       "metric": "f1", "feature_columns": None},
                      open(f"{cwd}/bu/task_spec.json", "w"))
        elif "Data Understanding" in prompt:
            _write_du_report(cwd)
        elif "Data Preparation" in prompt:
            os.makedirs(f"{cwd}/prep", exist_ok=True)
            pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_parquet(
                f"{cwd}/prep/prepared.parquet", index=False)
        elif "Modeling" in prompt:
            os.makedirs(f"{cwd}/model", exist_ok=True)
            import pickle
            from sklearn.dummy import DummyClassifier
            pickle.dump(DummyClassifier().fit([[0]], [0]), open(f"{cwd}/model/model.pkl", "wb"))
            import json
            json.dump({"metric": "f1", "val_score": 0.9, "dummy_score": 0.6}, open(f"{cwd}/model/metrics.json", "w"))
        elif "Evaluation" in prompt:
            os.makedirs(f"{cwd}/eval", exist_ok=True)
            # invertierte Labels -> F1 0.0, unter jeder Schwelle -> Gate failt immer
            truth = pd.read_parquet(f"{cwd}/../holdout/test.parquet")["target"]
            pd.DataFrame({"prediction": truth.map({0: 1, 1: 0})}).to_parquet(
                f"{cwd}/eval/predictions.parquet", index=False)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})


def test_graph_stops_after_max_retries_on_invalid_prep(tmp_path):
    """prep failt jedes Mal -> Router 'stop' bei attempts==max_retries.
    Lauf terminiert (kein Hang), bleibt erfolglos und kommt nie über prep hinaus."""
    backend = BadPrepBackend()
    final = _run(tmp_path, backend)
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path))
    assert final["success"] is False
    assert final["attempts"]["prep"] == cfg.max_retries == 3
    assert final["visited_phases"] == ["bu", "du", "prep"]
    assert backend.prep_calls == cfg.max_retries  # genau max_retries Worker-Calls


def test_graph_prep_retry_then_recover(tmp_path):
    """1. prep kaputt, 2. prep ok -> danach normaler Durchlauf bis eval-Success."""
    backend = FlakyPrepBackend()
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["final_f1"] == 1.0
    assert final["attempts"]["prep"] == 2          # ein Retry, dann durch
    assert final["visited_phases"] == ["bu", "du", "prep", "model", "eval"]
    assert backend.prep_calls == 2


def test_graph_safety_cap_halts_perpetual_eval_failure(tmp_path):
    """eval failt dauerhaft; mit kleinem safety_max_calls greift das Runaway-Notaus
    und der Lauf terminiert weit unter recursion_limit (kein Hang).

    Hinweis (echtes Verhalten): safety_max_calls ist ein WEICHER Cap. Der Router
    prüft 'total_calls >= safety_max_calls' nur an Retry-Entscheidungspunkten
    (failendes Gate). Zwischen zwei eval-Retries laufen aber auch model-Calls, die
    total_calls erhöhen, ohne dass dazwischen die Stop-Bedingung greift. Deshalb
    kann total_calls den Cap leicht überschießen — siehe Concern im Report."""
    cfg = RunConfig(run_id="r1", workdir=str(tmp_path), safety_max_calls=6)
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    graph = build_graph(FailEvalForeverBackend(), cfg)
    final = graph.invoke(new_state(cfg), config={"recursion_limit": cfg.recursion_limit})
    assert final["success"] is False
    # Terminiert nahe am Cap (kleiner Overshoot durch model-Calls im eval-Loop),
    # aber DEUTLICH unter dem recursion_limit-Backstop -> kein Runaway.
    assert cfg.safety_max_calls <= final["total_calls"] < cfg.recursion_limit
    assert final["total_calls"] <= cfg.safety_max_calls + 2
    # prep+model je 1x, dann eval/model-Loop bis das Notaus greift
    assert final["visited_phases"] == ["bu", "du", "prep", "model", "eval"]


class BrokenThenGoodEvalBackend(ScriptedBackend):
    """1. Eval-Call: Kontraktfehler (falsche Spalte 'pred' statt 'prediction'),
    ab dem 2.: Wahrheit. Zeichnet alle Prompts auf, um zu prüfen, dass das
    Gate-Feedback den EVAL-Worker erreicht (Self-Loop, kein Umweg über model)."""
    name = "broken_then_good_eval"

    def __init__(self):
        super().__init__()
        self.prompts = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self.prompts.append(prompt)
        if "Evaluation" in prompt:
            self.eval_calls += 1
            os.makedirs(f"{cwd}/eval", exist_ok=True)
            truth = pd.read_parquet(f"{cwd}/../holdout/test.parquet")["target"]
            col = "pred" if self.eval_calls == 1 else "prediction"
            pd.DataFrame({col: truth}).to_parquet(
                f"{cwd}/eval/predictions.parquet", index=False)
            return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def test_graph_eval_form_error_retries_eval_itself_not_model(tmp_path):
    """Kontraktfehler im Eval-Deliverable (Spalte falsch) -> Self-Loop zurück zu
    eval, NICHT Rücksprung über model: model kann den Fehler nicht fixen, und der
    Eval-Worker muss sein eigenes Gate-Feedback im Resume-Prompt sehen."""
    backend = BrokenThenGoodEvalBackend()
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["attempts"]["eval"] == 2
    assert final["attempts"]["model"] == 1     # KEIN Umweg über Modeling
    # Feedback erreichte den Eval-Worker: 2. Eval-Prompt ist ein Resume-Prompt
    # mit der konkreten Kontrakt-Diagnose (fehlende Spalte 'prediction').
    eval_prompts = [p for p in backend.prompts if "Evaluation" in p]
    assert len(eval_prompts) == 2
    assert "Quality Gate" in eval_prompts[1]
    assert "Column 'prediction' is missing" in eval_prompts[1]


class AbstainBackend(ScriptedBackend):
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        if "Business Understanding" in prompt:
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            import json
            json.dump({"cannot_determine": True, "reason": "Ziel existiert nicht", "missing_info": []},
                      open(f"{cwd}/bu/task_spec.json", "w"))
            from crispfsm.backend import WorkerResult, TokenUsage
            return WorkerResult(text="abstain", usage=TokenUsage(10, 2), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd, allowed_tools=allowed_tools)

class HallucinatingBuBackend(ScriptedBackend):
    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        if "Business Understanding" in prompt:
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            import json
            json.dump({"task_type": "classification", "target_column": "ghost", "metric": "f1"},
                      open(f"{cwd}/bu/task_spec.json", "w"))
            from crispfsm.backend import WorkerResult, TokenUsage
            return WorkerResult(text="ghost", usage=TokenUsage(10, 2), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd, allowed_tools=allowed_tools)

def test_graph_runs_bu_first(tmp_path):
    final = _run(tmp_path, ScriptedBackend())
    assert final["visited_phases"] == ["bu", "du", "prep", "model", "eval"]
    assert final["success"] is True
    assert final["bu_correct"] is True

def test_graph_abstain_terminates_without_prep(tmp_path):
    final = _run(tmp_path, AbstainBackend())
    assert final["abstained"] is True
    assert final["success"] is False
    assert final["visited_phases"] == ["bu"]
    assert final["attempts"]["bu"] == 1          # ABSTAIN retryt NICHT
    assert final["attempts"]["prep"] == 0

def test_graph_hallucinated_target_fails_after_retries(tmp_path):
    final = _run(tmp_path, HallucinatingBuBackend())
    assert final["success"] is False
    assert final["abstained"] is False
    assert final["attempts"]["bu"] == 3          # max_retries erschöpft -> FAIL
    assert final["attempts"]["prep"] == 0


class HallucinatingDuBackend(ScriptedBackend):
    """1. DU-Call halluziniert n_rows, ab dem 2. ehrlicher Report → Self-Retry
    mit Gate-Feedback im Resume-Prompt."""
    name = "hallucinating_du"

    def __init__(self):
        super().__init__()
        self.du_calls = 0
        self.prompts = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self.prompts.append(prompt)
        if "Data Understanding" in prompt:
            self.du_calls += 1
            if self.du_calls == 1:
                import json
                os.makedirs(f"{cwd}/du", exist_ok=True)
                json.dump({"n_rows": 999999, "columns": {}, "class_balance": {},
                           "quality_issues": [], "excluded_columns": []},
                          open(f"{cwd}/du/data_report.json", "w"))
            else:
                _write_du_report(cwd)
            return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def test_graph_du_hallucination_retries_du_with_feedback(tmp_path):
    backend = HallucinatingDuBackend()
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["attempts"]["du"] == 2
    assert final["attempts"]["bu"] == 1            # KEIN Umweg über bu
    du_prompts = [p for p in backend.prompts if "Data Understanding" in p]
    assert "Quality Gate" in du_prompts[1]
    assert "n_rows" in du_prompts[1]               # konkrete Faktentreue-Diagnose kam an


class SpecMismatchOnceBackend(ScriptedBackend):
    """1. DU-Call meldet spec_mismatch → Rücksprung zu bu (Loop-Engineering);
    ab dem 2. DU-Call ein ehrlicher Report → Lauf endet erfolgreich."""
    name = "spec_mismatch_once"

    def __init__(self):
        super().__init__()
        self.du_calls = 0
        self.du_prompts = []
        self.du_sessions = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        if "Data Understanding" in prompt:
            self.du_calls += 1
            self.du_prompts.append(prompt)
            self.du_sessions.append(session_id)
            if self.du_calls == 1:
                import json
                os.makedirs(f"{cwd}/du", exist_ok=True)
                json.dump({"spec_mismatch": True,
                           "spec_mismatch_reason": "Zielspalte widerspricht der Aufgabe"},
                          open(f"{cwd}/du/data_report.json", "w"))
            else:
                _write_du_report(cwd)
            return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def test_graph_du_spec_mismatch_loops_back_to_bu(tmp_path):
    backend = SpecMismatchOnceBackend()
    final = _run(tmp_path, backend)
    assert final["success"] is True
    assert final["attempts"]["bu"] == 2            # Rücksprung lief WIRKLICH über bu
    assert backend.du_calls == 2
    assert final["visited_phases"] == ["bu", "du", "prep", "model", "eval"]
    # spec_mismatch war KEIN Fehlversuch von DU: das Retry-Budget bleibt unberührt,
    # nur der erfolgreiche 2. Lauf zählt
    assert final["attempts"]["du"] == 1
    # 2. DU-Lauf startet frisch (neue Session, Initial-Prompt) — kein "Resume"
    # mit dem leeren Feedback des bestandenen bu-Gates
    assert backend.du_sessions[1] != backend.du_sessions[0]
    assert "did NOT pass" not in backend.du_prompts[1]


class IssueReportingDuBackend(ScriptedBackend):
    """DU meldet quality_issues + einen Spalten-Ausschluss; schneidet Prompts mit,
    um den Feed-Forward in den Prep-Prompt zu belegen."""
    name = "issue_reporting_du"

    def __init__(self):
        super().__init__()
        self.prompts = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        self.prompts.append(prompt)
        if "Data Understanding" in prompt:
            _write_du_report(cwd)
            import json
            path = f"{cwd}/du/data_report.json"
            report = json.load(open(path))
            report["quality_issues"] = ["Sentinel-Befund: Klassen leicht unbalanciert"]
            report["excluded_columns"] = [
                {"column": "mean radius", "reason": "Sentinel-Ausschluss"}]
            json.dump(report, open(path, "w"))
            return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def test_graph_du_findings_reach_prep_prompt(tmp_path):
    backend = IssueReportingDuBackend()
    final = _run(tmp_path, backend)
    assert final["success"] is True
    prep_prompts = [p for p in backend.prompts if "Data Preparation" in p]
    assert "Sentinel-Befund" in prep_prompts[0]
    assert "mean radius" in prep_prompts[0]
    # Report landet auch im finalen State (für summary.json)
    assert final["du_report"]["excluded_columns"][0]["column"] == "mean radius"


# --- S4: gemeinsames Run-Kontingent ------------------------------------------

def test_run_budget_stops_fsm_and_caps_calls(tmp_path):
    """Ein Worker, der pro Call 200 Turns verbraucht und nie ein gueltiges Artefakt
    schreibt: nach dem zweiten Call ist das Run-Kontingent (300) leer -> kein dritter
    Call, Router stoppt, budget_exhausted. Der zweite Call bekommt nur den Rest."""
    from crispfsm.backend import TokenUsage, WorkerResult
    from crispfsm.datasets import prepare_dataset

    class Hungry:
        name, model = "hungry", "m"
        def __init__(self):
            self.turn_caps = []
        def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
            self.turn_caps.append(max_turns)
            return WorkerResult(text="", usage=TokenUsage(1, 1), latency_ms=1.0,
                                session_id=session_id, raw={"num_turns": 200})

    cfg = RunConfig(run_id="r", workdir=str(tmp_path), max_turns=250)
    prepare_dataset(cfg.dataset, cfg.workdir, target=cfg.target)
    backend = Hungry()
    final = build_graph(backend, cfg).invoke(new_state(cfg),
                                             config={"recursion_limit": cfg.recursion_limit})
    assert backend.turn_caps == [250, 100]           # zweiter Call: min(250, 300 - 200)
    assert final["budget_exhausted"] is True and final["worker_turns"] == 400
    from crispfsm.run import fail_reason
    assert fail_reason(final, safety_max_calls=cfg.safety_max_calls) == "budget_exhausted"
