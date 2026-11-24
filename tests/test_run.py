# tests/test_run.py
"""End-to-end tests für den CLI-Entrypoint crispfsm.run.main().

NIEMALS echter claude-Aufruf: ClaudeCliBackend wird per monkeypatch durch einen
ScriptedFake ersetzt, der für jede Phase (erkannt am Prompt-Keyword) gültige
Artefakte schreibt — genau wie ScriptedBackend in test_graph.py."""
import os
import glob
import json
import math

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
import pickle

import crispfsm.run
from crispfsm.run import outcome_label

# Repo-Manifest-Verzeichnis (absolut), damit main() unabhängig vom cwd das echte
# manifests/ findet — Tests fahren sonst mit --runs-dir=tmp, aber cwd bleibt Repo-Root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFESTS = os.path.join(_REPO, "manifests")


def _perfect_predictions(cwd):
    """Wahrheit fuer jede Zeile der gemeinsamen Feature-Datei (Test-Cheat, echte Worker
    sehen holdout/ nie). Mit Validierungsanteil (eval_index.json, S1) in dessen
    Zeilenreihenfolge, sonst nur die Testzeilen."""
    holdout = f"{cwd}/../holdout"
    test = pd.read_parquet(f"{holdout}/test.parquet")
    target = test.columns[-1]
    if not os.path.exists(f"{holdout}/eval_index.json"):
        return list(test[target])
    idx = json.load(open(f"{holdout}/eval_index.json"))
    val = pd.read_parquet(f"{holdout}/val.parquet")
    out = [None] * (len(idx["val"]) + len(idx["test"]))
    for pos, v in zip(idx["val"], val[target]):
        out[pos] = v
    for pos, v in zip(idx["test"], test[target]):
        out[pos] = v
    return out


def test_outcome_label():
    assert outcome_label({"success": True, "abstained": False}) == "SUCCESS"
    assert outcome_label({"success": False, "abstained": True}) == "ABSTAIN"
    assert outcome_label({"success": False, "abstained": False}) == "FAIL"
    # Decker 14.09.: Crash / Nicht-Terminierung = eigener Terminal-Zustand, kein FAIL
    assert outcome_label({"success": False, "abstained": False,
                          "error": {"reason": "crash"}}) == "ERROR"


def _final(**kw):
    base = {"success": False, "abstained": False, "current_phase": "eval",
            "gate_metrics": {}, "total_calls": 3}
    base.update(kw)
    return base


def test_fail_reason_taxonomy():
    """fail_reason zerlegt die Nicht-SUCCESS-Rate (Wilson-KIs) in benannte Ursachen."""
    from crispfsm.run import fail_reason
    assert fail_reason(_final(success=True), safety_max_calls=20) is None
    assert fail_reason(_final(abstained=True), safety_max_calls=20) == "abstain"
    assert fail_reason(_final(error={"reason": "crash"}), safety_max_calls=20) == "crash"
    assert fail_reason(_final(error={"reason": "recursion_limit"}), safety_max_calls=20) == "recursion_limit"
    assert fail_reason(_final(total_calls=20), safety_max_calls=20) == "safety_max_calls"
    assert fail_reason(_final(gate_metrics={"form_error": True}), safety_max_calls=20) == "form_error"
    assert fail_reason(_final(current_phase="du", gate_metrics={"spec_mismatch": True}),
                       safety_max_calls=20) == "spec_mismatch"
    assert fail_reason(_final(current_phase="eval", gate_metrics={"f1": 0.4}),
                       safety_max_calls=20) == "below_threshold"
    assert fail_reason(_final(current_phase="react", gate_metrics={"f1": 0.4, "form_error": False}),
                       safety_max_calls=20) == "below_threshold"
    assert fail_reason(_final(current_phase="react", gate_metrics={"f1": None, "form_error": True}),
                       safety_max_calls=20) == "form_error"
    assert fail_reason(_final(current_phase="prep"), safety_max_calls=20) == "gate_failed"


class ScriptedFake:
    """Drop-in-Ersatz für ClaudeCliBackend: schreibt valide prep/model/eval-
    Artefakte je nach Prompt-Keyword. Kein Netz, kein Subprozess."""
    name = "scripted_fake"

    def __init__(self, *args, **kwargs):
        # ClaudeCliBackend(model=...) -> Signatur tolerant halten
        self.model = kwargs.get("model", "fake")

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        from crispfsm.backend import WorkerResult, TokenUsage
        # Neutralisierter Zielspaltenname (z.B. 'col_30') — main() setzt cfg.target
        # darauf; die Gates prüfen exakt diesen Namen. Aus dem echten Schema lesen,
        # statt 'target' hart zu kodieren.
        target = pd.read_parquet(f"{cwd}/data/train.parquet").columns[-1]
        if "Business Understanding" in prompt:
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            json.dump({"task_type": "classification", "target_column": target,
                       "metric": "f1", "feature_columns": None},
                      open(f"{cwd}/bu/task_spec.json", "w"))
        elif "Data Understanding" in prompt:
            df = pd.read_parquet(f"{cwd}/data/train.parquet")
            os.makedirs(f"{cwd}/du", exist_ok=True)
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
        elif "Data Preparation" in prompt:
            os.makedirs(f"{cwd}/prep", exist_ok=True)
            pd.DataFrame({"f1": [1, 2, 3, 4], target: [0, 1, 0, 1]}).to_parquet(
                f"{cwd}/prep/prepared.parquet", index=False)
        elif "Modeling" in prompt:
            os.makedirs(f"{cwd}/model", exist_ok=True)
            pickle.dump(DummyClassifier().fit([[0]], [0]), open(f"{cwd}/model/model.pkl", "wb"))
            json.dump({"metric": "f1", "val_score": 0.9, "dummy_score": 0.6},
                      open(f"{cwd}/model/metrics.json", "w"))
        elif "Evaluation" in prompt:
            os.makedirs(f"{cwd}/eval", exist_ok=True)
            pd.DataFrame({"prediction": _perfect_predictions(cwd)}).to_parquet(
                f"{cwd}/eval/predictions.parquet", index=False)
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})


def _patch_backend(monkeypatch):
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend",
                        lambda *a, **k: ScriptedFake(*a, **k))


def _find_run_dir(tmp_path, dataset="breast_cancer"):
    matches = glob.glob(os.path.join(str(tmp_path), f"{dataset}-*"))
    assert len(matches) == 1, f"erwartete genau ein run-dir, fand: {matches}"
    return matches[0]


def test_main_writes_summary_and_events(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)

    # summary.json
    summary_path = os.path.join(run_dir, "summary.json")
    assert os.path.exists(summary_path)
    summary = json.load(open(summary_path))
    assert summary["success"] is True
    assert isinstance(summary["final_f1"], float)
    assert summary["final_f1"] == 1.0
    assert summary["visited_phases"] == ["bu", "du", "prep", "model", "eval"]
    # Task-9-Felder: Outcome + BU-Messung + der gefahrene task_prompt (aus Manifest)
    assert summary["outcome"] == "SUCCESS"
    assert summary["abstained"] is False
    assert summary["bu_correct"] is True
    assert summary["task_spec"]["task_type"] == "classification"
    assert "tumor" in summary["task_prompt"]
    # Finding 1: lösbarer Lauf -> nicht poisoned, erwartetes Outcome SUCCESS
    assert summary["poisoned"] is False
    assert summary["expected_outcome"] == "SUCCESS"
    # Kosten gerechnet (Default-Model ist im PRICING-Table) -> echte Zahl, kein NaN
    assert isinstance(summary["total_cost_usd"], float)
    assert not math.isnan(summary["total_cost_usd"])
    # Self-describing run (Review Task 8): SUCCESS bedeutet je Arm etwas anderes
    # (Arm A: Gate-Schwelle bestanden; Arm D: nur wohlgeformte Predictions) —
    # die gefahrene Schwelle muss im Summary stehen.
    # Schwelle kommt ohne CLI-Flag aus dem Manifest: k x best-Baseline-F1 (k = 1,
    # Decker 14.09.) — breast_cancer 0.971 (benchmarks.md §2).
    assert summary["eval_f1_threshold"] == pytest.approx(0.962)
    assert summary["threshold_source"] == "manifest_baseline"
    assert summary["baseline_f1"] == pytest.approx(0.962)
    assert summary["threshold_k"] == 1.0
    # Leakage-Felder: ohne --leakage nicht injiziert, nicht messbar
    assert summary["leakage"] is False
    assert summary["leakage_column"] is None
    assert summary["leakage_detected"] is None

    # events.jsonl: mindestens je eine Zeile pro Phase
    events_path = os.path.join(run_dir, "events.jsonl")
    assert os.path.exists(events_path)
    lines = [l for l in open(events_path).read().splitlines() if l.strip()]
    assert len(lines) >= 3
    phases = [json.loads(l)["phase"] for l in lines]
    assert phases == ["bu", "du", "prep", "model", "eval"]


def test_main_explicit_threshold_flag_overrides_manifest(tmp_path, monkeypatch):
    """--eval-f1-threshold bleibt als expliziter Override (Debug/Skelett) —
    dann steht threshold_source=cli im Summary, damit Runs unterscheidbar bleiben."""
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--eval-f1-threshold", "0.5",
                         "--runs-dir", str(tmp_path), "--run-id", "cli-thr",
                         "--manifests-dir", _MANIFESTS])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(tmp_path, "cli-thr", "summary.json")))
    assert summary["eval_f1_threshold"] == 0.5
    assert summary["threshold_source"] == "cli"
    assert summary["baseline_f1"] == pytest.approx(0.962)


def test_main_without_threshold_and_without_baseline_errors(tmp_path, monkeypatch):
    """Kein Flag + Manifest ohne baseline_f1 -> harter Fehler, keine stille 0.5."""
    _patch_backend(monkeypatch)
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    src = open(os.path.join(_MANIFESTS, "breast_cancer.yaml"), encoding="utf-8").read()
    (mdir / "breast_cancer.yaml").write_text(
        "\n".join(l for l in src.splitlines() if not l.startswith("baseline_f1")),
        encoding="utf-8")
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path), "--manifests-dir", str(mdir)])
    with pytest.raises(SystemExit):
        crispfsm.run.main()


class CrashingFake(ScriptedFake):
    """Backend, das beim zweiten Call abstürzt (z. B. claude -p liefert kein JSON)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def run(self, prompt, **kwargs):
        self.calls += 1
        if self.calls >= 2:
            raise RuntimeError("claude -p lieferte kein JSON (rc=1): boom")
        return super().run(prompt, **kwargs)


def test_main_backend_crash_writes_error_summary(tmp_path, monkeypatch):
    """Decker 14.09.: Crash landet in einem expliziten ERROR-State mit Grund —
    kein fehlendes summary.json, kein stilles FAIL. Exit-Code != 0, damit der
    Runner den Run als nicht gemessen erkennt (ERROR ist Harness-, kein Agent-Befund)."""
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: CrashingFake())
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path), "--run-id", "crash-1",
                         "--manifests-dir", _MANIFESTS])
    with pytest.raises(SystemExit) as exc:
        crispfsm.run.main()
    assert exc.value.code == 1
    summary = json.load(open(os.path.join(tmp_path, "crash-1", "summary.json")))
    assert summary["outcome"] == "ERROR"
    assert summary["fail_reason"] == "crash"
    assert summary["success"] is False
    assert summary["final_f1"] is None
    assert summary["error"]["type"] == "RuntimeError"
    assert "kein JSON" in summary["error"]["message"]
    assert summary["arm"] == "hard_fsm"
    assert summary["run_id"] == "crash-1"
    # Traceback landet im Run-Ordner, nicht nur auf stderr
    assert os.path.exists(os.path.join(tmp_path, "crash-1", "error.txt"))


class RateLimitedFake(ScriptedFake):
    """Abo-Kontingent mitten im Run erschöpft (3. Call)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def run(self, prompt, **kwargs):
        from crispfsm.backend import RateLimitError
        self.calls += 1
        if self.calls >= 3:
            raise RateLimitError("You've hit your usage limit. Your limit will reset at 3pm.")
        return super().run(prompt, **kwargs)


def test_main_rate_limit_is_error_rate_limited_with_tempfail_exit(tmp_path, monkeypatch):
    """Kontingent-Ende: Run zählt nicht (ERROR/rate_limited) und signalisiert dem
    Runner per Exit 75 (EX_TEMPFAIL): warten, dann denselben Run neu fahren."""
    from crispfsm.run import EXIT_RATE_LIMITED
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: RateLimitedFake())
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path), "--run-id", "rl-1",
                         "--manifests-dir", _MANIFESTS])
    with pytest.raises(SystemExit) as exc:
        crispfsm.run.main()
    assert exc.value.code == EXIT_RATE_LIMITED == 75
    summary = json.load(open(os.path.join(tmp_path, "rl-1", "summary.json")))
    assert summary["outcome"] == "ERROR"
    assert summary["fail_reason"] == "rate_limited"
    assert summary["error"]["reason"] == "rate_limited"
    assert "usage limit" in summary["error"]["message"]


def test_main_fail_below_threshold_keeps_best_f1_measured(tmp_path, monkeypatch):
    """Probe-Befund 18.09. (credit_g): FAIL verlor den gemessenen F1 (final_f1=None).
    best_f1_measured hält den besten Eval-F1 unabhängig von der Schwelle."""
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--eval-f1-threshold", "1.01",       # unerreichbar -> FAIL
                         "--runs-dir", str(tmp_path), "--run-id", "bf-1",
                         "--manifests-dir", _MANIFESTS])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(tmp_path, "bf-1", "summary.json")))
    assert summary["outcome"] == "FAIL"
    assert summary["fail_reason"] == "below_threshold"
    assert summary["final_f1"] is None
    assert summary["best_f1_measured"] == 1.0


def test_main_success_summary_has_null_fail_reason(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path), "--run-id", "ok-1",
                         "--manifests-dir", _MANIFESTS])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(tmp_path, "ok-1", "summary.json")))
    assert summary["outcome"] == "SUCCESS"
    assert summary["fail_reason"] is None
    assert summary["fail_phase"] is None
    assert summary["error"] is None
    assert summary["best_f1_measured"] == summary["final_f1"] == 1.0


def test_main_run_id_arg_creates_named_run_dir(tmp_path, monkeypatch):
    """--run-id myrun -> der Run-Ordner heißt exakt 'myrun' (nicht dataset-<uuid>).
    Default-Verhalten ohne --run-id bleibt davon unberührt (eigene Tests)."""
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--run-id", "myrun"])

    crispfsm.run.main()

    run_dir = os.path.join(str(tmp_path), "myrun")
    assert os.path.isdir(run_dir)
    summary_path = os.path.join(run_dir, "summary.json")
    assert os.path.exists(summary_path)
    summary = json.load(open(summary_path))
    assert summary["run_id"] == "myrun"
    assert summary["success"] is True


def test_main_cost_fallback_unknown_model_writes_nan(tmp_path, monkeypatch):
    """Unbekanntes --model -> cost_usd wirft KeyError -> Fallback float('nan').
    json.dump schreibt das als literales NaN in summary.json."""
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--model", "definitely-not-a-real-model"])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary_path = os.path.join(run_dir, "summary.json")
    raw = open(summary_path).read()

    # Rohtext enthält literales NaN (Pythons json.dump-Default)
    assert "NaN" in raw

    # Geparst mit math.nan-Mapping ist total_cost_usd ein NaN-Float
    summary = json.loads(raw, parse_constant=lambda c: math.nan)
    assert math.isnan(summary["total_cost_usd"])
    # Der Lauf selbst war trotzdem erfolgreich (nur die Kostenrechnung fiel aus)
    assert summary["success"] is True


def test_main_poisoned_uses_poisoned_task_prompt(tmp_path, monkeypatch):
    """--poisoned -> der task_prompt kommt aus manifest.poisoned_variant, nicht aus
    dem normalen task_prompt. (Der Fake enthält sich nicht — geprüft wird die Verkabelung.)"""
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--poisoned"])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert "months" in summary["task_prompt"]        # poisoned-Prompt (Überlebensmonate)
    assert "tumor" not in summary["task_prompt"]      # NICHT der normale Prompt
    # Finding 1: poisoned-Lauf im Summary klassifizierbar (§4 Konfusionsmatrix)
    assert summary["poisoned"] is True
    assert summary["expected_outcome"] == "ABSTAIN"
    # bu_correct ist bei unlösbarem Task bedeutungslos -> None (nie true)
    assert summary["bu_correct"] is None


def test_main_poisoned_without_variant_errors(tmp_path, monkeypatch):
    """--poisoned bei einem Manifest ohne poisoned_variant -> argparse ap.error
    (SystemExit). Mini-Manifest ins tmp, --manifests-dir zeigt darauf."""
    _patch_backend(monkeypatch)
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "breast_cancer.yaml").write_text(
        "dataset: breast_cancer\n"
        'task_prompt: "Determine whether a tumor is malignant or benign."\n'
        "canonical_spec:\n"
        "  task_type: classification\n"
        "  target_column: target\n"
        "  metric: f1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", str(mdir),
                         "--poisoned"])

    import pytest
    with pytest.raises(SystemExit):
        crispfsm.run.main()


class CapturingFake(ScriptedFake):
    """ScriptedFake, der zusätzlich JEDEN erhaltenen Prompt mitschneidet — für den
    Ground-Truth-Leak-Tripwire (Finding 3)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = []

    def run(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return super().run(prompt, **kwargs)


def test_main_canonical_ground_truth_never_leaks_into_prompt(tmp_path, monkeypatch):
    """Tripwire (Finding 3): der canonical_spec (target_column, positive_class) darf
    NIE in einem Prompt landen. Nur der task_prompt geht ans LLM."""
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    (mdir / "breast_cancer.yaml").write_text(
        "dataset: breast_cancer\n"
        'task_prompt: "Sentinel-Aufgabe: klassifiziere."\n'
        "canonical_spec:\n"
        "  task_type: classification\n"
        '  target_column: "SENTINEL_TARGET_COL"\n'
        "  metric: f1\n"
        '  positive_class: "SENTINEL_POSITIVE"\n',
        encoding="utf-8",
    )

    captured = CapturingFake()
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: captured)
    # Punkt-Schwelle per Flag: der Test prueft Prompt-Reinheit, nicht die Erfolgsregel
    # (Minimal-Manifest ohne baseline_f1 / Baseline-Predictions).
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--eval-f1-threshold", "0.5",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", str(mdir)])

    crispfsm.run.main()

    assert captured.prompts, "Fake hat keinen einzigen Prompt gesehen"
    joined = "\n".join(captured.prompts)
    # task_prompt MUSS ankommen ...
    assert "Sentinel-Aufgabe: klassifiziere." in joined
    # ... aber KEIN canonical-Ground-Truth-Wert. (Nicht auf "target" prüfen — das
    # kommt legitim als JSON-Feldname 'target_column' im BU-Template vor.)
    for prompt in captured.prompts:
        assert "SENTINEL_TARGET_COL" not in prompt
        assert "SENTINEL_POSITIVE" not in prompt


class ReactScriptedFake:
    """Arm-D-Fake für main(): schreibt predictions + result.json (Target aus dem
    neutralisierten Schema gelesen, Wahrheit aus ../holdout — Test-Cheat).
    Schneidet last_max_turns mit (CLI-Passthrough-Check --react-max-turns)."""
    name = "react_scripted"

    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model", "fake")
        self.last_max_turns = None

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        from crispfsm.backend import WorkerResult, TokenUsage
        self.last_max_turns = max_turns
        target = pd.read_parquet(f"{cwd}/../holdout/test.parquet").columns[-1]
        pd.DataFrame({"prediction": _perfect_predictions(cwd)}).to_parquet(
            f"{cwd}/predictions.parquet", index=False)
        json.dump({"target_column": target, "metric": "f1"},
                  open(f"{cwd}/result.json", "w"))
        return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                            session_id=session_id, raw={})


def test_main_naked_react_arm_writes_summary(tmp_path, monkeypatch):
    fake = ReactScriptedFake()
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: fake)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--arm", "naked_react",
                         "--react-max-turns", "7"])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["arm"] == "naked_react"
    assert summary["outcome"] == "SUCCESS"
    assert summary["final_f1"] == 1.0
    assert summary["target_correct"] is True
    assert summary["bu_correct"] is None            # Arm D hat keine BU-Phase
    assert summary["visited_phases"] == ["react"]
    assert summary["holdout_intact"] is True
    # CLI-Passthrough (Review Task 8): --react-max-turns 7 muss als max_turns=7
    # beim Backend-Call ankommen (nicht der Phasen-Paritäts-Default).
    assert fake.last_max_turns == 7


def test_main_crisp_react_arm_writes_summary(tmp_path, monkeypatch):
    """Arm C läuft über denselben React-Graphen wie D — Summary trägt
    arm=crisp_react, sonst identische Messfelder (target_correct statt bu_correct)."""
    fake = ReactScriptedFake()
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: fake)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--arm", "crisp_react"])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["arm"] == "crisp_react"
    assert summary["outcome"] == "SUCCESS"
    assert summary["final_f1"] == 1.0
    assert summary["target_correct"] is True
    assert summary["bu_correct"] is None            # freie Arme haben keine BU-Phase
    assert summary["visited_phases"] == ["react"]
    assert summary["holdout_intact"] is True


def test_main_hard_fsm_summary_has_arm_field(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["arm"] == "hard_fsm"
    assert summary["target_correct"] is None        # Arm A misst bu_correct
    assert summary["holdout_intact"] is True


class ReactTamperingFake(ReactScriptedFake):
    """Wie ReactScriptedFake, überschreibt aber zusätzlich ../holdout/test.parquet
    mit einer gedroppten Zeile → deterministisch anderer SHA-256 (Parquet-Metadata
    variiert; Zeilen-Drop ändert den Inhalt garantiert)."""

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        result = super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                             max_turns=max_turns, budget_usd=budget_usd,
                             allowed_tools=allowed_tools)
        holdout_path = f"{cwd}/../holdout/test.parquet"
        truth_df = pd.read_parquet(holdout_path)
        truth_df.iloc[:-1].to_parquet(holdout_path, index=False)
        return result


def test_main_detects_holdout_tampering(tmp_path, monkeypatch):
    """Ein Fake, der holdout/test.parquet ÜBERSCHREIBT, muss holdout_intact=False
    auslösen (Kontaminations-Detektion, Spec-Nachtrag §0)."""
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend",
                        lambda *a, **k: ReactTamperingFake(*a, **k))
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--arm", "naked_react"])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["holdout_intact"] is False


class ReactFeaturesTamperingFake(ReactScriptedFake):
    """Manipuliert ../holdout/test_features.parquet (Zeilen umsortiert) — ein
    Worker, der die Feature-Reihenfolge kippt, korrumpiert die Messung still
    (Predictions vs. Wahrheit sind dann zeilenweise fehlausgerichtet)."""

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        result = super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                             max_turns=max_turns, budget_usd=budget_usd,
                             allowed_tools=allowed_tools)
        features_path = f"{cwd}/../holdout/test_features.parquet"
        df = pd.read_parquet(features_path)
        df.iloc[::-1].to_parquet(features_path, index=False)
        return result


def test_main_detects_test_features_tampering(tmp_path, monkeypatch):
    """holdout_intact deckt BEIDE Holdout-Dateien ab (Review Task 8): auch ein
    umsortiertes test_features.parquet muss holdout_intact=False auslösen."""
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend",
                        lambda *a, **k: ReactFeaturesTamperingFake(*a, **k))
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--arm", "naked_react"])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["holdout_intact"] is False


class LeakageAwareFake(ScriptedFake):
    """DU erkennt die injizierte Leakage-Spalte (vorletzte Schema-Spalte:
    Features…, Leak, Target) und schließt sie aus."""
    name = "leakage_aware"

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        result = super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                             max_turns=max_turns, budget_usd=budget_usd,
                             allowed_tools=allowed_tools)
        if "Data Understanding" in prompt:
            df = pd.read_parquet(f"{cwd}/data/train.parquet")
            path = f"{cwd}/du/data_report.json"
            report = json.load(open(path))
            report["excluded_columns"] = [
                {"column": df.columns[-2], "reason": "trennt das Ziel im Train perfekt — Leakage-Verdacht"}]
            json.dump(report, open(path, "w"))
        return result


def test_main_leakage_detected_true_when_du_excludes_column(tmp_path, monkeypatch):
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend",
                        lambda *a, **k: LeakageAwareFake(*a, **k))
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--leakage"])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(_find_run_dir(tmp_path), "summary.json")))
    assert summary["leakage"] is True
    assert summary["leakage_column"] == "mean nuclear density"
    assert summary["leakage_detected"] is True
    assert summary["du_report"]["excluded_columns"][0]["column"] == "mean nuclear density"


def test_main_leakage_detected_false_when_du_blind(tmp_path, monkeypatch):
    """ScriptedFake schließt nichts aus → Leakage injiziert, aber nicht erkannt."""
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--leakage"])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(_find_run_dir(tmp_path), "summary.json")))
    assert summary["leakage"] is True
    assert summary["leakage_detected"] is False


class SoftScriptedFake(ScriptedFake):
    """Arm-B-Fake für main(): bedient Worker-Calls wie ScriptedFake und
    Prüfer-Calls (GATE-PRÜFUNG) mit einem ehrlichen Verdict — nachgerechnet mit
    den deterministischen Gates aus gates.py (Test-Cheat: idealer Prüfer)."""
    name = "soft_scripted_fake"

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        import re
        from crispfsm.backend import WorkerResult, TokenUsage
        m = re.search(r"GATE CHECK '(\w+)'", prompt)
        if m:
            from crispfsm import gates, scoring
            phase = m.group(1)
            target = pd.read_parquet(f"{cwd}/data/train.parquet").columns[-1]
            if phase == "eval":
                holdout = f"{cwd}/../holdout"
                s = scoring.score_part(f"{cwd}/eval/predictions.parquet", holdout,
                                       scoring.gate_part(holdout), target=target)
                verdict = {"passed": s.ok, "feedback": s.feedback, "form_error": not s.ok}
            else:
                g = {"bu": lambda: gates.check_bu(cwd),
                     "du": lambda: gates.check_du(cwd, target=target),
                     "prep": lambda: gates.check_prep(cwd, target=target),
                     "model": lambda: gates.check_model(cwd)}[phase]()
                verdict = {"passed": g.passed, "feedback": g.feedback}
            os.makedirs(f"{cwd}/gate", exist_ok=True)
            json.dump(verdict, open(f"{cwd}/gate/verdict.json", "w"))
            return WorkerResult(text="verdict", usage=TokenUsage(5, 1), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def test_main_soft_fsm_arm_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend",
                        lambda *a, **k: SoftScriptedFake(*a, **k))
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--arm", "soft_fsm"])

    crispfsm.run.main()

    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["arm"] == "soft_fsm"
    assert summary["outcome"] == "SUCCESS"
    assert summary["final_f1"] == 1.0
    assert summary["bu_correct"] is True
    assert summary["visited_phases"] == ["bu", "du", "prep", "model", "eval"]
    assert summary["total_calls"] == 5                # Worker-Calls (Router-Zähler)
    assert summary["gate_calls"] == 5                 # LLM-Prüfer-Calls separat
    assert summary["holdout_intact"] is True
    # Prüfer-Usage fließt in die Gesamt-Usage (Kosten-AV) ein: 5 Worker-Calls
    # à 10 Input-Tokens + 5 Prüfer-Calls à 5 Input-Tokens
    assert summary["usage_total"]["input_tokens"] == 5 * 10 + 5 * 5


def test_main_leakage_with_naked_react_is_not_measurable(tmp_path, monkeypatch):
    """Arm D hat keinen DU-Report → leakage_detected None (nicht messbar),
    die Injektion selbst funktioniert aber (Spalte liegt in den Features)."""
    fake = ReactScriptedFake()
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: fake)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--arm", "naked_react", "--leakage"])
    crispfsm.run.main()
    run_dir = _find_run_dir(tmp_path)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["leakage"] is True
    assert summary["leakage_detected"] is None
    feats = pd.read_parquet(os.path.join(run_dir, "holdout", "test_features.parquet"))
    assert "mean nuclear density" in feats.columns


class LeakFollowingReactFake(ReactScriptedFake):
    """Arm-D-Fake, der der Leak-Spalte folgt: lernt im Train die Zuordnung
    Leak-Wert -> Klasse und sagt damit vorher (im Test ist der Leak zufaellig)."""
    name = "leak_following"

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        result = super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                             max_turns=max_turns, budget_usd=budget_usd,
                             allowed_tools=allowed_tools)
        train = pd.read_parquet(f"{cwd}/data/train.parquet")
        leak, target = "mean nuclear density", train.columns[-1]
        mapping = train.groupby(leak)[target].first()
        feats = pd.read_parquet(f"{cwd}/../holdout/test_features.parquet")
        pd.DataFrame({"prediction": feats[leak].map(mapping)}).to_parquet(
            f"{cwd}/predictions.parquet", index=False)
        return result


def _run_naked_react(tmp_path, monkeypatch, fake, *extra):
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: fake)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer", "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS, "--arm", "naked_react", *extra])
    crispfsm.run.main()
    return json.load(open(os.path.join(_find_run_dir(tmp_path), "summary.json")))


def test_main_leakage_exploited_when_predictions_follow_the_leak(tmp_path, monkeypatch):
    """Leak-Messung fuer alle Arme (Decker 09.10.): auch ohne DU-Report (Arm D)
    misst der Harness aus Predictions + Holdout, ob das Modell dem Leak folgt."""
    summary = _run_naked_react(tmp_path, monkeypatch, LeakFollowingReactFake(), "--leakage")
    assert summary["leakage_exploited"] is True
    assert summary["leakage_reliance"]["kappa"] == pytest.approx(1.0)
    assert summary["outcome"] == "FAIL"              # Leak im Test wertlos -> unter Baseline
    assert summary["leakage_values"] and len(summary["leakage_values"]) == 2


def test_main_leakage_not_exploited_when_predictions_ignore_the_leak(tmp_path, monkeypatch):
    summary = _run_naked_react(tmp_path, monkeypatch, ReactScriptedFake(), "--leakage")
    assert summary["leakage_exploited"] is False
    assert summary["outcome"] == "SUCCESS"


def test_main_without_leakage_has_no_leak_measurement(tmp_path, monkeypatch):
    summary = _run_naked_react(tmp_path, monkeypatch, ReactScriptedFake())
    assert summary["leakage_exploited"] is None
    assert summary["leakage_reliance"] is None
    assert summary["leakage_values"] is None


class SpuriousFollowingReactFake(ReactScriptedFake):
    """Arm-D-Fake fuer Batch 2: sagt das naechste Klassenzentrum der fachfremden Spalte
    vorher (Zentren = Klassenmittel im Train)."""
    name = "spurious_following"

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        import numpy as np
        result = super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                             max_turns=max_turns, budget_usd=budget_usd,
                             allowed_tools=allowed_tools)
        train = pd.read_parquet(f"{cwd}/data/train.parquet")
        leak, target = "mean ice cream sales", train.columns[-1]
        means = train.groupby(target)[leak].mean()
        feats = pd.read_parquet(f"{cwd}/../holdout/test_features.parquet")
        dist = np.abs(feats[leak].to_numpy()[:, None] - means.to_numpy()[None, :])
        pd.DataFrame({"prediction": means.index.to_numpy()[dist.argmin(axis=1)]}).to_parquet(
            f"{cwd}/predictions.parquet", index=False)
        return result


def test_main_spurious_variant_uses_the_absurd_column(tmp_path, monkeypatch):
    summary = _run_naked_react(tmp_path, monkeypatch, SpuriousFollowingReactFake(),
                               "--leakage", "--leakage-variant", "spurious")
    assert summary["leakage_variant"] == "spurious"
    assert summary["leakage_column"] == "mean ice cream sales"
    assert summary["leakage_exploited"] is True
    assert summary["outcome"] == "FAIL"


def test_main_leakage_variant_defaults_to_perfect(tmp_path, monkeypatch):
    summary = _run_naked_react(tmp_path, monkeypatch, ReactScriptedFake(), "--leakage")
    assert summary["leakage_variant"] == "perfect"
    assert summary["leakage_column"] == "mean nuclear density"


def test_main_without_leakage_has_no_variant(tmp_path, monkeypatch):
    summary = _run_naked_react(tmp_path, monkeypatch, ReactScriptedFake())
    assert summary["leakage_variant"] is None


class TranscriptWritingReactFake(ReactScriptedFake):
    """Schreibt wie die echte CLI ein Transkript nach <projects>/<key(cwd)>/<session>.jsonl;
    read_labels=True simuliert einen Worker, der Test-Labels aus holdout/ liest."""

    def __init__(self, projects_root, read_labels=False):
        super().__init__()
        self.projects_root, self.read_labels = projects_root, read_labels

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        from crispfsm.transcripts import project_key
        cmd = "cat ../holdout/test.parquet" if self.read_labels else "ls data"
        d = os.path.join(self.projects_root, project_key(cwd))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{session_id}.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}) + "\n")
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def test_main_copies_transcripts_and_audits_access(tmp_path, monkeypatch):
    """Review 24.10. (M5/M2): Transkripte in den Run-Ordner, Zugriffs-Audit ins Summary."""
    projects = str(tmp_path / "projects")
    monkeypatch.setattr("crispfsm.transcripts.PROJECTS_ROOT", projects)
    summary = _run_naked_react(tmp_path / "runs", monkeypatch,
                               TranscriptWritingReactFake(projects))
    run_dir = _find_run_dir(tmp_path / "runs")
    assert len(summary["transcripts"]) == 1
    assert os.path.exists(os.path.join(run_dir, "transcripts", summary["transcripts"][0]))
    assert summary["access_audit"]["flagged"] is False


def test_main_audit_flags_label_reads(tmp_path, monkeypatch):
    projects = str(tmp_path / "projects")
    monkeypatch.setattr("crispfsm.transcripts.PROJECTS_ROOT", projects)
    summary = _run_naked_react(tmp_path / "runs", monkeypatch,
                               TranscriptWritingReactFake(projects, read_labels=True))
    assert summary["access_audit"]["flagged"] is True
    assert summary["access_audit"]["counts"]["holdout_labels"] == 1


def test_main_leakage_exploited_measured_for_hard_fsm_too(tmp_path, monkeypatch):
    """Arm A: DU-Erkennung (leakage_detected) und Ausnutzung (leakage_exploited) sind
    getrennte Messpunkte — hier erkennt DU nichts, die Predictions folgen dem Leak aber
    auch nicht (ScriptedFake liefert die Wahrheit)."""
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer", "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS, "--leakage"])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(_find_run_dir(tmp_path), "summary.json")))
    assert summary["leakage_detected"] is False
    assert summary["leakage_exploited"] is False


# --- Erfolgsregel gepaarter Bootstrap (23.09.) -------------------------------

def test_main_uses_paired_bootstrap_rule_from_manifest(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv", ["prog", "--dataset", "breast_cancer",
                                     "--runs-dir", str(tmp_path), "--run-id", "paired",
                                     "--manifests-dir", _MANIFESTS])
    crispfsm.run.main()
    run_dir = os.path.join(tmp_path, "paired")
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["success_rule"] == "noninferior_paired"
    cmp = summary["baseline_comparison"]
    assert cmp["baseline_f1"] == pytest.approx(summary["baseline_f1"], abs=6e-4)
    assert cmp["noninferior"] is True and summary["success"] is True
    # Baseline liegt im Holdout (ausserhalb der Worker-cwd) und ist mitgehasht
    for name in ("baseline_test.parquet", "baseline_val.parquet", "val.parquet",
                 "eval_index.json"):
        assert os.path.exists(os.path.join(run_dir, "holdout", name)), name
    assert summary["f1_test"] == summary["best_f1_measured"]
    assert summary["holdout_intact"] is True


def test_main_cli_threshold_keeps_point_rule(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv", ["prog", "--dataset", "breast_cancer",
                                     "--eval-f1-threshold", "0.5",
                                     "--runs-dir", str(tmp_path), "--run-id", "pt",
                                     "--manifests-dir", _MANIFESTS])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(tmp_path, "pt", "summary.json")))
    assert summary["success_rule"] == "threshold"
    assert summary["baseline_comparison"] is None


def _manifests_copy(tmp_path, with_preds=None):
    mdir = tmp_path / "manifests"
    (mdir / "baseline_preds").mkdir(parents=True)
    import shutil
    shutil.copy(os.path.join(_MANIFESTS, "breast_cancer.yaml"), mdir / "breast_cancer.yaml")
    if with_preds is not None:
        pd.DataFrame({"prediction": with_preds}).to_parquet(
            mdir / "baseline_preds" / "breast_cancer.test.parquet", index=False)
        shutil.copy(os.path.join(_MANIFESTS, "baseline_preds", "breast_cancer.val.parquet"),
                    mdir / "baseline_preds" / "breast_cancer.val.parquet")
    return str(mdir)


def test_main_without_baseline_predictions_errors(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    monkeypatch.setattr("sys.argv", ["prog", "--dataset", "breast_cancer",
                                     "--runs-dir", str(tmp_path / "runs"),
                                     "--manifests-dir", _manifests_copy(tmp_path)])
    with pytest.raises(SystemExit):
        crispfsm.run.main()


def test_main_misaligned_baseline_predictions_abort(tmp_path, monkeypatch):
    """Predictions eines anderen Splits (hier: umsortiert) duerfen nie still als
    Referenz dienen — der F1-Check gegen baseline_f1 bricht den Run ab."""
    _patch_backend(monkeypatch)
    real = pd.read_parquet(os.path.join(_MANIFESTS, "baseline_preds", "breast_cancer.test.parquet"))
    shuffled = real["prediction"].sample(frac=1.0, random_state=0).tolist()
    monkeypatch.setattr("sys.argv", ["prog", "--dataset", "breast_cancer",
                                     "--runs-dir", str(tmp_path / "runs"),
                                     "--manifests-dir", _manifests_copy(tmp_path, shuffled)])
    with pytest.raises(SystemExit, match="passen nicht"):
        crispfsm.run.main()


# --- Endurteil fuer alle Arme (S1 + B3, 23.09.) --------------------------------

from crispfsm.config import RunConfig as _RC
from crispfsm.datasets import prepare_dataset as _prep
from crispfsm.run import final_verdict, harness_version


def _verdict_setup(tmp_path, arm, preds_for):
    """Holdout mit Validierungsanteil; Baseline = Wahrheit (perfekt). preds_for(part,
    truth) liefert die Agent-Predictions je Anteil; gespeichert als gemeinsame Datei."""
    wd = tmp_path / arm
    info = _prep("breast_cancer", str(wd), neutralize_target=True, val_size=0.1)
    t = info["target_column"]
    h = wd / "holdout"
    idx = json.load(open(h / "eval_index.json"))
    out = [None] * (len(idx["val"]) + len(idx["test"]))
    for part in ("val", "test"):
        truth = list(pd.read_parquet(h / f"{part}.parquet")[t])
        pd.DataFrame({"prediction": truth}).to_parquet(h / f"baseline_{part}.parquet", index=False)
        for pos, v in zip(idx[part], preds_for(part, truth)):
            out[pos] = v
    pred_dir = wd / "work" if arm in ("crisp_react", "naked_react") else wd / "work" / "eval"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": out}).to_parquet(pred_dir / "predictions.parquet", index=False)
    cfg = _RC(run_id="v", workdir=str(wd), arm=arm, target=t,
              baseline_pred_path=str(h / "baseline_test.parquet"),
              baseline_val_path=str(h / "baseline_val.parquet"))
    return cfg, str(wd)


_PERFECT = lambda part, truth: truth
_FLIP = lambda truth: ["benign" if v == "malignant" else "malignant" for v in truth]


def test_verdict_free_arm_success(tmp_path):
    cfg, wd = _verdict_setup(tmp_path, "naked_react", _PERFECT)
    v = final_verdict(cfg, {"target_correct": True, "abstained": False}, wd, safety_max_calls=20)
    assert v["success"] and v["fail_reason"] is None and v["f1_test"] == 1.0


def test_verdict_wrong_target_fails_even_with_perfect_predictions(tmp_path):
    """B3: falsche Zielspalte = FAIL in allen Armen, auch wenn die Predictions passen."""
    cfg, wd = _verdict_setup(tmp_path, "naked_react", _PERFECT)
    v = final_verdict(cfg, {"target_correct": False, "abstained": False}, wd, safety_max_calls=20)
    assert not v["success"] and v["fail_reason"] == "wrong_target"
    assert v["f1_test"] == 1.0                      # Messung bleibt erhalten
    cfg, wd = _verdict_setup(tmp_path / "a", "hard_fsm", _PERFECT)
    v = final_verdict(cfg, {"bu_correct": False, "gate_passed": True, "current_phase": "eval",
                            "abstained": False}, wd, safety_max_calls=20)
    assert v["fail_reason"] == "wrong_target"


def test_verdict_gate_pass_on_val_is_not_success_if_test_is_worse(tmp_path):
    """S1: Das A/B-Gate misst auf val; das Endurteil auf test. Val perfekt, Test komplett
    falsch -> Prozess abgeschlossen, aber FAIL below_threshold."""
    cfg, wd = _verdict_setup(tmp_path, "hard_fsm",
                             lambda part, truth: truth if part == "val" else _FLIP(truth))
    final = {"bu_correct": True, "gate_passed": True, "current_phase": "eval", "abstained": False}
    v = final_verdict(cfg, final, wd, safety_max_calls=20)
    assert v["process_completed"] and not v["success"]
    assert v["fail_reason"] == "below_threshold" and v["f1_test"] < 0.1


def test_verdict_process_failure_keeps_process_reason(tmp_path):
    cfg, wd = _verdict_setup(tmp_path, "hard_fsm", _PERFECT)
    final = {"bu_correct": True, "gate_passed": False, "current_phase": "prep",
             "abstained": False, "gate_metrics": {}, "total_calls": 5}
    v = final_verdict(cfg, final, wd, safety_max_calls=20)
    assert not v["success"] and not v["process_completed"] and v["fail_reason"] == "gate_failed"


def test_verdict_abstain(tmp_path):
    cfg, wd = _verdict_setup(tmp_path, "crisp_react", _PERFECT)
    v = final_verdict(cfg, {"abstained": True}, wd, safety_max_calls=20)
    assert not v["success"] and v["fail_reason"] == "abstain"


def test_harness_version_records_commit_and_prompt_fingerprint():
    hv = harness_version()
    assert set(hv) == {"commit", "dirty", "prompt_sha256", "claude_version", "auto_memory",
                       "git_context"}
    assert len(hv["prompt_sha256"]) == 64


def test_harness_version_records_cli_version_and_memory_lock(monkeypatch):
    """Befund 24.10.: welche Claude-Code-Version lief und ob das Auto-Memory gesperrt war,
    muss pro Run belegbar sein (Paritaet zwischen Armen, Kern-Neustart)."""
    import subprocess
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="2.1.280 (Claude Code)\n")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr("subprocess.run", fake_run)
    hv = harness_version()
    assert hv["claude_version"] == "2.1.280 (Claude Code)"
    assert hv["auto_memory"] == "disabled"
    assert hv["git_context"] == "disabled"


class WrongTargetFake(ScriptedFake):
    """BU waehlt eine plausible, aber falsche Spalte (cmc: Wifes_education) statt der
    Zielspalte; alle spaeteren Prompts werden mitgeschnitten (B3: sie muessen mit dieser
    Wahl weiterarbeiten)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prompts = []

    def run(self, prompt, *, cwd, session_id, resume, max_turns, budget_usd, allowed_tools):
        from crispfsm.backend import WorkerResult, TokenUsage
        self.prompts.append(prompt)
        if "Business Understanding" in prompt:
            wrong = "Wifes_education"          # kategorial (4 Werte) -> BU-Gate passiert
            os.makedirs(f"{cwd}/bu", exist_ok=True)
            json.dump({"task_type": "classification", "target_column": wrong,
                       "metric": "f1", "feature_columns": None},
                      open(f"{cwd}/bu/task_spec.json", "w"))
            return WorkerResult(text="ok", usage=TokenUsage(10, 2), latency_ms=1.0,
                                session_id=session_id, raw={})
        return super().run(prompt, cwd=cwd, session_id=session_id, resume=resume,
                           max_turns=max_turns, budget_usd=budget_usd,
                           allowed_tools=allowed_tools)


def test_main_later_phases_work_with_bu_choice_and_run_fails(tmp_path, monkeypatch):
    """B3: A/B arbeiten ab DU mit der Zielspalte der eigenen BU-Phase (nicht der
    kanonischen). Eine falsche Wahl traegt der Run selbst — Ergebnis FAIL, wie bei D."""
    fake = WrongTargetFake()
    monkeypatch.setattr("crispfsm.run.ClaudeCliBackend", lambda *a, **k: fake)
    monkeypatch.setattr("sys.argv", ["prog", "--dataset", "cmc",
                                     "--runs-dir", str(tmp_path), "--run-id", "wt",
                                     "--manifests-dir", _MANIFESTS])
    crispfsm.run.main()
    summary = json.load(open(os.path.join(tmp_path, "wt", "summary.json")))
    wrong = "Wifes_education"
    du_prompts = [p for p in fake.prompts if "Data Understanding" in p]
    assert du_prompts and all(f"'{wrong}'" in p for p in du_prompts)
    assert summary["bu_correct"] is False and summary["bu_correct_first"] is False
    assert summary["success"] is False and summary["outcome"] == "FAIL"


def test_main_rerun_after_error_starts_with_clean_workdir(tmp_path, monkeypatch):
    """Neustart nach ERROR (crash/rate_limited): Artefakte des abgebrochenen
    Versuchs duerfen im neuen Run nicht mehr liegen."""
    _patch_backend(monkeypatch)
    run_dir = tmp_path / "myrun"
    (run_dir / "work" / "bu").mkdir(parents=True)
    (run_dir / "work" / "bu" / "stale.txt").write_text("alter Versuch")
    (run_dir / "summary.json").write_text(json.dumps({"outcome": "ERROR"}))
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--run-id", "myrun"])

    crispfsm.run.main()

    assert not (run_dir / "work" / "bu" / "stale.txt").exists()
    assert json.load(open(run_dir / "summary.json"))["success"] is True


def test_main_refuses_to_overwrite_measured_run(tmp_path, monkeypatch):
    _patch_backend(monkeypatch)
    run_dir = tmp_path / "myrun"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"outcome": "SUCCESS"}))
    monkeypatch.setattr("sys.argv",
                        ["prog", "--dataset", "breast_cancer",
                         "--runs-dir", str(tmp_path),
                         "--manifests-dir", _MANIFESTS,
                         "--run-id", "myrun"])

    with pytest.raises(SystemExit):
        crispfsm.run.main()
    assert json.load(open(run_dir / "summary.json"))["outcome"] == "SUCCESS"
