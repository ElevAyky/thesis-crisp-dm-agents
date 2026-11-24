# src/crispfsm/softgates.py
"""Arm B — Soft CRISP (roadmap §1, experiment-design.md UV1): gleiche
LangGraph-Topologie und Router wie Arm A, aber das Quality Gate entscheidet ein
LLM-Prüfer statt deterministischer Code. Die Ablation A vs. B isoliert damit den
Wert des Code-Gate-Enforcements (Kern von H2).

Designlinien (Details: docs/soft-gate-design.md):
- Gate-INPUT wie in Arm A (Artefakte + data/train.parquet im Workdir), nur der
  Gate-MECHANISMUS — nachrechnen, entscheiden, Feedback formulieren — ist LLM.
- Ausnahme eval: Die Test-Labels bleiben auch dem Prüfer verborgen. Der Harness
  misst selbst (scoring.py) und reicht NUR das Aggregat in den Prüfer-Prompt
  (f1/accuracy bzw. Formfehler-Diagnose) — dieselbe Information, die in Arm A
  über das Gate-Feedback zu den Workern fließt (Informations-Parität).
- Verdict-Protokoll: Der Prüfer schreibt gate/verdict.json. Der Harness löscht
  die Datei VOR dem Prüfer-Call (ein Worker kann kein Verdict vorpflanzen) und
  nach dem Lesen (Worker sehen keine Prüfer-Interna).
- judge_error (Verdict fehlt / unparsebar / ohne boolesches 'passed') zählt als
  normaler Gate-Fail und verbraucht das Retry-Budget der Phase: die
  Unzuverlässigkeit des LLM-Gates IST das Treatment, sie wird nicht wegretryt.
"""
from __future__ import annotations
import json
import os
import uuid

from crispfsm.config import RunConfig
from crispfsm.gates import GateResult
from crispfsm.scoring import (BaselineComparison, ScoreResult, compare_part_to_baseline,
                              score_part)

# Bewusst identisch zu phases.ALLOWED_TOOLS (Tool-Parität Prüfer <-> Worker);
# dupliziert statt importiert, um den Import-Zyklus phases <-> softgates zu
# vermeiden — test_softgates sichert die Gleichheit ab.
JUDGE_ALLOWED_TOOLS = ["Bash", "Read", "Write", "Edit"]

VERDICT_RELPATH = os.path.join("gate", "verdict.json")

# WICHTIG für Prompt-Wortwahl: Die Worker-Phasen-Keywords ("Business
# Understanding", "Data Understanding", "Data Preparation", "Modeling",
# "Evaluation") dürfen in Prüfer-Prompts NICHT vorkommen — Fakes und
# Log-Analyse routen darüber (prompts.py, test_phase_keywords_are_unambiguous).
# Prüfer-Prompts referenzieren Phasen darum nur über ihre Kurz-Codes.

_JUDGE_COMMON = (
    "You are the independent reviewer of a Quality Gate (GATE CHECK '{phase}') "
    "in an ML pipeline. A worker has previously written artifacts to the current "
    "directory. Check with Python/pandas (via Bash) whether the "
    "CRITERIA below are met — recompute exactly, do not estimate and "
    "do not trust any self-report of the worker unless a criterion explicitly "
    "says so. Do not modify any artifacts and do not access anything outside the "
    "current directory.\n"
    "Then write your verdict as JSON to EXACTLY ONE new file "
    "'gate/verdict.json':\n"
    "  {{\"passed\": true|false, \"feedback\": \"<if false: concrete, fixable "
    "diagnosis for the worker; if true: short confirmation>\"{extra}}}\n"
    "The field \"passed\" decides whether the pipeline continues.\n"
)

_EXTRA_FIELDS = {
    "bu": ", \"abstain\": true|false",
    "du": ", \"spec_mismatch\": true|false",
    "eval": ", \"form_error\": true|false",
}

_BU = (
    "CRITERIA for 'bu/task_spec.json':\n"
    "1. The file exists and is a JSON object.\n"
    "2. Abstention: {{\"cannot_determine\": true, ...}} is a legitimate verdict "
    "of the worker (target cannot be derived from task and data). In that case write "
    "{{\"passed\": false, \"abstain\": true, \"feedback\": \"<the worker's "
    "justification>\"}}.\n"
    "3. Otherwise required fields: task_type, target_column, metric.\n"
    "4. task_type must be \"classification\" — nothing else is in scope.\n"
    "5. target_column must exist as a column in 'data/train.parquet'.\n"
    "6. metric must be \"f1\" or \"accuracy\".\n"
    "7. feature_columns (if not null): every column must exist in the "
    "schema, and target_column must NOT appear in it (leakage).\n"
    "8. target_column must not be a continuous numeric field: more than "
    "20 distinct values with a numeric dtype = regression signal, not a "
    "valid classification target (boolean columns are ok).\n"
)

_DU = (
    "CRITERIA for 'du/data_report.json' (recompute every number EXACTLY against "
    "'data/train.parquet'):\n"
    "1. The file exists and is a JSON object.\n"
    "2. Contradiction: {{\"spec_mismatch\": true, \"spec_mismatch_reason\": ...}} "
    "is a legitimate verdict of the worker (the data contradict the derived "
    "task). In that case write {{\"passed\": false, \"spec_mismatch\": true, "
    "\"feedback\": \"<the worker's reason>\"}}.\n"
    "3. Otherwise required fields: n_rows, columns, class_balance.\n"
    "4. n_rows must equal the actual row count.\n"
    "5. columns must contain EVERY column of the schema (none invented, none "
    "missing), each as an object {{\"dtype\": ..., \"n_missing\": ...}}. "
    "dtype rule: boolean pandas dtype -> \"bool\", otherwise numeric dtype "
    "-> \"numeric\", everything else -> \"categorical\". n_missing must be exactly "
    "right.\n"
    "6. class_balance must equal the exact value_counts of the target column '{target}'.\n"
    "7. excluded_columns: list of {{\"column\", \"reason\"}} objects; every "
    "column must exist; the target column '{target}' must not be "
    "excluded.\n"
)

_PREP = (
    "CRITERIA for 'prep/prepared.parquet':\n"
    "1. The file exists and has more than 0 rows.\n"
    "2. Column '{target}' is present.\n"
    "3. No NaN in the feature columns.\n"
    # R2-9: wie gates.check_prep (select_dtypes(exclude="number")) — bool ist dort NICHT numerisch
    "4. All feature columns have a numeric int/float dtype (encoding completed; bool counts "
    "as non-numeric).\n"
    "5. '{target}' has at least 2 classes.\n"
)

_MODEL = (
    "CRITERIA for the model artifact:\n"
    "1. 'model/model.pkl' exists, can be loaded with pickle and has a "
    ".predict() method.\n"
    # R2-5: wie gates.check_model — Metrik aus der Spec, nicht fest Accuracy
    "2. 'model/metrics.json' exists with the field metric and the numeric fields "
    "val_score and dummy_score; metric must equal the metric in 'bu/task_spec.json'.\n"
    # S6 (23.09.): Der Validierungs-Split des Workers ist nicht reproduzierbar — der
    # Pruefer vergleicht die berichteten Werte (wie gates.check_model in Arm A).
    "3. val_score must be ABOVE dummy_score as reported in 'model/metrics.json' "
    "(the worker's validation split cannot be reproduced; do not re-split or retrain) — "
    "otherwise the model has learned nothing.\n"
)

_EVAL = (
    "CRITERIA for 'eval/predictions.parquet':\n"
    # R2-12: wie gates.check_eval — liest nur die Spalte 'prediction', Zusatzspalten sind ok
    "1. A column 'prediction' with the same row count as "
    "'data/test_features.parquet' and no NaN.\n"
    "2. Success contract: {contract}\n"
    "The test labels are not available to you; do not search for them — the harness has "
    "already measured deterministically:\n"
    "  MEASUREMENT: {score_info}\n"
    "Decide based on the criteria and the measurement:\n"
    "- Form error (the file violates the contract from criterion 1): "
    "{{\"passed\": false, \"form_error\": true, ...}} — your feedback goes to "
    "the worker who produced the predictions.\n"
    "- File formally valid, but quality insufficient: {{\"passed\": false, "
    "\"form_error\": false, ...}} — your feedback goes to the worker who trained the "
    "model.\n"
    "- Both criteria met: {{\"passed\": true, \"form_error\": false, ...}}.\n"
)


def judge_prompt(phase: str, cfg: RunConfig, score: ScoreResult | None = None,
                 comparison: BaselineComparison | None = None,
                 target: str | None = None) -> str:
    """Prüfer-Prompt einer Phase. Für eval MUSS die Harness-Messung (score)
    mitgegeben werden — der Prüfer selbst hat keinen Label-Zugriff. Mit
    cfg.baseline_pred_path gilt die Baseline-Regel (23.09.): der Prüfer bekommt
    denselben gepaarten Vergleich, den gates.check_eval in Arm A auswertet."""
    head = _JUDGE_COMMON.format(phase=phase, extra=_EXTRA_FIELDS.get(phase, ""))
    if phase == "bu":
        return head + _BU.format()
    if phase == "du":
        # B3: Pruefer prueft gegen die Zielspalte, mit der der Prozess arbeitet
        return head + _DU.format(target=target or cfg.target)
    if phase == "prep":
        return head + _PREP.format(target=target or cfg.target)
    if phase == "model":
        return head + _MODEL.format()
    if phase == "eval":
        if cfg.baseline_pred_path:
            # S7 (23.09.): "reference baseline" + KI-Obergrenze >= 0 (= Formulierung der
            # Nicht-Unterlegenheits-Regel aus gates.check_eval).
            contract = (f"{cfg.eval_metric} must not be significantly below the reference "
                        "baseline. It is measured paired on the same rows: the upper bound "
                        "of the 95% confidence interval of the difference (model minus "
                        "reference baseline) must be ≥ 0.")
        else:
            contract = f"{cfg.eval_metric} must reach at least {cfg.eval_f1_threshold}."
        if not score.ok:
            info = f"Form error — {score.feedback}"
        elif comparison is not None:
            info = (f"f1={score.f1:.4f}, accuracy={score.accuracy:.4f}, "
                    f"reference_baseline_f1={comparison.baseline_f1:.4f}, "
                    f"difference={comparison.diff:+.4f}, 95% CI of the difference="
                    f"[{comparison.ci_lo:+.4f}, {comparison.ci_hi:+.4f}]")
        else:
            info = f"f1={score.f1:.4f}, accuracy={score.accuracy:.4f}"
        return head + _EVAL.format(contract=contract, score_info=info)
    raise KeyError(phase)


def _read_verdict(work: str) -> tuple[dict | None, str]:
    """gate/verdict.json -> (verdict, fehlerdiagnose). Analog react._read_result_json:
    unbrauchbar zählt wie fehlend, aber die Diagnose bleibt unterscheidbar."""
    path = os.path.join(work, VERDICT_RELPATH)
    if not os.path.exists(path):
        return None, "gate/verdict.json is missing"
    try:
        with open(path, encoding="utf-8") as f:
            verdict = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        return None, f"gate/verdict.json is not valid JSON: {e}"
    if not isinstance(verdict, dict):
        return None, "gate/verdict.json is not a JSON object"
    if not isinstance(verdict.get("passed"), bool):
        return None, "gate/verdict.json has no boolean field 'passed'"
    return verdict, ""


def _load_artifact(path: str) -> dict | None:
    """Toleranter Artefakt-Load für Feed-Forward/Messung nach einem Judge-Pass.
    Ein Prüfer kann (fälschlich) ein kaputtes Artefakt durchwinken — dann fehlt
    das Feed-Forward eben; das ist ein Soft-Arm-Befund, kein Harness-Crash."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def run_soft_gate(phase: str, backend, cfg: RunConfig, work: str, *,
                  target: str | None = None):
    """Führt die LLM-Gate-Prüfung einer Phase aus: frische Prüfer-Session,
    Verdict lesen, in ein GateResult mit denselben Routing-Flags wie die
    deterministischen Gates übersetzen (abstain / spec_mismatch / form_error —
    die Router in graph.py bleiben unverändert). Gibt (GateResult, WorkerResult
    des Prüfer-Calls) zurück; der Aufrufer (phases.py) verbucht Usage + Event."""
    verdict_path = os.path.join(work, VERDICT_RELPATH)
    os.makedirs(os.path.dirname(verdict_path), exist_ok=True)
    if os.path.exists(verdict_path):
        os.remove(verdict_path)          # kein vom Worker vorgepflanztes Verdict

    score = comparison = None
    if phase == "eval":
        # Gleicher Anteil + gleiche Baseline wie das deterministische Gate von Arm A (S1):
        # im Kern die Validierungszeilen, nie das finale Test-Set.
        from crispfsm.phases import gate_part_and_baseline   # lokal: Import-Zyklus
        part, baseline = gate_part_and_baseline(cfg)
        holdout_dir = os.path.join(cfg.workdir, "holdout")
        pred_path = os.path.join(work, "eval", "predictions.parquet")
        score = score_part(pred_path, holdout_dir, part, target=cfg.target)
        if score.ok and baseline and cfg.eval_metric == "f1":
            comparison = compare_part_to_baseline(pred_path, holdout_dir, part, baseline,
                                                  target=cfg.target)

    result = backend.run(judge_prompt(phase, cfg, score=score, comparison=comparison,
                                      target=target), cwd=work,
                         session_id=str(uuid.uuid4()), resume=False,
                         max_turns=cfg.max_turns,
                         budget_usd=cfg.budget_usd_per_call,
                         allowed_tools=JUDGE_ALLOWED_TOOLS)

    verdict, err = _read_verdict(work)
    if os.path.exists(verdict_path):
        os.remove(verdict_path)          # Worker sehen keine Prüfer-Interna

    metrics: dict = {}
    if phase == "eval":
        # Harness-Messung immer mitführen: phases.py leitet success/final_f1
        # daraus ab (fabrikationssicher — ein Judge-Pass allein reicht nicht).
        metrics.update({"f1": score.f1, "accuracy": score.accuracy,
                        "scored_ok": score.ok})
        if comparison is not None:
            metrics.update({"baseline_f1": comparison.baseline_f1, "diff": comparison.diff,
                            "ci_lo": comparison.ci_lo, "ci_hi": comparison.ci_hi,
                            "noninferior": comparison.noninferior})

    if verdict is None:
        metrics["judge_error"] = True
        return GateResult(False, metrics=metrics, feedback=(
            f"Gate check without a usable verdict ({err}). "
            "Produce your artifact again exactly according to the task contract."
        )), result

    passed = verdict["passed"]
    feedback = str(verdict.get("feedback", ""))
    metrics["verdict"] = verdict
    abstain = False

    if phase == "bu":
        abstain = bool(verdict.get("abstain")) and not passed
        if passed:
            spec = _load_artifact(os.path.join(work, "bu", "task_spec.json"))
            if spec is not None:
                metrics["spec"] = spec
    elif phase == "du":
        if not passed and verdict.get("spec_mismatch"):
            metrics["spec_mismatch"] = True
        if passed:
            report = _load_artifact(os.path.join(work, "du", "data_report.json"))
            if report is not None:
                metrics["report"] = report
    elif phase == "eval":
        if not passed and verdict.get("form_error"):
            metrics["form_error"] = True

    return GateResult(passed, metrics=metrics, feedback=feedback,
                      abstain=abstain), result
