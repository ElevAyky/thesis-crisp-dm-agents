# src/crispfsm/gates.py
from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass, field
import pandas as pd
from crispfsm.scoring import compare_part_to_baseline, score_part

@dataclass
class GateResult:
    passed: bool
    metrics: dict = field(default_factory=dict)
    feedback: str = ""
    abstain: bool = False

# muss mit der eval_metric-Validierung in config.RunConfig übereinstimmen
VALID_METRICS = {"classification": {"f1", "accuracy"}}
MAX_TARGET_CARDINALITY = 20   # mehr distinkte Werte + numerischer dtype => Regressions-Signal

def _load_parquet(path: str):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def check_prep(workdir: str, *, target: str = "target") -> GateResult:
    path = os.path.join(workdir, "prep", "prepared.parquet")
    df = _load_parquet(path)
    if df is None:
        return GateResult(False, feedback="prep/prepared.parquet is missing — the phase produced no artifact.")
    if target not in df.columns:
        return GateResult(False, feedback=f"Column '{target}' is missing in prepared.parquet.")
    features = df.drop(columns=[target])
    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        return GateResult(False, feedback=f"NaN in feature columns: {nan_cols}.")
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        return GateResult(False, feedback=f"Non-numeric features (encoding missing): {non_numeric}.")
    if len(df) == 0:
        return GateResult(False, feedback="prepared.parquet has 0 rows.")
    n_classes = df[target].nunique()
    if n_classes < 2:
        return GateResult(False, feedback=f"target has only {n_classes} class(es), need >=2.")
    return GateResult(True, metrics={"n_rows": len(df), "n_features": features.shape[1],
                                     "n_classes": int(n_classes)})

def check_model(workdir: str) -> GateResult:
    model_path = os.path.join(workdir, "model", "model.pkl")
    metrics_path = os.path.join(workdir, "model", "metrics.json")
    if not os.path.exists(model_path):
        return GateResult(False, feedback="model/model.pkl is missing.")
    try:
        # Safe: model.pkl is a self-generated artifact written by this
        # prototype's own worker into its local work/ dir, not untrusted input.
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return GateResult(False, feedback=f"model.pkl cannot be loaded: {e}")
    if not hasattr(model, "predict"):
        return GateResult(False, feedback="Loaded object has no .predict() method.")
    if not os.path.exists(metrics_path):
        return GateResult(False, feedback="model/metrics.json is missing.")
    m = json.load(open(metrics_path))
    # R2-5 (23.09., Johannes): Vergleich in der Metrik, die BU gewaehlt hat — nicht fest in
    # Accuracy. Sonst faellt ein auf F1 optimiertes Modell auf unbalancierten Sets (z. B.
    # class_weight='balanced') unter die Dummy-Accuracy, obwohl es nach der Zielmetrik
    # besser ist. Die Metrik muss zur Spec passen (kein Ausweichen auf eine leichtere).
    val = m.get("val_score")
    dummy = m.get("dummy_score")
    if val is None or dummy is None or m.get("metric") is None:
        return GateResult(False, feedback="metrics.json needs metric, val_score and dummy_score.")
    spec_path = os.path.join(workdir, "bu", "task_spec.json")
    expected = None
    if os.path.exists(spec_path):
        try:
            expected = json.load(open(spec_path)).get("metric")
        except (json.JSONDecodeError, AttributeError):
            expected = None
    if expected is not None and m["metric"] != expected:
        return GateResult(False, metrics=m, feedback=(
            f"metrics.json reports metric '{m['metric']}', but the task specification in "
            f"'bu/task_spec.json' uses '{expected}'. Report val_score and dummy_score in "
            f"'{expected}'."))
    if val <= dummy:
        return GateResult(False, metrics=m, feedback=(
            f"val_score {val:.3f} ({m['metric']}) not above dummy_score {dummy:.3f}."))
    return GateResult(True, metrics=m)

def check_bu(workdir: str) -> GateResult:
    """BU-Gate (bu-gate.md §3): validiert die ML-Task-Spec deterministisch gegen das
    echte Daten-Schema. Kennt die Ground Truth NICHT — Korrektheit wird separat
    gemessen (phases.py, bu_correct)."""
    spec_path = os.path.join(workdir, "bu", "task_spec.json")
    if not os.path.exists(spec_path):
        return GateResult(False, feedback="bu/task_spec.json is missing — the phase produced no artifact.")
    try:
        spec = json.load(open(spec_path))
    except json.JSONDecodeError as e:
        return GateResult(False, feedback=f"bu/task_spec.json is not valid JSON: {e}")
    if not isinstance(spec, dict):
        return GateResult(False, feedback="task_spec.json must be a JSON object.")

    if spec.get("cannot_determine"):
        reason = spec.get("reason", "no reason given")
        return GateResult(False, abstain=True, metrics={"spec": spec},
                          feedback=f"BU: cannot be determined — {reason}")

    df = _load_parquet(os.path.join(workdir, "data", "train.parquet"))
    if df is None:
        return GateResult(False, feedback="data/train.parquet is missing — no schema to validate against.")

    missing = [k for k in ("task_type", "target_column", "metric") if not spec.get(k)]
    if missing:
        return GateResult(False, feedback=f"Required fields missing in task_spec.json: {missing}.")

    task_type = spec["task_type"]
    if task_type not in VALID_METRICS:
        return GateResult(False, feedback=(
            f"task_type '{task_type}' not supported (scope: {sorted(VALID_METRICS)})."))

    target = spec["target_column"]
    if target not in df.columns:
        return GateResult(False, feedback=(
            f"target_column '{target}' does not exist in the data schema. "
            f"Available columns: {list(df.columns)}."))

    metric = spec["metric"]
    if metric not in VALID_METRICS[task_type]:
        return GateResult(False, feedback=(
            f"metric '{metric}' invalid for {task_type} (allowed: {sorted(VALID_METRICS[task_type])})."))

    features = spec.get("feature_columns")
    if features is not None:
        ghost = [c for c in features if c not in df.columns]
        if ghost:
            return GateResult(False, feedback=f"feature_columns do not exist in the schema: {ghost}.")
        if target in features:
            return GateResult(False, feedback="target_column must not be in feature_columns (leakage).")

    # Typ-Konsistenz Spec <-> Daten: hochkardinales numerisches Target (Float ODER
    # Integer, z.B. IDs) ist kein Classification-Target. Bool ausgenommen (0/1-Target
    # ist legitim); niedrige Kardinalität fängt die Schwelle ab.
    col = df[target]
    if (pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
            and col.nunique() > MAX_TARGET_CARDINALITY):
        return GateResult(False, feedback=(
            f"target_column '{target}' is continuous ({col.nunique()} distinct numeric values) — "
            f"regression signal, not a valid classification target."))

    return GateResult(True, metrics={"spec": spec, "n_classes": int(col.nunique())})

_DTYPE_KINDS = ("numeric", "bool", "categorical")

def _dtype_kind(col: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(col):
        return "bool"
    if pd.api.types.is_numeric_dtype(col):
        return "numeric"
    return "categorical"

def check_du(workdir: str, *, target: str) -> GateResult:
    """DU-Gate (roadmap §1): Faktentreue-Check — der Harness rechnet dieselben
    Statistiken selbst aus data/train.parquet und vergleicht („hast du
    halluziniert?"). Deterministisch, gleiche Philosophie wie check_bu.
    Ausreißer bewusst NICHT im harten Vergleich: jede Outlier-Definition ist
    methodenabhängig (IQR-Faktor, Quantil-Interpolation) und würde das Gate
    flaky machen — Ausreißer-Befunde laufen als Freitext über quality_issues.
    excluded_columns ist Messpunkt (leakage_detected in run.py), kein
    Pass-Kriterium — validiert wird nur Form, Existenz und Target-Schutz."""
    report_path = os.path.join(workdir, "du", "data_report.json")
    if not os.path.exists(report_path):
        return GateResult(False, feedback="du/data_report.json is missing — the phase produced no artifact.")
    try:
        report = json.load(open(report_path))
    except json.JSONDecodeError as e:
        return GateResult(False, feedback=f"du/data_report.json is not valid JSON: {e}")
    if not isinstance(report, dict):
        return GateResult(False, feedback="data_report.json must be a JSON object.")

    if report.get("spec_mismatch"):
        reason = report.get("spec_mismatch_reason", "no reason given")
        return GateResult(False, metrics={"spec_mismatch": True, "report": report},
                          feedback=("The data understanding (DU) phase found that the data "
                                    f"contradict the task specification: {reason}"))

    df = _load_parquet(os.path.join(workdir, "data", "train.parquet"))
    if df is None:
        return GateResult(False, feedback="data/train.parquet is missing — no data to check against.")

    missing = [k for k in ("n_rows", "columns", "class_balance") if k not in report]
    if missing:
        return GateResult(False, feedback=f"Required fields missing in data_report.json: {missing}.")

    if report["n_rows"] != len(df):
        return GateResult(False, feedback=(
            f"n_rows {report['n_rows']} is wrong — train.parquet has {len(df)} rows."))

    cols = report["columns"]
    if not isinstance(cols, dict):
        return GateResult(False, feedback="'columns' must be an object {column: {dtype, n_missing}}.")
    if set(cols) != set(df.columns):
        invented = sorted(set(cols) - set(df.columns))
        omitted = sorted(set(df.columns) - set(cols))
        return GateResult(False, feedback=(
            f"Column list is wrong — invented: {invented}, missing: {omitted}."))
    for name, info in cols.items():
        if not isinstance(info, dict):
            return GateResult(False, feedback=f"columns['{name}'] must be an object {{dtype, n_missing}}.")
        true_kind = _dtype_kind(df[name])
        if info.get("dtype") != true_kind:
            return GateResult(False, feedback=(
                f"dtype of '{name}' is wrong: '{info.get('dtype')}' (actual: '{true_kind}')."))
        true_missing = int(df[name].isna().sum())
        if info.get("n_missing") != true_missing:
            return GateResult(False, feedback=(
                f"n_missing of '{name}' is wrong: {info.get('n_missing')} (actual: {true_missing})."))

    true_balance = {str(k): int(v) for k, v in df[target].value_counts().items()}
    reported = report["class_balance"]
    if (not isinstance(reported, dict)
            or {str(k): v for k, v in reported.items()} != true_balance):
        return GateResult(False, feedback=f"class_balance is wrong — actual: {true_balance}.")

    excluded = report.get("excluded_columns") or []
    if (not isinstance(excluded, list)
            or any(not isinstance(e, dict) or "column" not in e for e in excluded)):
        return GateResult(False, feedback="excluded_columns must be a list of {column, reason} objects.")
    ghost = [e["column"] for e in excluded if e["column"] not in df.columns]
    if ghost:
        return GateResult(False, feedback=f"excluded_columns do not exist in the schema: {ghost}.")
    if any(e["column"] == target for e in excluded):
        return GateResult(False, feedback=f"Target column '{target}' must not be excluded.")

    return GateResult(True, metrics={"report": report})

def check_eval(workdir: str, *, holdout_dir: str, target: str,
               metric: str = "f1", threshold: float = 0.5,
               baseline_pred_path: str | None = None, part: str = "test") -> GateResult:
    """Eval-Gate (Spec 2026-07-18 §1.3): scored work/eval/predictions.parquet
    deterministisch gegen die verborgenen Labels. Kein Self-Report — der Worker sieht
    keine Labels und kann die Zahl nicht liefern/gamen. part (S1, 23.09.): im Kern 'val'
    — das Gate misst auf dem Validierungsanteil des Harness, NIE auf dem finalen
    Test-Set; baseline_pred_path muss zum Anteil passen."""
    pred_path = os.path.join(workdir, "eval", "predictions.parquet")
    score = score_part(pred_path, holdout_dir, part, target=target)
    if not score.ok:
        # Formfehler (Datei fehlt/kaputt/falsche Spalte/Zeilenzahl/NaN): Fehler des
        # EVAL-Deliverables, nicht der Modellqualität. Flag für den Router (graph.py),
        # damit das Feedback den Eval-Worker erreicht statt den Model-Worker.
        return GateResult(False, metrics={"form_error": True}, feedback=score.feedback)
    metrics = {"f1": score.f1, "accuracy": score.accuracy}
    val = metrics.get(metric)
    if val is None:
        return GateResult(False, metrics=metrics,
                          feedback=f"metric '{metric}' unknown (allowed: f1, accuracy).")
    if baseline_pred_path is not None and metric == "f1":
        # Erfolgsregel seit 23.09.: nicht signifikant schlechter als die Default-
        # Baseline auf denselben Holdout-Zeilen (gepaarter Bootstrap, 95 %).
        cmp = compare_part_to_baseline(pred_path, holdout_dir, part, baseline_pred_path,
                                       target=target)
        metrics.update({"baseline_f1": cmp.baseline_f1, "diff": cmp.diff,
                        "ci_lo": cmp.ci_lo, "ci_hi": cmp.ci_hi,
                        "noninferior": cmp.noninferior, "part": part})
        if not cmp.noninferior:
            return GateResult(False, metrics=metrics, feedback=(
                f"On the rows used by this quality check, f1 {cmp.f1:.3f} is significantly "
                f"below the reference baseline {cmp.baseline_f1:.3f} (a default "
                f"scikit-learn model trained on the same data, not your dummy; difference "
                f"{cmp.diff:+.3f}, 95% CI [{cmp.ci_lo:+.3f}, {cmp.ci_hi:+.3f}])."))
        return GateResult(True, metrics=metrics)
    if val < threshold:
        return GateResult(False, metrics=metrics,
                          feedback=f"{metric} {val:.3f} below threshold {threshold}.")
    return GateResult(True, metrics=metrics)
