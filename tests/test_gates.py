# tests/test_gates.py
import os, json, pickle
import pandas as pd
from sklearn.dummy import DummyClassifier
from crispfsm.gates import check_prep, check_model, check_eval, check_bu, check_du

def _workdir(tmp_path):
    w = str(tmp_path / "work")
    for sub in ("prep", "model", "eval", "data"):
        os.makedirs(os.path.join(w, sub), exist_ok=True)
    return w

def test_check_prep_passes_on_clean_data(tmp_path):
    w = _workdir(tmp_path)
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    df.to_parquet(os.path.join(w, "prep", "prepared.parquet"), index=False)
    res = check_prep(w, target="target")
    assert res.passed is True

def test_check_prep_fails_on_nan(tmp_path):
    w = _workdir(tmp_path)
    df = pd.DataFrame({"f1": [1.0, None, 3.0, 4.0], "target": [0, 1, 0, 1]})
    df.to_parquet(os.path.join(w, "prep", "prepared.parquet"), index=False)
    res = check_prep(w, target="target")
    assert res.passed is False
    assert "nan" in res.feedback.lower()

def test_check_prep_fails_on_missing_file(tmp_path):
    w = _workdir(tmp_path)
    res = check_prep(w, target="target")
    assert res.passed is False
    assert "prepared.parquet" in res.feedback

def test_check_model_fails_when_not_above_dummy(tmp_path):
    w = _workdir(tmp_path)
    clf = DummyClassifier(strategy="most_frequent").fit([[0]], [0])
    with open(os.path.join(w, "model", "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    json.dump({"metric": "f1", "val_score": 0.50, "dummy_score": 0.50},
              open(os.path.join(w, "model", "metrics.json"), "w"))
    res = check_model(w)
    assert res.passed is False

def test_check_model_passes_above_dummy(tmp_path):
    w = _workdir(tmp_path)
    clf = DummyClassifier(strategy="most_frequent").fit([[0]], [0])
    with open(os.path.join(w, "model", "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    json.dump({"metric": "f1", "val_score": 0.92, "dummy_score": 0.63},
              open(os.path.join(w, "model", "metrics.json"), "w"))
    res = check_model(w)
    assert res.passed is True

# --- check_eval: Predictions-basiert (Spec 2026-07-18 §1.3) ------------------

def _eval_setup(tmp_path, y_true, y_pred):
    """Baut workdir (mit eval/predictions.parquet) + holdout (mit test.parquet)."""
    w = str(tmp_path / "work")
    h = str(tmp_path / "holdout")
    os.makedirs(os.path.join(w, "eval"), exist_ok=True)
    os.makedirs(h, exist_ok=True)
    pd.DataFrame({"f1": range(len(y_true)), "target": y_true}).to_parquet(
        os.path.join(h, "test.parquet"), index=False)
    if y_pred is not None:
        pd.DataFrame({"prediction": y_pred}).to_parquet(
            os.path.join(w, "eval", "predictions.parquet"), index=False)
    return w, h


def test_check_eval_passes_on_good_predictions(tmp_path):
    w, h = _eval_setup(tmp_path, [0, 1, 0, 1], [0, 1, 0, 1])
    res = check_eval(w, holdout_dir=h, target="target", threshold=0.5)
    assert res.passed is True
    assert res.metrics["f1"] == 1.0
    assert res.metrics["accuracy"] == 1.0


def test_check_eval_fails_below_threshold_with_feedback(tmp_path):
    # invertierte Labels -> F1 = 0.0
    w, h = _eval_setup(tmp_path, [0, 1, 0, 1], [1, 0, 1, 0])
    res = check_eval(w, holdout_dir=h, target="target", threshold=0.5)
    assert res.passed is False
    assert "threshold" in res.feedback
    assert res.metrics["f1"] == 0.0
    # Schwellen-Fail ist Modellqualität, KEIN Formfehler (Router: retry -> model)
    assert "form_error" not in res.metrics


def test_check_eval_fails_when_predictions_missing(tmp_path):
    w, h = _eval_setup(tmp_path, [0, 1], None)
    res = check_eval(w, holdout_dir=h, target="target", threshold=0.5)
    assert res.passed is False
    assert "missing" in res.feedback
    # Kontraktfehler -> form_error-Flag, damit der Router eval selbst retryt
    assert res.metrics.get("form_error") is True


def test_check_eval_fails_on_row_mismatch(tmp_path):
    w, h = _eval_setup(tmp_path, [0, 1, 0, 1], [0, 1])
    res = check_eval(w, holdout_dir=h, target="target", threshold=0.5)
    assert res.passed is False
    assert "Row count" in res.feedback
    assert res.metrics.get("form_error") is True


def test_check_eval_passes_at_exact_threshold(tmp_path):
    w, h = _eval_setup(tmp_path, [0, 1, 0, 1], [0, 1, 0, 1])
    res = check_eval(w, holdout_dir=h, target="target", threshold=1.0)
    assert res.passed is True


def test_check_eval_respects_metric_parameter(tmp_path):
    """Metrik-agnostisch: bewertet wird cfg.eval_metric — der Parameter wählt den
    bewerteten Schlüssel; unbekannte Metrik failt mit Diagnose."""
    w, h = _eval_setup(tmp_path, [0, 1, 0, 1], [0, 1, 0, 1])
    assert check_eval(w, holdout_dir=h, target="target",
                      metric="accuracy", threshold=0.5).passed is True
    res = check_eval(w, holdout_dir=h, target="target",
                     metric="bogus", threshold=0.5)
    assert res.passed is False
    assert "bogus" in res.feedback


# --- check_prep: additional branches -------------------------------------

def test_check_prep_fails_when_target_column_missing(tmp_path):
    w = _workdir(tmp_path)
    # df has features but no "target" column
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    df.to_parquet(os.path.join(w, "prep", "prepared.parquet"), index=False)
    res = check_prep(w, target="target")
    assert res.passed is False
    assert "target" in res.feedback
    assert "missing" in res.feedback


def test_check_prep_fails_on_non_numeric_feature(tmp_path):
    w = _workdir(tmp_path)
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0],
        "cat": ["a", "b", "a", "b"],   # string feature -> needs encoding
        "target": [0, 1, 0, 1],
    })
    df.to_parquet(os.path.join(w, "prep", "prepared.parquet"), index=False)
    res = check_prep(w, target="target")
    assert res.passed is False
    assert "Non-numeric features" in res.feedback
    assert "encoding" in res.feedback
    assert "cat" in res.feedback


def test_check_prep_fails_on_zero_rows(tmp_path):
    w = _workdir(tmp_path)
    # valid numeric columns but no rows
    df = pd.DataFrame({"f1": pd.Series([], dtype="float64"),
                       "target": pd.Series([], dtype="int64")})
    df.to_parquet(os.path.join(w, "prep", "prepared.parquet"), index=False)
    res = check_prep(w, target="target")
    assert res.passed is False
    assert "0 rows" in res.feedback


def test_check_prep_fails_on_single_class_target(tmp_path):
    w = _workdir(tmp_path)
    # >=1 feature, >=2 rows, numeric, no NaN, but target has only one class
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [1, 1, 1, 1]})
    df.to_parquet(os.path.join(w, "prep", "prepared.parquet"), index=False)
    res = check_prep(w, target="target")
    assert res.passed is False
    assert "only 1 class" in res.feedback


def test_check_prep_metrics_on_clean_pass(tmp_path):
    w = _workdir(tmp_path)
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 1, 2, 0, 1, 2],
    })
    df.to_parquet(os.path.join(w, "prep", "prepared.parquet"), index=False)
    res = check_prep(w, target="target")
    assert res.passed is True
    assert res.metrics["n_rows"] == 6
    assert res.metrics["n_features"] == 2
    assert res.metrics["n_classes"] == 3


# --- check_model: additional branches ------------------------------------

def test_check_model_fails_when_model_pkl_missing(tmp_path):
    w = _workdir(tmp_path)
    # no model.pkl written at all
    res = check_model(w)
    assert res.passed is False
    assert "model/model.pkl is missing" in res.feedback


def test_check_model_fails_on_unpicklable_model(tmp_path):
    w = _workdir(tmp_path)
    # raw garbage bytes -> pickle.load raises
    with open(os.path.join(w, "model", "model.pkl"), "wb") as f:
        f.write(b"\x00\x01not a pickle at all\xff")
    res = check_model(w)
    assert res.passed is False
    assert "cannot be loaded" in res.feedback


def test_check_model_fails_when_object_has_no_predict(tmp_path):
    w = _workdir(tmp_path)
    # pickle an object without .predict (a plain dict)
    with open(os.path.join(w, "model", "model.pkl"), "wb") as f:
        pickle.dump({"not": "a model"}, f)
    res = check_model(w)
    assert res.passed is False
    assert ".predict" in res.feedback


def test_check_model_fails_when_metrics_json_missing(tmp_path):
    w = _workdir(tmp_path)
    # model loads & has .predict, but no metrics.json
    clf = DummyClassifier(strategy="most_frequent").fit([[0]], [0])
    with open(os.path.join(w, "model", "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    res = check_model(w)
    assert res.passed is False
    assert "model/metrics.json is missing" in res.feedback


def test_check_model_fails_when_metrics_missing_required_key(tmp_path):
    w = _workdir(tmp_path)
    clf = DummyClassifier(strategy="most_frequent").fit([[0]], [0])
    with open(os.path.join(w, "model", "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    # missing dummy_score
    json.dump({"metric": "f1", "val_score": 0.92},
              open(os.path.join(w, "model", "metrics.json"), "w"))
    res = check_model(w)
    assert res.passed is False
    assert "metric, val_score and dummy_score" in res.feedback

    # also missing val_score
    json.dump({"metric": "f1", "dummy_score": 0.63},
              open(os.path.join(w, "model", "metrics.json"), "w"))
    res = check_model(w)
    assert res.passed is False
    assert "metric, val_score and dummy_score" in res.feedback


# --- check_bu: BU-Gate (Spec-Validierung gegen echtes Schema) -------------

def _bu_workdir(tmp_path, target_col="col_4"):
    w = str(tmp_path / "work")
    os.makedirs(os.path.join(w, "bu"), exist_ok=True)
    os.makedirs(os.path.join(w, "data"), exist_ok=True)
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.1, 0.2, 0.3, 0.4],
                       target_col: ["a", "b", "a", "b"]})
    df.to_parquet(os.path.join(w, "data", "train.parquet"), index=False)
    return w

def _write_spec(w, spec):
    json.dump(spec, open(os.path.join(w, "bu", "task_spec.json"), "w"))

def test_check_bu_passes_on_valid_spec(tmp_path):
    w = _bu_workdir(tmp_path)
    _write_spec(w, {"task_type": "classification", "target_column": "col_4",
                    "metric": "f1", "feature_columns": ["f1", "f2"]})
    res = check_bu(w)
    assert res.passed is True and res.abstain is False
    assert res.metrics["spec"]["target_column"] == "col_4"

def test_check_bu_missing_file_fails(tmp_path):
    w = _bu_workdir(tmp_path)
    res = check_bu(w)
    assert res.passed is False and res.abstain is False
    assert "task_spec.json" in res.feedback

def test_check_bu_invalid_json_fails(tmp_path):
    w = _bu_workdir(tmp_path)
    open(os.path.join(w, "bu", "task_spec.json"), "w").write("{kaputt")
    res = check_bu(w)
    assert res.passed is False and res.abstain is False

def test_check_bu_cannot_determine_is_abstain_not_fail(tmp_path):
    w = _bu_workdir(tmp_path)
    _write_spec(w, {"cannot_determine": True, "reason": "Zielgröße existiert nicht in den Daten",
                    "missing_info": ["Überlebensdauer-Spalte"]})
    res = check_bu(w)
    assert res.passed is False and res.abstain is True
    assert "existiert nicht" in res.feedback

def test_check_bu_hallucinated_target_fails(tmp_path):
    w = _bu_workdir(tmp_path)
    _write_spec(w, {"task_type": "classification", "target_column": "survival_months", "metric": "f1"})
    res = check_bu(w)
    assert res.passed is False and res.abstain is False
    assert "survival_months" in res.feedback

def test_check_bu_unsupported_task_type_fails(tmp_path):
    w = _bu_workdir(tmp_path)
    _write_spec(w, {"task_type": "regression", "target_column": "col_4", "metric": "rmse"})
    res = check_bu(w)
    assert res.passed is False

def test_check_bu_invalid_metric_fails(tmp_path):
    w = _bu_workdir(tmp_path)
    _write_spec(w, {"task_type": "classification", "target_column": "col_4", "metric": "rmse"})
    res = check_bu(w)
    assert res.passed is False and "rmse" in res.feedback

def test_check_bu_continuous_target_rejected(tmp_path):
    # Typ-Konsistenz (bu-gate.md §3): kontinuierliche Spalte als Classification-Target -> REJECT
    w = str(tmp_path / "work")
    os.makedirs(os.path.join(w, "bu"), exist_ok=True)
    os.makedirs(os.path.join(w, "data"), exist_ok=True)
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": range(100), "col_1": rng.random(100)})
    df.to_parquet(os.path.join(w, "data", "train.parquet"), index=False)
    _write_spec(w, {"task_type": "classification", "target_column": "col_1", "metric": "f1"})
    res = check_bu(w)
    assert res.passed is False and res.abstain is False

def test_check_bu_highcardinality_int_target_rejected(tmp_path):
    # Typ-Konsistenz (bu-gate.md §3): hochkardinale Integer-Spalte (z.B. ID) als
    # Classification-Target -> REJECT (nicht nur Floats fangen).
    w = str(tmp_path / "work")
    os.makedirs(os.path.join(w, "bu"), exist_ok=True)
    os.makedirs(os.path.join(w, "data"), exist_ok=True)
    df = pd.DataFrame({"f1": [0.1] * 100,
                       "patient_id": pd.Series(range(100), dtype="int64")})
    df.to_parquet(os.path.join(w, "data", "train.parquet"), index=False)
    _write_spec(w, {"task_type": "classification", "target_column": "patient_id", "metric": "f1"})
    res = check_bu(w)
    assert res.passed is False and res.abstain is False


def test_check_bu_feature_columns_must_exist(tmp_path):
    w = _bu_workdir(tmp_path)
    _write_spec(w, {"task_type": "classification", "target_column": "col_4",
                    "metric": "f1", "feature_columns": ["f1", "ghost"]})
    res = check_bu(w)
    assert res.passed is False and "ghost" in res.feedback


# --- check_du: Faktentreue-Check (roadmap §1) ------------------------------

def _du_workdir(tmp_path):
    """train.parquet mit bekannten Fakten: 4 Zeilen, f1 numerisch (1 NaN),
    f2 kategorial (0 NaN), col_3 = Target (2x a, 2x b)."""
    w = str(tmp_path / "work")
    os.makedirs(os.path.join(w, "du"), exist_ok=True)
    os.makedirs(os.path.join(w, "data"), exist_ok=True)
    df = pd.DataFrame({"f1": [1.0, None, 3.0, 4.0],
                       "f2": ["x", "y", "x", "y"],
                       "col_3": ["a", "b", "a", "b"]})
    df.to_parquet(os.path.join(w, "data", "train.parquet"), index=False)
    return w

def _true_report(**overrides):
    report = {
        "n_rows": 4,
        "columns": {"f1": {"dtype": "numeric", "n_missing": 1},
                    "f2": {"dtype": "categorical", "n_missing": 0},
                    "col_3": {"dtype": "categorical", "n_missing": 0}},
        "class_balance": {"a": 2, "b": 2},
        "quality_issues": [],
        "excluded_columns": [],
    }
    report.update(overrides)
    return report

def _write_report(w, report):
    json.dump(report, open(os.path.join(w, "du", "data_report.json"), "w"))

def test_check_du_passes_on_faithful_report(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, _true_report())
    res = check_du(w, target="col_3")
    assert res.passed is True
    assert res.metrics["report"]["n_rows"] == 4

def test_check_du_missing_file_fails(tmp_path):
    w = _du_workdir(tmp_path)
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "data_report.json" in res.feedback

def test_check_du_invalid_json_fails(tmp_path):
    w = _du_workdir(tmp_path)
    open(os.path.join(w, "du", "data_report.json"), "w").write("{kaputt")
    res = check_du(w, target="col_3")
    assert res.passed is False

def test_check_du_hallucinated_n_rows_fails(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, _true_report(n_rows=999))
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "n_rows" in res.feedback

def test_check_du_wrong_missing_count_fails(tmp_path):
    w = _du_workdir(tmp_path)
    r = _true_report()
    r["columns"]["f1"]["n_missing"] = 0          # tatsächlich 1
    _write_report(w, r)
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "n_missing" in res.feedback and "f1" in res.feedback

def test_check_du_wrong_dtype_fails(tmp_path):
    w = _du_workdir(tmp_path)
    r = _true_report()
    r["columns"]["f2"]["dtype"] = "numeric"      # tatsächlich categorical
    _write_report(w, r)
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "dtype" in res.feedback and "f2" in res.feedback

def test_check_du_wrong_class_balance_fails(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, _true_report(class_balance={"a": 3, "b": 1}))
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "class_balance" in res.feedback

def test_check_du_int_class_balance_keys_normalized(tmp_path):
    """JSON-Keys sind Strings — bei int-kodiertem Target ('0'/'1') muss der
    Vergleich trotzdem klappen (str-Normalisierung beider Seiten)."""
    w = str(tmp_path / "work")
    os.makedirs(os.path.join(w, "du"), exist_ok=True)
    os.makedirs(os.path.join(w, "data"), exist_ok=True)
    pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]}).to_parquet(
        os.path.join(w, "data", "train.parquet"), index=False)
    _write_report(w, {"n_rows": 4,
                      "columns": {"f1": {"dtype": "numeric", "n_missing": 0},
                                  "target": {"dtype": "numeric", "n_missing": 0}},
                      "class_balance": {"0": 2, "1": 2},
                      "quality_issues": [], "excluded_columns": []})
    assert check_du(w, target="target").passed is True

def test_check_du_invented_column_fails(tmp_path):
    w = _du_workdir(tmp_path)
    r = _true_report()
    r["columns"]["ghost"] = {"dtype": "numeric", "n_missing": 0}
    _write_report(w, r)
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "ghost" in res.feedback

def test_check_du_omitted_column_fails(tmp_path):
    w = _du_workdir(tmp_path)
    r = _true_report()
    del r["columns"]["f2"]
    _write_report(w, r)
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "f2" in res.feedback

def test_check_du_excluded_ghost_column_fails(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, _true_report(excluded_columns=[{"column": "ghost", "reason": "ID"}]))
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "ghost" in res.feedback

def test_check_du_excluding_target_fails(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, _true_report(excluded_columns=[{"column": "col_3", "reason": "weg damit"}]))
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "col_3" in res.feedback

def test_check_du_valid_exclusion_passes_and_surfaces_report(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, _true_report(excluded_columns=[{"column": "f2", "reason": "konstant-ähnlich"}]))
    res = check_du(w, target="col_3")
    assert res.passed is True
    assert res.metrics["report"]["excluded_columns"][0]["column"] == "f2"

def test_check_du_spec_mismatch_sets_flag_not_pass(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, {"spec_mismatch": True, "spec_mismatch_reason": "Zielspalte passt nicht zur Aufgabe"})
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert res.metrics.get("spec_mismatch") is True
    assert "passt nicht" in res.feedback

def test_check_du_missing_required_fields_fails(tmp_path):
    w = _du_workdir(tmp_path)
    _write_report(w, {"n_rows": 4})
    res = check_du(w, target="col_3")
    assert res.passed is False
    assert "Required fields" in res.feedback

def test_check_bu_ignores_critical_columns(tmp_path):
    """critical_columns ist Info für DU, NICHT pass-kritisch (roadmap §1) —
    auch eine Ghost-Spalte darin darf das Gate nicht failen."""
    w = _bu_workdir(tmp_path)
    _write_spec(w, {"task_type": "classification", "target_column": "col_4",
                    "metric": "f1", "critical_columns": ["ghost_id"]})
    assert check_bu(w).passed is True


# --- check_eval mit Baseline-Vergleich (gepaarter Bootstrap, 23.09.) ---------

def _baseline(tmp_path, values):
    p = str(tmp_path / "baseline_predictions.parquet")
    pd.DataFrame({"prediction": values}).to_parquet(p, index=False)
    return p


def test_check_eval_noninferior_to_baseline_passes(tmp_path):
    """Ein Fehler auf 36 Zeilen gegen eine perfekte Baseline (wine-Fall) ist kein
    Befund -> Pass, obwohl F1 punktweise unter baseline_f1 liegt."""
    y = ["a"] * 12 + ["b"] * 12 + ["c"] * 12
    agent = list(y); agent[0] = "b"
    w, h = _eval_setup(tmp_path, y, agent)
    res = check_eval(w, holdout_dir=h, target="target", threshold=1.0,
                     baseline_pred_path=_baseline(tmp_path, y))
    assert res.passed is True
    assert res.metrics["f1"] < res.metrics["baseline_f1"] == 1.0
    assert res.metrics["noninferior"] is True and res.metrics["ci_hi"] >= 0


def test_check_eval_significantly_worse_than_baseline_fails(tmp_path):
    y = ["a", "b"] * 100
    agent = ["a"] * 200                                   # Majority-Raten
    w, h = _eval_setup(tmp_path, y, agent)
    res = check_eval(w, holdout_dir=h, target="target", threshold=0.0,
                     baseline_pred_path=_baseline(tmp_path, y))
    assert res.passed is False                            # Schwelle 0.0 zaehlt nicht mehr
    assert "baseline" in res.feedback and "CI" in res.feedback
    assert "form_error" not in res.metrics                # Modellqualitaet -> retry model
    assert res.metrics["noninferior"] is False


def test_check_eval_form_error_wins_over_baseline(tmp_path):
    w, h = _eval_setup(tmp_path, ["a", "b"], ["a"])
    res = check_eval(w, holdout_dir=h, target="target",
                     baseline_pred_path=_baseline(tmp_path, ["a", "b"]))
    assert res.passed is False and res.metrics.get("form_error") is True


def _model_with_metrics(tmp_path, metrics, spec_metric=None):
    w = _workdir(tmp_path)
    clf = DummyClassifier(strategy="most_frequent").fit([[0]], [0])
    with open(os.path.join(w, "model", "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    json.dump(metrics, open(os.path.join(w, "model", "metrics.json"), "w"))
    if spec_metric is not None:
        os.makedirs(os.path.join(w, "bu"), exist_ok=True)
        json.dump({"task_type": "classification", "target_column": "t", "metric": spec_metric},
                  open(os.path.join(w, "bu", "task_spec.json"), "w"))
    return w


def test_check_model_compares_in_spec_metric_not_accuracy(tmp_path):
    """R2-5: ein auf F1 optimiertes Modell mit Accuracy unter der Mehrheitsklasse besteht,
    wenn es in der Spec-Metrik (f1) ueber dem Dummy liegt."""
    w = _model_with_metrics(tmp_path, {"metric": "f1", "val_score": 0.71, "dummy_score": 0.47,
                                       "val_accuracy": 0.84, "dummy_accuracy": 0.88}, "f1")
    assert check_model(w).passed is True


def test_check_model_rejects_metric_other_than_spec(tmp_path):
    """R2-5: kein Ausweichen auf eine andere Metrik als in bu/task_spec.json."""
    w = _model_with_metrics(tmp_path, {"metric": "accuracy", "val_score": 0.9,
                                       "dummy_score": 0.6}, "f1")
    res = check_model(w)
    assert res.passed is False
    assert "uses 'f1'" in res.feedback
