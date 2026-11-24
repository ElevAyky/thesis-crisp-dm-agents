# src/crispfsm/scoring.py
"""Deterministischer Scorer (Spec 2026-07-18 §1.2): rechnet Stufe 2 (F1/Accuracy)
aus einer Predictions-Datei gegen holdout/test.parquet. Reine Messung — kein
LLM-Zugriff, keine Seiteneffekte. Self-Report der Worker zählt nicht (MLR-Bench:
freie Agents fabrizieren Ergebnisse)."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Gepaarter Bootstrap gegen die Baseline (23.09.): fest, damit jede Gate-Pruefung
# deterministisch ist (gleiche Predictions -> gleiches Urteil, auch im Retry).
N_BOOT = 2000
BOOT_SEED = 42
ALPHA = 0.05


@dataclass
class ScoreResult:
    ok: bool                  # Predictions vorhanden, wohlgeformt, Zeilenzahl passt
    feedback: str = ""        # Diagnose bei ok=False (Gate-Feedback-tauglich)
    f1: float | None = None   # macro (Parität zur Baseline-Schwere, s. u.)
    accuracy: float | None = None


def _normalize(truth: pd.Series, pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Typ-Normierung: Labels sind je nach Neutralisierung Strings ('malignant')
    # oder Ints (0/1); Parquet-dtypes der Worker variieren. Sind BEIDE Seiten
    # numerisch, numerisch vergleichen (sonst mappt astype(str) Int-Label 1 auf
    # "1", Float-Prediction 1.0 aber auf "1.0" → stilles F1=0.0). Sonst str-Fallback.
    if pd.api.types.is_numeric_dtype(truth) and pd.api.types.is_numeric_dtype(pred):
        return truth, pred.astype(truth.dtype)
    return truth.astype(str), pred.astype(str)


def _score(loaded) -> ScoreResult:
    if isinstance(loaded, ScoreResult):
        return loaded
    y_true, y_pred = loaded
    # macro-F1, nicht weighted (Umstellung 18.09., Decker 14.09.): die Erfolgs-
    # schwelle ist baseline-relativ (k = 1) und scripts/baseline_difficulty.py
    # misst macro-F1 — Harness und Schwelle müssen dieselbe Größe messen. Auf
    # unbalancierten Sets liegt weighted deutlich über macro (Majority-Raten
    # sähe sonst "erfolgreich" aus). Runs vor dem 18.09. tragen weighted-F1.
    return ScoreResult(
        True,
        f1=float(f1_score(y_true, y_pred, average="macro")),
        accuracy=float(accuracy_score(y_true, y_pred)),
    )


def score_predictions(pred_path: str, holdout_test_path: str, *, target: str) -> ScoreResult:
    """Predictions-Datei ist zeilengleich mit holdout_test_path (Holdout ohne Validierungs-
    anteil). Fuer Holdouts mit eval_index.json: score_part."""
    return _score(_load_labels(pred_path, holdout_test_path, target=target))


# --- Holdout mit Validierungsanteil (S1, 23.09.) -------------------------------
# Die Feature-Datei der Agenten enthaelt Validierungs- und Testzeilen gemischt;
# holdout/eval_index.json kennt die Positionen. Gates von A/B werten part="val" aus,
# die finale Bewertung aller Arme part="test".

EVAL_INDEX_FILE = "eval_index.json"   # Spiegel von datasets.EVAL_INDEX_FILE


def eval_index(holdout_dir: str) -> dict | None:
    path = os.path.join(holdout_dir, EVAL_INDEX_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gate_part(holdout_dir: str) -> str:
    """Anteil, auf dem Gates messen: 'val', wenn es einen Validierungsanteil gibt, sonst
    'test' (Holdout ohne Validierung, z. B. Unit-Tests)."""
    return "val" if eval_index(holdout_dir) is not None else "test"


def score_part(pred_path: str, holdout_dir: str, part: str, *, target: str) -> ScoreResult:
    return _score(_load_part(pred_path, holdout_dir, part, target=target))


def _read_prediction_column(pred_path: str) -> pd.Series | ScoreResult:
    if not os.path.exists(pred_path):
        return ScoreResult(False, feedback=(
            f"'{os.path.basename(pred_path)}' is missing — no predictions delivered."))
    try:
        preds = pd.read_parquet(pred_path)
    except Exception as e:
        return ScoreResult(False, feedback=f"Predictions file not readable (not a valid Parquet): {e}")
    if "prediction" not in preds.columns:
        return ScoreResult(False, feedback=(
            f"Column 'prediction' is missing in the predictions file "
            f"(present: {list(preds.columns)})."))
    return preds["prediction"]


def _check_length_and_nan(pred: pd.Series, n_expected: int) -> ScoreResult | None:
    if len(pred) != n_expected:
        return ScoreResult(False, feedback=(
            f"Row count mismatch: {len(pred)} predictions delivered, expected {n_expected} "
            f"(one per row of test_features.parquet, same order)."))
    n_nan = int(pred.isna().sum())
    if n_nan:
        return ScoreResult(False, feedback=f"Predictions contain NaN ({n_nan} rows).")
    return None


def _load_part(pred_path: str, holdout_dir: str, part: str, *,
               target: str) -> ScoreResult | tuple[pd.Series, pd.Series]:
    """Wie _load_labels, aber fuer einen Anteil der gemeinsamen Feature-Datei. Form-Checks
    (Existenz, Spalte, Zeilenzahl, NaN) gelten fuer die GANZE Datei — der Kontrakt ist
    derselbe, egal welcher Anteil gerade ausgewertet wird."""
    idx = eval_index(holdout_dir)
    if idx is None:
        if part != "test":
            raise ValueError(f"Holdout {holdout_dir} hat keinen Validierungsanteil ('{part}')")
        return _load_labels(pred_path, os.path.join(holdout_dir, "test.parquet"), target=target)
    pred = _read_prediction_column(pred_path)
    if isinstance(pred, ScoreResult):
        return pred
    bad = _check_length_and_nan(pred, len(idx["val"]) + len(idx["test"]))
    if bad is not None:
        return bad
    truth = pd.read_parquet(os.path.join(holdout_dir, f"{part}.parquet"))[target]
    if len(truth) != len(idx[part]):
        # Holdout passt nicht mehr zum Index (z. B. von einem Worker ueberschrieben) —
        # nicht scorbar statt Absturz; run.py markiert den Run ueber holdout_intact.
        return ScoreResult(False, feedback=(
            f"Holdout '{part}' is inconsistent with the evaluation index "
            f"({len(truth)} rows vs. {len(idx[part])})."))
    return _normalize_and_check_labels(truth.reset_index(drop=True),
                                       pred.iloc[idx[part]].reset_index(drop=True))


def _load_labels(pred_path: str, holdout_test_path: str, *,
                 target: str) -> ScoreResult | tuple[pd.Series, pd.Series]:
    """Form-Checks + Typ-Normierung. Formfehler -> ScoreResult(ok=False), sonst
    (y_true, y_pred) — gemeinsame Basis fuer score_predictions und den Baseline-Vergleich."""
    pred = _read_prediction_column(pred_path)
    if isinstance(pred, ScoreResult):
        return pred
    truth = pd.read_parquet(holdout_test_path)[target]
    bad = _check_length_and_nan(pred, len(truth))
    if bad is not None:
        return bad
    return _normalize_and_check_labels(truth, pred)


def _normalize_and_check_labels(truth: pd.Series, pred: pd.Series
                                ) -> ScoreResult | tuple[pd.Series, pd.Series]:
    y_true, y_pred = _normalize(truth, pred)
    # Disjunkter Label-Raum = Formfehler, kein Modell-Score (Scan-Befund 08.08.:
    # Worker lieferten Int-Codes statt 'true'/'false' bzw. Wahrscheinlichkeiten
    # statt Labels — das lief still in F1=0.0 durch und verfälschte die Messung).
    # Teilweise Überlappung scored normal (einzelne unbekannte Labels zählen als
    # falsch); nur der Komplett-Mismatch ist eindeutig ein Format-Problem.
    pred_labels = set(map(str, pd.unique(y_pred)))
    true_labels = set(map(str, pd.unique(y_true)))
    if not (pred_labels & true_labels):
        return ScoreResult(False, feedback=(
            f"Prediction labels do not match the target label space: delivered e.g. "
            f"{sorted(pred_labels)[:5]}, expected values from {sorted(true_labels)[:5]} — "
            f"deliver class labels of the target column, not codes/probabilities."))
    return y_true, y_pred


# --- Erfolgskriterium: nicht signifikant schlechter als die Baseline ---------
#
# Entscheidung 23.09. (ersetzt die Punkt-Schwelle F1 >= baseline_f1): Agent und
# Default-Baseline werden auf DENSELBEN Holdout-Zeilen verglichen; per gepaartem
# Bootstrap wird das 95-%-KI der Differenz F1_Agent - F1_Baseline geschaetzt.
# SUCCESS <=> obere KI-Grenze >= 0 ("nicht signifikant schlechter"). Die Toleranz
# skaliert so mit der Holdout-Groesse (wine n=36: ein Fehler ist kein Befund;
# covertype n=116k: praktisch k = 1), und die gemeinsame Test-Schwere kuerzt sich
# heraus, weil beide auf identischen Ziehungen gemessen werden. Nie strenger als
# die Punkt-Regel: ist der Agent punktweise >= Baseline, ist er auch nicht-unterlegen.


@dataclass(frozen=True)
class PairedDiff:
    diff: float          # Punktschaetzung F1_agent - F1_baseline (volle Stichprobe)
    ci_lo: float
    ci_hi: float
    noninferior: bool    # ci_hi >= 0


@dataclass(frozen=True)
class BaselineComparison:
    f1: float
    baseline_f1: float
    diff: float
    ci_lo: float
    ci_hi: float
    noninferior: bool


def _macro_f1_batch(cms: np.ndarray) -> np.ndarray:
    """Macro-F1 fuer einen Stapel Konfusionsmatrizen (..., K, K; Zeilen = wahr).
    sklearn-Semantik: nur Klassen aus y_true ∪ y_pred zaehlen, F1 = 2tp/(2tp+fp+fn)."""
    tp = np.diagonal(cms, axis1=-2, axis2=-1)
    fp = cms.sum(axis=-2) - tp
    fn = cms.sum(axis=-1) - tp
    denom = 2 * tp + fp + fn
    present = denom > 0
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=present)
    n_present = present.sum(axis=-1)
    return np.divide(per_class.sum(axis=-1), n_present,
                     out=np.zeros(per_class.shape[:-1]), where=n_present > 0)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    return float(_macro_f1_batch(np.asarray(cm, dtype=float)))


def paired_bootstrap_f1_diff(y_true, y_agent, y_base, *, n_boot: int = N_BOOT,
                             seed: int = BOOT_SEED, alpha: float = ALPHA) -> PairedDiff:
    """Gepaarter Bootstrap der macro-F1-Differenz Agent - Baseline.

    Statt Zeilen neu zu ziehen: Zeilen nach (wahr, agent, baseline)-Zelle zaehlen,
    pro Replikat eine Multinomial-Ziehung ueber die Zellen — verteilungsgleich zum
    Zeilen-Resampling, aber O(Zellen) statt O(n) (Holdouts bis 138k Zeilen)."""
    y_true, y_agent, y_base = (np.asarray(v).astype(str) for v in (y_true, y_agent, y_base))
    n = len(y_true)
    labels, inv = np.unique(np.concatenate([y_true, y_agent, y_base]), return_inverse=True)
    K = len(labels)
    t, a, b = inv[:n], inv[n:2 * n], inv[2 * n:]
    cells, counts = np.unique((t * K + a) * K + b, return_counts=True)
    ct, rest = np.divmod(cells, K * K)
    ca, cb = np.divmod(rest, K)
    # Zelle -> Position in der (wahr x agent)- bzw. (wahr x baseline)-Matrix
    to_a = np.zeros((len(cells), K * K)); to_a[np.arange(len(cells)), ct * K + ca] = 1
    to_b = np.zeros((len(cells), K * K)); to_b[np.arange(len(cells)), ct * K + cb] = 1

    def diff_of(weights: np.ndarray) -> np.ndarray:
        cm_a = (weights @ to_a).reshape(-1, K, K)
        cm_b = (weights @ to_b).reshape(-1, K, K)
        return _macro_f1_batch(cm_a) - _macro_f1_batch(cm_b)

    point = float(diff_of(counts[None, :].astype(float))[0])
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=n_boot).astype(float)
    lo, hi = np.quantile(diff_of(draws), [alpha / 2, 1 - alpha / 2])
    return PairedDiff(diff=point, ci_lo=float(lo), ci_hi=float(hi), noninferior=bool(hi >= 0))


def compare_to_baseline(pred_path: str, holdout_test_path: str, baseline_pred_path: str,
                        *, target: str) -> BaselineComparison | None:
    """Agent-Predictions vs. Baseline-Predictions auf demselben Holdout. None, wenn
    die Agent-Predictions formal nicht scorbar sind (Formfehler entscheidet vorher)."""
    return _compare(_load_labels(pred_path, holdout_test_path, target=target), baseline_pred_path)


def compare_part_to_baseline(pred_path: str, holdout_dir: str, part: str,
                             baseline_pred_path: str, *, target: str) -> BaselineComparison | None:
    """Baseline-Vergleich auf einem Anteil (S1). baseline_pred_path muss zeilengleich mit
    holdout/<part>.parquet sein (baseline_preds/<dataset>.<part>.parquet)."""
    return _compare(_load_part(pred_path, holdout_dir, part, target=target), baseline_pred_path)


def _compare(loaded, baseline_pred_path: str) -> BaselineComparison | None:
    if isinstance(loaded, ScoreResult):
        return None
    y_true, y_agent = loaded
    base = pd.read_parquet(baseline_pred_path)["prediction"]
    if len(base) != len(y_true):
        raise ValueError(f"Baseline-Predictions ({len(base)} Zeilen) passen nicht zum "
                         f"Holdout ({len(y_true)} Zeilen): {baseline_pred_path}")
    if pd.api.types.is_numeric_dtype(y_true) and not pd.api.types.is_numeric_dtype(base):
        # Baseline-Datei speichert Labels als Strings; numerische Wahrheit (nur im
        # nicht neutralisierten Skelett) -> auf denselben Typ bringen wie y_agent.
        base = pd.to_numeric(base, errors="coerce")
    _, y_base = _normalize(y_true, base)
    r = paired_bootstrap_f1_diff(y_true, y_agent, y_base)
    return BaselineComparison(
        f1=float(f1_score(y_true, y_agent, average="macro")),
        baseline_f1=float(f1_score(y_true, y_base, average="macro")),
        diff=r.diff, ci_lo=r.ci_lo, ci_hi=r.ci_hi, noninferior=r.noninferior)


def baseline_f1_on_holdout(baseline_pred_path: str, holdout_test_path: str, *,
                           target: str) -> float:
    """macro-F1 der Baseline-Predictions auf dem Holdout — Ausrichtungs-Check beim
    Run-Setup: muss manifest.baseline_f1 treffen, sonst gehoert die Datei zu einem
    anderen Split (falsche Zeilenreihenfolge/Version) und der Vergleich waere Unsinn."""
    truth = pd.read_parquet(holdout_test_path)[target]
    base = pd.read_parquet(baseline_pred_path)["prediction"]
    if len(base) != len(truth):
        raise ValueError(f"Baseline-Predictions ({len(base)} Zeilen) passen nicht zum "
                         f"Holdout ({len(truth)} Zeilen): {baseline_pred_path}")
    if pd.api.types.is_numeric_dtype(truth) and not pd.api.types.is_numeric_dtype(base):
        base = pd.to_numeric(base, errors="coerce")
    y_true, y_base = _normalize(truth, base)
    return float(f1_score(y_true, y_base, average="macro"))
