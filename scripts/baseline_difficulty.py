# scripts/baseline_difficulty.py
"""Empirische Schwere-Staffelung (benchmarks.md §2, roadmap §3): klassische
Default-Baselines per stratified 5-fold-CV, macro-F1 — deterministisch, LLM-frei,
reproduzierbar. Validiert die a-priori-Struktur-Heuristik der Schwere-Spalte
in benchmarks.md, BEVOR die Haupt-Runs starten (a-priori-Festschreibung).

Schwere-Konvention (a priori, benchmarks.md §2):
    best-Baseline-macro-F1 >= 0.95  -> leicht
    0.80 <= f1 < 0.95               -> mittel
    f1 < 0.80                       -> schwer

Zwei Messungen (23.09.):
- **CV (Default):** 5-fold-CV-Mittel -> Schwere/Terzile (Manifest: baseline_f1_cv).
- **--holdout:** Fit auf dem Run-Train-Split, Score auf dem Run-Holdout, erzeugt
  ueber datasets.prepare_from_manifest (dieselbe Funktion wie run.py) -> die
  ERFOLGSSCHWELLE (Manifest: baseline_f1). Grund: der Agent wird auf genau einem
  Holdout gemessen; ein CV-Mittel als Schwelle ist asymmetrisch — die Baseline
  bestand ihre eigene Schwelle nicht (phoneme 0.878 < 0.890, cnae_9 0.940 < 0.942).

Aufruf (aus dem Repo-Root):
    python scripts/baseline_difficulty.py                       # alle Manifeste
    python scripts/baseline_difficulty.py --datasets credit_g covertype
    python scripts/baseline_difficulty.py --max-rows 0          # ohne Subsample-Cap
    python scripts/baseline_difficulty.py --holdout             # Schwellen-Baseline
    python scripts/baseline_difficulty.py --holdout --write-manifests
    python scripts/baseline_difficulty.py --holdout --manifests-dir manifests/post_cutoff --write-manifests
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import tempfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from crispfsm.datasets import load_frame, prepare_from_manifest
from crispfsm.manifests import load_manifest

SEED = 42


def bucket(f1: float) -> str:
    if f1 >= 0.95:
        return "leicht"
    if f1 >= 0.80:
        return "mittel"
    return "schwer"


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(include=["object", "string", "category", "bool"]).columns
    num = X.columns.difference(cat)
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), list(num)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(cat)),
    ])


def score_dataset(name: str, manifests_dir: str, max_rows: int) -> dict:
    m = load_manifest(name, manifests_dir=manifests_dir)
    X, y = load_frame(m.dataset, source=m.source, openml_data_id=m.openml_data_id,
                      kaggle_competition=m.kaggle_competition,
                      source_target=m.canonical_spec.target_column,
                      drop_columns=m.drop_columns, data_sha256=m.data_sha256)
    subsampled = max_rows and len(X) > max_rows
    if subsampled:
        # stratifiziertes Subsample nur fürs Baseline-Ranking (Cap dokumentiert
        # in der Ausgabe) — die Runs selbst fahren immer das volle Set.
        X, _, y, _ = train_test_split(X, y, train_size=max_rows,
                                      stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    }
    scores = {}
    for label, clf in models.items():
        pipe = Pipeline([("prep", _preprocessor(X)), ("clf", clf)])
        scores[label] = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro").mean()
    best = max(scores["logreg"], scores["rf"])
    return {
        "dataset": m.dataset,
        "n": len(X), "subsampled": bool(subsampled),
        "n_features": X.shape[1], "n_classes": int(y.nunique()),
        "majority_share": y.value_counts(normalize=True).iloc[0],
        **{f"f1_{k}": v for k, v in scores.items()},
        "f1_best": best, "schwere": bucket(best),
    }


def score_holdout(name: str, manifests_dir: str) -> dict:
    """Schwellen-Baseline: LogReg/RF (Default) auf dem Run-Split. Der Split kommt
    aus prepare_from_manifest -> identisch zu jedem Agent-Run, by construction."""
    m = load_manifest(name, manifests_dir=manifests_dir)
    with tempfile.TemporaryDirectory() as tmp:
        info = prepare_from_manifest(m, tmp, inject_leakage=False)
        train = pd.read_parquet(os.path.join(tmp, "work", "data", "train.parquet"))
        parts = {p: pd.read_parquet(os.path.join(tmp, "holdout", f"{p}.parquet"))
                 for p in ("val", "test")}
    target = info["target_column"]
    X_tr, y_tr = train.drop(columns=[target]), train[target].astype(str)
    Xy = {p: (df.drop(columns=[target]), df[target].astype(str)) for p, df in parts.items()}
    scores = {p: {} for p in parts}
    preds = {p: {} for p in parts}
    for label, clf in {"logreg": LogisticRegression(max_iter=2000),
                       "rf": RandomForestClassifier(random_state=SEED, n_jobs=-1)}.items():
        pipe = Pipeline([("prep", _preprocessor(X_tr)), ("clf", clf)]).fit(X_tr, y_tr)
        for p, (X, y) in Xy.items():
            preds[p][label] = [str(v) for v in pipe.predict(X)]
            scores[p][label] = f1_score(y, preds[p][label], average="macro")
    # Referenzmodell auf der VALIDIERUNG gewaehlt (S1, 23.09.): keine Auswahl auf dem
    # finalen Test-Set, auch nicht fuer die Baseline. Dasselbe Modell liefert die
    # Referenz fuer das A/B-Gate (val) und fuer das Endurteil aller Arme (test).
    best = max(scores["val"], key=scores["val"].get)
    return {
        "dataset": m.dataset, "n_train": len(X_tr),
        "n_val": len(Xy["val"][0]), "n_test": len(Xy["test"][0]),
        "f1_logreg_holdout": scores["test"]["logreg"], "f1_rf_holdout": scores["test"]["rf"],
        "best_model": best,
        "f1_best_holdout": scores["test"][best], "f1_best_val": scores["val"][best],
        # Vergleichspartner fuer den gepaarten Bootstrap (scoring.compare_part_to_baseline)
        "predictions": preds["test"][best], "predictions_val": preds["val"][best],
        # CV-Wert fuer Schwere/Terzile: vor der Umstellung stand er in baseline_f1
        "f1_cv": m.baseline_f1_cv if m.baseline_f1_cv is not None else m.baseline_f1,
    }


def write_manifest_baseline(path: str, holdout_f1: float, cv_f1: float | None,
                            val_f1: float | None = None) -> None:
    """Setzt baseline_f1 (Test-Anteil = Referenz des Endurteils), baseline_f1_val
    (Validierungsanteil = Referenz der A/B-Gates) und baseline_f1_cv (Schwere) ans
    Manifest-Ende; bestehende Zeilen der Keys werden ersetzt (idempotent)."""
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines()
                 if not re.match(r"^baseline_f1(_cv|_val)?\s*:", line)]
    lines.append(f"baseline_f1: {holdout_f1:.3f}   # Referenzmodell (LogReg/RF default, auf val gewaehlt), "
                 f"macro-F1 auf dem TEST-Anteil (70/10/20, seed 42); Predictions in "
                 f"baseline_preds/<set>.test.parquet -> Endurteil aller Arme")
    if val_f1 is not None:
        lines.append(f"baseline_f1_val: {val_f1:.3f}   # dasselbe Modell auf dem VALIDIERUNGS-Anteil; "
                     f"baseline_preds/<set>.val.parquet -> Referenz der A/B-Gates")
    if cv_f1 is not None:
        lines.append(f"baseline_f1_cv: {cv_f1:.3f}   # 5-fold CV full-size (benchmarks.md par. 2) "
                     f"-> nur Schwere/Terzile, NICHT Schwelle")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_baseline_predictions(manifests_dir: str, dataset: str, predictions,
                               part: str = "test") -> str:
    """<manifests_dir>/baseline_preds/<dataset>.<part>.parquet, Spalte 'prediction', eine
    Zeile pro Zeile von holdout/<part>.parquet in dessen Reihenfolge. Versioniert im Repo,
    damit jeder Rechner dieselbe Referenz nutzt (RF/LogReg koennen plattformabhaengig
    minimal abweichen). run.py prueft beim Setup die Ausrichtung gegen baseline_f1(_val)."""
    folder = os.path.join(manifests_dir, "baseline_preds")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{dataset}.{part}.parquet")
    pd.DataFrame({"prediction": [str(v) for v in predictions]}).to_parquet(path, index=False)
    return path


def main_holdout(names: list[str], manifests_dir: str, write: bool) -> None:
    rows = []
    for name in names:
        print(f"[holdout] {name} ...", flush=True)
        rows.append(score_holdout(name, manifests_dir))
    if write:   # vor der Ausgabe: ein Konsolen-Encoding-Fehler darf die Messung nicht verwerfen
        for r in rows:
            write_manifest_baseline(os.path.join(manifests_dir, f"{r['dataset']}.yaml"),
                                    r["f1_best_holdout"], r["f1_cv"], r["f1_best_val"])
            write_baseline_predictions(manifests_dir, r["dataset"], r["predictions"], "test")
            write_baseline_predictions(manifests_dir, r["dataset"], r["predictions_val"], "val")
        print(f"{len(rows)} Manifeste aktualisiert ({manifests_dir})")
    # Ausgabe rein ASCII (Windows-Konsole cp1252)
    print("\n| Dataset | n train / val / test | LogReg (test) | RF (test) | Modell | "
          "**Referenz test** | Referenz val | CV (alt) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        cv = r["f1_cv"]
        cv_s = "-" if cv is None else f"{cv:.3f}"
        print(f"| {r['dataset']} | {r['n_train']} / {r['n_val']} / {r['n_test']} "
              f"| {r['f1_logreg_holdout']:.3f} | {r['f1_rf_holdout']:.3f} | {r['best_model']} "
              f"| **{r['f1_best_holdout']:.3f}** | {r['f1_best_val']:.3f} | {cv_s} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Manifest-Namen (Default: alle YAMLs in --manifests-dir)")
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="stratifiziertes Subsample-Cap für die CV (0 = volles Set)")
    ap.add_argument("--holdout", action="store_true",
                    help="Schwellen-Baseline auf dem Run-Split statt CV")
    ap.add_argument("--write-manifests", action="store_true",
                    help="mit --holdout: baseline_f1 (+ baseline_f1_cv) in die Manifeste schreiben")
    args = ap.parse_args()

    names = args.datasets or sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(args.manifests_dir, "*.yaml")))

    if args.holdout:
        main_holdout(names, args.manifests_dir, args.write_manifests)
        return

    rows = []
    for name in names:
        print(f"[baseline] {name} ...", flush=True)
        rows.append(score_dataset(name, args.manifests_dir, args.max_rows))

    print("\n| Dataset | n (CV) | Feat. | Klassen | Majority | Dummy-F1 | LogReg-F1 | RF-F1 | best | Schwere |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        n = f"{r['n']}{'*' if r['subsampled'] else ''}"
        print(f"| {r['dataset']} | {n} | {r['n_features']} | {r['n_classes']} "
              f"| {r['majority_share']:.2f} | {r['f1_dummy']:.3f} | {r['f1_logreg']:.3f} "
              f"| {r['f1_rf']:.3f} | {r['f1_best']:.3f} | **{r['schwere']}** |")
    if any(r["subsampled"] for r in rows):
        print(f"\n\\* stratifiziert subsampled auf --max-rows (Cap nur fürs "
              f"Baseline-Ranking; Runs fahren das volle Set)")


if __name__ == "__main__":
    main()
