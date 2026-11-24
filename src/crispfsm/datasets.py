# src/crispfsm/datasets.py
from __future__ import annotations
import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

from crispfsm.config import VAL_FRACTION

# holdout/eval_index.json: Positionen der Validierungs- und Testzeilen in der gemeinsamen
# Feature-Datei (holdout/test_features.parquet). Fehlt die Datei, ist der Holdout ohne
# Validierungsanteil erzeugt (val_size=0) und die Feature-Datei enthaelt nur Testzeilen.
EVAL_INDEX_FILE = "eval_index.json"

_LOADERS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
}

# Getarnte Namen für die injizierte Leakage-Spalte — im Stil der echten
# Feature-Namen des jeweiligen Datasets, damit die Spalte nicht schon per
# Namensschema auffällt (ein col_*-Name neben dem col_*-Target wäre semantisch
# lesbar). Fallback für die sklearn-Builtins; OpenML-Sets tragen den Namen im
# Manifest (leakage_column), das Manifest-Feld gewinnt auch für Builtins.
_LEAKAGE_COLUMNS = {
    "breast_cancer": "mean nuclear density",
    "wine": "proline_ratio",
}

# Leakage-Batch 2 "spurious" (docs/leakage-runs.md §7): Klassenzentren + Normalrauschen,
# so dass die Spalte allein im Train ~85 % trifft (Naechstes-Zentrum, gleich grosse Klassen).
SPURIOUS_BASE = 15.0
LEAK_SPACING = 5.53
SPURIOUS_ACCURACY = 0.85


def spurious_sd(n_classes: int) -> float:
    """sigma, bei dem die Naechstes-Zentrum-Zuordnung SPURIOUS_ACCURACY trifft: Randklassen
    Phi(a), innere 2 Phi(a) - 1 mit a = Abstand / 2 sigma -> nach Phi(a) aufgeloest."""
    from scipy.stats import norm
    k = n_classes
    a = norm.ppf((SPURIOUS_ACCURACY * k + k - 2) / (2 * (k - 1)))
    return LEAK_SPACING / (2 * a)

def kaggle_cache_path(competition: str) -> str:
    """Lokaler Cache fuer Kaggle-Competition-Daten. Der Harness laedt nie selbst
    aus dem Netz (kein Kaggle-Token/-Paket im Run noetig) — befuellt wird der
    Cache einmalig per scripts/fetch_kaggle.py. CRISPFSM_DATA_DIR ueberschreibt."""
    root = os.environ.get("CRISPFSM_DATA_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "crispfsm")
    return os.path.join(root, "kaggle", competition, "train.csv")


def _load_kaggle(name: str, competition: str | None, source_target: str | None,
                 drop_columns, data_sha256: str | None) -> tuple[pd.DataFrame, pd.Series]:
    if not competition:
        raise ValueError(f"dataset '{name}': source=kaggle braucht kaggle_competition")
    if not source_target:
        raise ValueError(f"dataset '{name}': source=kaggle braucht die Zielspalte (source_target)")
    path = kaggle_cache_path(competition)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dataset '{name}': {path} fehlt — einmalig laden mit "
            f"`uvx --from kaggle python scripts/fetch_kaggle.py {competition}`")
    if data_sha256:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != data_sha256:
            raise ValueError(f"dataset '{name}': sha256 von {path} ist {digest}, "
                             f"Manifest pinnt {data_sha256} — Kaggle-Daten geaendert?")
    df = pd.read_csv(path)
    df = df.drop(columns=list(drop_columns or ()))
    y = df.pop(source_target).astype(str)
    return df, y


def load_frame(name: str, *, source: str = "sklearn",
               openml_data_id: int | None = None,
               kaggle_competition: str | None = None,
               source_target: str | None = None,
               drop_columns=(),
               data_sha256: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Rohdaten (X, y) ohne Split/Neutralisierung, y mit semantischen Klassen-Labels
    als plain strings — für prepare_dataset (OpenML-/Kaggle-Pfad) und Analyse-Skripte
    (scripts/baseline_difficulty.py)."""
    if source == "openml":
        if openml_data_id is None:
            raise ValueError(f"dataset '{name}': source=openml braucht openml_data_id")
        bunch = fetch_openml(data_id=openml_data_id, as_frame=True)
        # astype(str) normalisiert den Categorical-Dtype des OpenML-Targets —
        # Parität zum Builtin-Pfad (Labels als plain strings).
        return bunch.data, bunch.target.astype(str)
    if source == "kaggle":
        return _load_kaggle(name, kaggle_competition, source_target, drop_columns, data_sha256)
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset: {name}")
    bunch = _LOADERS[name]()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series([bunch.target_names[i] for i in bunch.target], name="target")
    return X, y


def prepare_dataset(name: str, workdir: str, *, target: str = "target",
                    neutralize_target: bool = False,
                    inject_leakage: bool = False,
                    test_size: float = 0.2, seed: int = 42,
                    source: str = "sklearn",
                    openml_data_id: int | None = None,
                    leakage_column: str | None = None,
                    kaggle_competition: str | None = None,
                    source_target: str | None = None,
                    drop_columns=(),
                    data_sha256: str | None = None,
                    val_size: float = 0.0,
                    leak_variant: str = "perfect") -> dict:
    """Lädt ein sklearn-Builtin oder ein OpenML-Set (source='openml' + Dataset-ID,
    beides aus dem Manifest), splittet, schreibt train ins work/-Verzeichnis
    und hält test in holdout/ zurück (außerhalb der Worker-cwd).

    neutralize_target: benennt die Zielspalte in 'col_<n_features>' um und mappt die
    Werte auf semantische Klassen-Labels (bu-gate.md §5): BU soll das Target semantisch
    finden, nicht per String-Match auf 'target'/'class'."""
    if source in ("openml", "kaggle"):
        X, y_sem = load_frame(name, source=source, openml_data_id=openml_data_id,
                              kaggle_competition=kaggle_competition,
                              source_target=source_target, drop_columns=drop_columns,
                              data_sha256=data_sha256)
        target_column = target
        if neutralize_target:
            target_column = f"col_{X.shape[1]}"
        # OpenML-/Kaggle-Targets tragen bereits semantische Klassen-Labels ('good'/'bad');
        # neutralisiert wird nur der Spaltenname.
        y = pd.Series(y_sem.to_numpy(), name=target_column, index=X.index)
    elif name in _LOADERS:
        bunch = _LOADERS[name]()
        X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        target_column = target
        y_values = bunch.target
        if neutralize_target:
            target_column = f"col_{X.shape[1]}"
            y_values = [bunch.target_names[i] for i in bunch.target]
        y = pd.Series(y_values, name=target_column)
    else:
        raise KeyError(f"unknown dataset: {name}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    # Validierungsanteil des Harness (S1, 23.09.): aus dem Trainingsteil, NACH dem
    # Test-Split (Test-Zeilen bleiben identisch zu val_size=0). Kein Agent sieht ihn als
    # solchen; nur die Gates von Arm A/B messen darauf, damit ihr Feedback nie aus dem
    # finalen Test-Set stammt. Alle Arme trainieren auf demselben (verkleinerten) Train.
    X_val = y_val = None
    if val_size > 0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=val_size / (1 - test_size), random_state=seed,
            stratify=y_tr)

    # 3. Vergiftungstyp (roadmap §1, Decker 20.07.): injizierte Leakage-Spalte —
    # im Train perfekt mit dem Target korreliert, im Test zufällig (seeded
    # Permutation, gleiche Randverteilung). Tarnung: getarnter Feature-Name
    # (_LEAKAGE_COLUMNS) und Klassencodes affin auf Floats gemappt — sonst wäre
    # sie die einzige 0/1-Int-Spalte zwischen Floats. Perfekte Korrelation
    # bleibt unter der injektiven Abbildung erhalten: die Erkennung (DU-Report,
    # excluded_columns) soll statistisch sein, nicht semantisch.
    if leak_variant not in ("perfect", "spurious"):
        raise ValueError(f"unbekannte leak_variant '{leak_variant}' (perfect | spurious)")
    leak_column = leak_map = None
    if inject_leakage and leak_variant == "perfect":
        leak_column = leakage_column or _LEAKAGE_COLUMNS.get(name)
        if leak_column is None:
            raise ValueError(f"dataset '{name}': inject_leakage braucht "
                             "leakage_column im Manifest")
        assert leak_column not in X.columns
        rng = np.random.default_rng(seed)
        # Klassencodes in der TRAIN-Reihenfolge fuer alle Anteile (Review 24.10., M3):
        # factorize pro Split vertauschte sonst die Zuordnung Wert -> Klasse zwischen
        # Train und Test — ein Verteilungssprung als zusaetzlicher Hinweis auf den Leak.
        train_classes = pd.factorize(y_tr)[1]
        _leak_values = lambda y_split: 2.31 + 5.53 * pd.Categorical(
            y_split, categories=train_classes).codes
        # Zuordnung Leak-Wert -> Klasse, wie sie im TRAIN gilt (= was ein Modell lernt,
        # das dem Leak folgt). Grundlage der Leak-Messung auf dem Test (leakage.py).
        leak_map = [[float(2.31 + 5.53 * k), str(c)] for k, c in enumerate(train_classes)]
        X_tr = X_tr.assign(**{leak_column: _leak_values(y_tr)})
        X_te = X_te.assign(**{leak_column: rng.permutation(_leak_values(y_te))})
        if X_val is not None:   # Validierung wie Test: ungesehene Daten, Leak zufaellig
            X_val = X_val.assign(**{leak_column: rng.permutation(_leak_values(y_val))})
    elif inject_leakage:
        # Batch 2 "spurious" (docs/leakage-runs.md §7): fachfremder Name (Manifest
        # spurious_column, kein Builtin-Fallback), Train stark aber nicht perfekt korreliert
        # (Klassenzentrum + Rauschen, kontinuierlich), Validierung + Test gleich erzeugt
        # und permutiert -> dort wertlos wie oben. Erkennung soll semantisch sein.
        if not leakage_column:
            raise ValueError(f"dataset '{name}': leak_variant=spurious braucht "
                             "spurious_column im Manifest")
        leak_column = leakage_column
        assert leak_column not in X.columns
        rng = np.random.default_rng(seed)
        classes = pd.factorize(y_tr)[1]
        sd = spurious_sd(len(classes))

        def _noisy(y_split):
            codes = pd.Categorical(y_split, categories=classes).codes
            return np.round(SPURIOUS_BASE + LEAK_SPACING * codes
                            + rng.normal(0.0, sd, len(y_split)), 2)

        leak_map = [[float(SPURIOUS_BASE + LEAK_SPACING * k), str(c)]
                    for k, c in enumerate(classes)]
        X_tr = X_tr.assign(**{leak_column: _noisy(y_tr)})
        X_te = X_te.assign(**{leak_column: rng.permutation(_noisy(y_te))})
        if X_val is not None:
            X_val = X_val.assign(**{leak_column: rng.permutation(_noisy(y_val))})

    data_dir = os.path.join(workdir, "work", "data")
    holdout_dir = os.path.join(workdir, "holdout")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    pd.concat([X_tr, y_tr], axis=1).to_parquet(
        os.path.join(data_dir, "train.parquet"), index=False)
    pd.concat([X_te, y_te], axis=1).to_parquet(
        os.path.join(holdout_dir, "test.parquet"), index=False)
    if X_val is None:
        # Test-Features OHNE Labels: das Einzige, was Worker je vom Test-Split sehen
        # (Spec 2026-07-18 §1.1 — Labels verlassen holdout/ nie).
        X_te.to_parquet(os.path.join(holdout_dir, "test_features.parquet"), index=False)
    else:
        pd.concat([X_val, y_val], axis=1).to_parquet(
            os.path.join(holdout_dir, "val.parquet"), index=False)
        # EINE Feature-Datei fuer alle Arme: Validierungs- + Testzeilen, seeded gemischt.
        # Welche Zeile wohin gehoert, steht nur in holdout/eval_index.json (Positionen in
        # der Reihenfolge von val.parquet bzw. test.parquet). Vorhersage-Kontrakt bleibt:
        # eine Datei, eine Spalte, eine Zeile pro Feature-Zeile.
        combined = pd.concat([X_val, X_te], ignore_index=True)
        perm = np.random.default_rng(seed + 1).permutation(len(combined))
        combined.iloc[perm].reset_index(drop=True).to_parquet(
            os.path.join(holdout_dir, "test_features.parquet"), index=False)
        position_of = np.argsort(perm)          # Original-Zeile -> Position in der Datei
        n_val = len(X_val)
        with open(os.path.join(holdout_dir, EVAL_INDEX_FILE), "w", encoding="utf-8") as f:
            json.dump({"val": position_of[:n_val].tolist(),
                       "test": position_of[n_val:].tolist()}, f)

    return {
        "n_features": X.shape[1],
        "n_classes": int(y.nunique()),
        "n_train": len(X_tr),
        "n_val": 0 if X_val is None else len(X_val),
        "n_test": len(X_te),
        "target_column": target_column,
        "leakage_column": leak_column,
        "leakage_values": leak_map,
    }


def prepare_from_manifest(manifest, workdir: str, *, inject_leakage: bool = False,
                          leak_variant: str = "perfect") -> dict:
    """Run-Setup aus einem Manifest — die EINE Stelle, an der Split, Neutralisierung
    und Leakage-Injektion fuer einen Run festgelegt werden. run.py und die
    Holdout-Baseline (scripts/baseline_difficulty.py --holdout) rufen beide diese
    Funktion: Agent und Erfolgsschwelle werden so auf identischem Train/Holdout
    gemessen (Befund 23.09.: CV-Baseline vs. Einzel-Holdout war asymmetrisch)."""
    return prepare_dataset(manifest.dataset, workdir,
                           neutralize_target=True, inject_leakage=inject_leakage,
                           source=manifest.source,
                           openml_data_id=manifest.openml_data_id,
                           leakage_column=(manifest.spurious_column if leak_variant == "spurious"
                                           else manifest.leakage_column),
                           leak_variant=leak_variant,
                           kaggle_competition=manifest.kaggle_competition,
                           source_target=manifest.canonical_spec.target_column,
                           drop_columns=manifest.drop_columns,
                           data_sha256=manifest.data_sha256,
                           val_size=VAL_FRACTION)
