"""Dataset-Manifeste (business-understanding-gate.md §2): task_prompt geht ans LLM,
canonical_spec ist Ground Truth fürs Messen — NIEMALS in einen Prompt interpolieren."""
from __future__ import annotations
import os
from dataclasses import dataclass
import yaml


@dataclass(frozen=True)
class CanonicalSpec:
    task_type: str
    target_column: str
    metric: str
    positive_class: str | None = None


@dataclass(frozen=True)
class PoisonedVariant:
    task_prompt: str
    expected: str = "ABSTAIN"


@dataclass(frozen=True)
class Manifest:
    dataset: str
    task_prompt: str
    canonical_spec: CanonicalSpec
    poisoned_variant: PoisonedVariant | None = None
    # source="openml" lädt via fetch_openml(data_id=openml_data_id) statt über
    # das sklearn-Builtin-Registry (datasets._LOADERS).
    source: str = "sklearn"
    openml_data_id: int | None = None
    # Getarnter Name der injizierten Leakage-Spalte (--leakage), im Stil der
    # echten Feature-Namen des Datasets. Für sklearn-Builtins existiert ein
    # Fallback in datasets._LEAKAGE_COLUMNS; für OpenML-Sets ist das Feld Pflicht,
    # sobald --leakage gefahren wird.
    leakage_column: str | None = None
    # Leakage-Batch 2 (--leakage spurious, docs/leakage-runs.md §7): fachfremd-absurder
    # Name im Schreibstil des Sets — erkennbar über den Inhalt, nicht über die Statistik.
    spurious_column: str | None = None
    # Best-Baseline macro-F1 (LogReg/RF, Default-Parameter) auf dem RUN-HOLDOUT:
    # Fit auf work/data/train.parquet, Score auf holdout/test.parquet — derselbe
    # Split wie jeder Agent-Run (datasets.prepare_from_manifest; 23.09., vorher
    # 5-fold-CV-Mittel = asymmetrische Messung). Basis der Erfolgsschwelle:
    # SUCCESS <=> macro-F1 >= THRESHOLD_K x baseline_f1 (k = 1, Decker 14.09.).
    # Nie aus Agent-Runs nachjustieren.
    baseline_f1: float | None = None
    # Dieselbe Baseline als 5-fold-CV-Mittel full-size (09.08.): nur Schwere-/
    # Terzil-Einteilung (benchmarks.md §2, Moderator UV4) — NICHT die Schwelle.
    baseline_f1_cv: float | None = None
    # Dasselbe Referenzmodell auf dem Validierungsanteil (S1): Referenz der A/B-Gates.
    baseline_f1_val: float | None = None
    # source="kaggle" (Post-Cutoff-Kontrolle, Decker 14.09.): train.csv der
    # Competition aus dem lokalen Cache (datasets.kaggle_cache_path, befuellt per
    # scripts/fetch_kaggle.py). drop_columns entfernt z. B. die Kaggle-`id`;
    # data_sha256 pinnt die Datei (Kaggle kann Daten nachtraeglich tauschen).
    kaggle_competition: str | None = None
    drop_columns: tuple[str, ...] = ()
    data_sha256: str | None = None


def _unit_interval(raw: dict, key: str, name: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    value = float(value)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"Manifest '{name}': {key} muss in (0, 1] liegen, ist {value}")
    return value


def load_manifest(name: str, manifests_dir: str = "manifests") -> Manifest:
    path = os.path.join(manifests_dir, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"kein Manifest für Dataset '{name}': {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    poisoned = raw.get("poisoned_variant")
    source = raw.get("source", "sklearn")
    openml_data_id = raw.get("openml_data_id")
    if source == "openml" and openml_data_id is None:
        raise ValueError(f"Manifest '{name}': source=openml braucht openml_data_id")
    if source == "kaggle" and not raw.get("kaggle_competition"):
        raise ValueError(f"Manifest '{name}': source=kaggle braucht kaggle_competition")
    baseline_f1 = _unit_interval(raw, "baseline_f1", name)
    baseline_f1_cv = _unit_interval(raw, "baseline_f1_cv", name)
    baseline_f1_val = _unit_interval(raw, "baseline_f1_val", name)
    return Manifest(
        dataset=raw["dataset"],
        task_prompt=raw["task_prompt"],
        canonical_spec=CanonicalSpec(**raw["canonical_spec"]),
        poisoned_variant=PoisonedVariant(**poisoned) if poisoned else None,
        source=source,
        openml_data_id=openml_data_id,
        leakage_column=raw.get("leakage_column"),
        spurious_column=raw.get("spurious_column"),
        baseline_f1=baseline_f1,
        baseline_f1_cv=baseline_f1_cv,
        baseline_f1_val=baseline_f1_val,
        kaggle_competition=raw.get("kaggle_competition"),
        drop_columns=tuple(raw.get("drop_columns") or ()),
        data_sha256=raw.get("data_sha256"),
    )
