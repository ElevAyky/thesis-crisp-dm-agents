import glob
import os

import pytest
from crispfsm.manifests import load_manifest, Manifest

ALL_MANIFESTS = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join("manifests", "*.yaml")))


def test_load_breast_cancer_manifest():
    m = load_manifest("breast_cancer")
    assert isinstance(m, Manifest)
    assert m.dataset == "breast_cancer"
    assert "tumor" in m.task_prompt
    assert m.canonical_spec.task_type == "classification"
    assert m.canonical_spec.metric == "f1"
    assert m.poisoned_variant is not None
    assert m.poisoned_variant.expected == "ABSTAIN"


def test_load_wine_manifest():
    m = load_manifest("wine")
    assert m.canonical_spec.task_type == "classification"


def test_unknown_manifest_raises():
    with pytest.raises(FileNotFoundError):
        load_manifest("does_not_exist")


def test_sklearn_manifest_defaults_source_sklearn():
    m = load_manifest("breast_cancer")
    assert m.source == "sklearn"
    assert m.openml_data_id is None
    assert m.leakage_column is None


def test_load_credit_g_manifest_openml_fields():
    m = load_manifest("credit_g")
    assert m.source == "openml"
    assert m.openml_data_id == 31
    assert m.leakage_column == "repayment_status"
    assert m.canonical_spec.positive_class == "bad"
    assert m.poisoned_variant.expected == "ABSTAIN"


def test_load_covertype_manifest_openml_fields():
    m = load_manifest("covertype")
    assert m.source == "openml"
    assert m.openml_data_id == 1596
    assert m.leakage_column == "Canopy_Density"
    assert m.canonical_spec.positive_class is None  # multiclass -> macro-F1


# Erfolgsschwelle k = 1 (Decker 14.09.): SUCCESS <=> macro-F1 >= best-Baseline-F1.
# Seit 23.09. auf dem RUN-HOLDOUT gemessen (baseline_difficulty.py --holdout,
# gleicher Split wie der Agent); die CV-Werte (09.08.) bleiben fuer Schwere/Terzile.
# Seit S1 (23.09.): Split 70/10/20, Referenzmodell auf der Validierung gewaehlt;
# baseline_f1 = sein Test-F1 (Endurteil), baseline_f1_val = sein Val-F1 (A/B-Gates).
BASELINE_F1 = {
    "wine": 0.971, "breast_cancer": 0.962, "cnae_9": 0.959, "covertype": 0.914,
    "phoneme": 0.862, "fashion_mnist": 0.877, "adult": 0.781, "vehicle": 0.756,
    "bank_marketing": 0.721, "credit_g": 0.632, "connect_4": 0.622,
    "eucalyptus": 0.627, "cmc": 0.492,
}
BASELINE_F1_VAL = {
    "wine": 1.000, "breast_cancer": 0.981, "cnae_9": 0.962, "covertype": 0.913,
    "phoneme": 0.873, "fashion_mnist": 0.880, "adult": 0.785, "vehicle": 0.774,
    "bank_marketing": 0.722, "credit_g": 0.644, "connect_4": 0.626,
    "eucalyptus": 0.646, "cmc": 0.461,
}


@pytest.mark.parametrize("name,expected", sorted(BASELINE_F1_VAL.items()))
def test_suite_manifests_carry_val_baseline(name, expected):
    assert load_manifest(name).baseline_f1_val == pytest.approx(expected)
BASELINE_F1_CV = {
    "wine": 0.983, "breast_cancer": 0.971, "cnae_9": 0.942, "covertype": 0.918,
    "phoneme": 0.890, "fashion_mnist": 0.882, "adult": 0.788, "vehicle": 0.787,
    "bank_marketing": 0.723, "credit_g": 0.680, "connect_4": 0.628,
    "eucalyptus": 0.613, "cmc": 0.485,
}


@pytest.mark.parametrize("name,expected", sorted(BASELINE_F1.items()))
def test_suite_manifests_carry_baseline_f1(name, expected):
    m = load_manifest(name)
    assert m.baseline_f1 == pytest.approx(expected)


@pytest.mark.parametrize("name,expected", sorted(BASELINE_F1_CV.items()))
def test_suite_manifests_keep_cv_baseline_for_difficulty(name, expected):
    assert load_manifest(name).baseline_f1_cv == pytest.approx(expected)


def test_baseline_f1_optional_but_validated(tmp_path):
    base = ("dataset: x\ntask_prompt: p\n"
            "canonical_spec: {task_type: classification, target_column: t, metric: f1}\n")
    (tmp_path / "x.yaml").write_text(base, encoding="utf-8")
    assert load_manifest("x", manifests_dir=str(tmp_path)).baseline_f1 is None
    (tmp_path / "x.yaml").write_text(base + "baseline_f1: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="baseline_f1"):
        load_manifest("x", manifests_dir=str(tmp_path))


def test_suite_has_expected_size():
    # Kandidaten-Suite (roadmap §3, beschlossen 08.08.): 10 Kern + 3 harte
    # CC18-Sets; die 2 Post-Cutoff-Sets kommen nach der Cutoff-Festlegung dazu.
    assert len(ALL_MANIFESTS) >= 13


@pytest.mark.parametrize("name", ALL_MANIFESTS)
def test_manifest_invariants(name):
    m = load_manifest(name)
    assert m.dataset == name
    assert m.canonical_spec.task_type == "classification"
    assert m.canonical_spec.metric == "f1"
    assert m.task_prompt
    # jede Suite braucht die vergiftete Variante (BU-Robustheit) mit ABSTAIN
    assert m.poisoned_variant is not None
    assert m.poisoned_variant.expected == "ABSTAIN"
    if m.source == "openml":
        assert m.openml_data_id is not None
        # --leakage braucht den getarnten Spaltennamen (kein Builtin-Fallback)
        assert m.leakage_column


@pytest.mark.parametrize("name", ALL_MANIFESTS)
def test_core_manifests_carry_a_spurious_column(name):
    """Leakage-Batch 2 (docs/leakage-runs.md §7): jedes Kern-Set braucht einen
    fachfremden Spaltennamen, verschieden vom Tarnnamen der perfekten Variante."""
    m = load_manifest(name)
    assert m.spurious_column
    assert m.spurious_column != m.leakage_column


def test_openml_manifest_without_data_id_raises(tmp_path):
    (tmp_path / "broken.yaml").write_text(
        "dataset: broken\nsource: openml\ntask_prompt: x\n"
        "canonical_spec:\n  task_type: classification\n"
        "  target_column: class\n  metric: f1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="openml_data_id"):
        load_manifest("broken", manifests_dir=str(tmp_path))


# --- Post-Cutoff-Manifeste (eigener Ordner, NICHT Teil der Kern-Suite) -------

POST_CUTOFF_DIR = os.path.join("manifests", "post_cutoff")
POST_CUTOFF = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(POST_CUTOFF_DIR, "*.yaml")))


def test_kaggle_manifest_without_competition_raises(tmp_path):
    (tmp_path / "broken.yaml").write_text(
        "dataset: broken\nsource: kaggle\ntask_prompt: x\n"
        "canonical_spec:\n  task_type: classification\n"
        "  target_column: y\n  metric: f1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="kaggle_competition"):
        load_manifest("broken", manifests_dir=str(tmp_path))


def test_kaggle_manifest_fields(tmp_path):
    (tmp_path / "k.yaml").write_text(
        "dataset: k\nsource: kaggle\nkaggle_competition: playground-series-s6e9\n"
        "drop_columns: [id]\ndata_sha256: " + "a" * 64 + "\ntask_prompt: x\n"
        "canonical_spec:\n  task_type: classification\n"
        "  target_column: Will_Buy_EV\n  metric: f1\n", encoding="utf-8")
    m = load_manifest("k", manifests_dir=str(tmp_path))
    assert m.kaggle_competition == "playground-series-s6e9"
    assert m.drop_columns == ("id",)
    assert m.data_sha256 == "a" * 64


def test_post_cutoff_sets_stay_out_of_core_suite():
    """main_runs.suite_datasets globbt manifests/*.yaml — Post-Cutoff-Sets
    duerfen dort nicht liegen, sonst wird N still 15 (Decker 14.09.: zusaetzlich zu N)."""
    assert len(POST_CUTOFF) == 2
    assert not set(POST_CUTOFF) & set(ALL_MANIFESTS)


@pytest.mark.parametrize("name", POST_CUTOFF)
def test_post_cutoff_manifest_invariants(name):
    m = load_manifest(name, manifests_dir=POST_CUTOFF_DIR)
    assert m.dataset == name
    assert m.source == "kaggle" and m.kaggle_competition
    assert m.data_sha256 and len(m.data_sha256) == 64
    assert m.canonical_spec.task_type == "classification"
    assert m.canonical_spec.metric == "f1"
    assert m.poisoned_variant is not None and m.poisoned_variant.expected == "ABSTAIN"
    assert m.leakage_column
    assert m.baseline_f1 is not None
