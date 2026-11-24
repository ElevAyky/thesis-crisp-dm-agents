# tests/test_datasets.py
import json
import os
import pandas as pd
import pytest
from crispfsm.datasets import prepare_dataset

def test_prepare_dataset_layout(tmp_path):
    workdir = str(tmp_path / "run")
    info = prepare_dataset("breast_cancer", workdir, target="target", seed=42)

    train_p = os.path.join(workdir, "work", "data", "train.parquet")
    test_p = os.path.join(workdir, "holdout", "test.parquet")

    assert os.path.exists(train_p)
    assert os.path.exists(test_p)
    # Test liegt NICHT in work/ (Leakage-Kontrolle by construction)
    assert not os.path.exists(os.path.join(workdir, "work", "data", "test.parquet"))

    train = pd.read_parquet(train_p)
    test = pd.read_parquet(test_p)
    assert "target" in train.columns and "target" in test.columns
    assert len(train) > len(test)                 # 80/20
    assert info["n_classes"] == 2
    assert info["n_features"] == train.shape[1] - 1

def test_prepare_dataset_no_overlap(tmp_path):
    workdir = str(tmp_path / "run")
    prepare_dataset("breast_cancer", workdir, target="target", seed=42)
    train = pd.read_parquet(os.path.join(workdir, "work", "data", "train.parquet"))
    test = pd.read_parquet(os.path.join(workdir, "holdout", "test.parquet"))
    # disjunkte Indizes (kein Sample in beiden)
    merged = train.merge(test, how="inner")
    assert len(merged) < len(test)  # praktisch kein vollständiges Überlappen


def _train_path(workdir):
    return os.path.join(workdir, "work", "data", "train.parquet")


def _test_path(workdir):
    return os.path.join(workdir, "holdout", "test.parquet")


def test_unknown_dataset_raises_keyerror(tmp_path):
    workdir = str(tmp_path / "run")
    with pytest.raises(KeyError):
        prepare_dataset("nope", workdir)


def test_wine_end_to_end(tmp_path):
    workdir = str(tmp_path / "run")
    info = prepare_dataset("wine", workdir, target="target", seed=42)

    train_p = _train_path(workdir)
    test_p = _test_path(workdir)
    assert os.path.exists(train_p)
    assert os.path.exists(test_p)

    assert info["n_classes"] == 3
    assert info["n_features"] == 13

    train = pd.read_parquet(train_p)
    test = pd.read_parquet(test_p)
    assert "target" in train.columns and "target" in test.columns
    # n_features excludes the target column
    assert train.shape[1] - 1 == info["n_features"]
    assert test.shape[1] - 1 == info["n_features"]


def test_breast_cancer_return_dict_shape(tmp_path):
    workdir = str(tmp_path / "run")
    info = prepare_dataset("breast_cancer", workdir, test_size=0.2, seed=42)

    assert set(info.keys()) == {"n_features", "n_classes", "n_train", "n_val", "n_test",
                                "target_column", "leakage_column", "leakage_values"}
    assert info["n_val"] == 0                        # Default: kein Validierungsanteil

    # breast_cancer has 569 samples total
    assert info["n_train"] + info["n_test"] == 569
    # 20% of 569 = 113.8 -> sklearn rounds up to 114
    assert info["n_test"] == 114
    assert info["n_train"] == 455
    assert info["n_train"] > info["n_test"]


def test_test_size_half_near_equal(tmp_path):
    workdir = str(tmp_path / "run")
    info = prepare_dataset("breast_cancer", workdir, test_size=0.5, seed=42)

    assert info["n_train"] + info["n_test"] == 569
    # 569 is odd, so the split can differ by at most 1
    assert abs(info["n_train"] - info["n_test"]) <= 1


def test_seed_reproducibility(tmp_path):
    wd_a = str(tmp_path / "a")
    wd_b = str(tmp_path / "b")
    wd_c = str(tmp_path / "c")

    prepare_dataset("breast_cancer", wd_a, seed=42)
    prepare_dataset("breast_cancer", wd_b, seed=42)
    prepare_dataset("breast_cancer", wd_c, seed=7)

    train_a = pd.read_parquet(_train_path(wd_a))
    train_b = pd.read_parquet(_train_path(wd_b))
    train_c = pd.read_parquet(_train_path(wd_c))

    # identical seed -> byte-for-byte identical contents (same rows, same order)
    pd.testing.assert_frame_equal(train_a, train_b)

    # different seed -> different split (not identical)
    assert not train_a.equals(train_c)


def test_stratify_preserves_class_proportions(tmp_path):
    workdir = str(tmp_path / "run")
    prepare_dataset("breast_cancer", workdir, test_size=0.2, seed=42)

    train = pd.read_parquet(_train_path(workdir))
    test = pd.read_parquet(_test_path(workdir))

    full = pd.concat([train, test], axis=0)
    full_props = full["target"].value_counts(normalize=True).sort_index()
    train_props = train["target"].value_counts(normalize=True).sort_index()

    # same set of classes present in train
    assert set(train_props.index) == set(full_props.index)
    # stratified split keeps class proportions within a few percent
    for cls in full_props.index:
        assert abs(train_props[cls] - full_props[cls]) < 0.03


def test_prepare_dataset_neutralizes_target_name(tmp_path):
    info = prepare_dataset("breast_cancer", str(tmp_path), neutralize_target=True)
    assert info["target_column"] == "col_30"          # 30 Features -> col_30
    df = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert "col_30" in df.columns
    assert "target" not in df.columns


def test_neutralized_target_keeps_semantic_values(tmp_path):
    info = prepare_dataset("breast_cancer", str(tmp_path), neutralize_target=True)
    df = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert set(df[info["target_column"]].unique()) == {"malignant", "benign"}


def test_prepare_dataset_default_stays_compatible(tmp_path):
    info = prepare_dataset("breast_cancer", str(tmp_path))
    assert info["target_column"] == "target"
    df = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert "target" in df.columns


def test_prepare_writes_test_features_without_target(tmp_path):
    """Spec §1.1: holdout/test_features.parquet = Test-Split OHNE Zielspalte,
    gleiche Zeilenzahl/-reihenfolge wie holdout/test.parquet."""
    info = prepare_dataset("wine", str(tmp_path), neutralize_target=True)
    test = pd.read_parquet(os.path.join(tmp_path, "holdout", "test.parquet"))
    feats = pd.read_parquet(os.path.join(tmp_path, "holdout", "test_features.parquet"))

    assert info["target_column"] not in feats.columns
    assert len(feats) == len(test)
    # Reihenfolge identisch: Feature-Spalten sind exakt test ohne Target
    pd.testing.assert_frame_equal(feats, test.drop(columns=[info["target_column"]]))


def test_prepare_dataset_inject_leakage(tmp_path):
    info = prepare_dataset("breast_cancer", str(tmp_path), neutralize_target=True,
                           inject_leakage=True)
    leak = info["leakage_column"]
    # Getarnter Name im Stil der echten Features — NICHT col_* (das würde neben
    # dem col_*-Target semantisch auffallen statt statistisch gefunden zu werden)
    assert leak == "mean nuclear density"
    assert not leak.startswith("col_")
    target = info["target_column"]

    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert leak in train.columns
    # Float-Spalte mit Nicht-0/1-Werten — kein Int-dtype-Tell zwischen Floats
    assert train[leak].dtype == float
    assert set(train[leak].unique()) != {0, 1}
    # Train: perfekt korreliert — pro Leak-Wert genau eine Zielklasse
    assert (train.groupby(leak)[target].nunique() == 1).all()

    test = pd.read_parquet(tmp_path / "holdout" / "test.parquet")
    feats = pd.read_parquet(tmp_path / "holdout" / "test_features.parquet")
    assert leak in test.columns and leak in feats.columns
    # Test: zufällig — mindestens ein Leak-Wert mischt beide Klassen
    assert (test.groupby(leak)[target].nunique() > 1).any()
    # gleiche Werte-Menge wie im Train (gleiche Randverteilung, kein neues Signal)
    assert set(test[leak].unique()) <= set(train[leak].unique())


@pytest.mark.parametrize("dataset", ["breast_cancer", "wine"])
def test_perfect_leak_keeps_train_value_class_mapping_in_test(tmp_path, dataset):
    """Review 24.10. (M3): factorize pro Split vertauschte die Zuordnung Wert -> Klasse
    zwischen Train und Test (adult: Train 76 % 7.84, Test 76 % 2.31) — ein Verteilungs-
    sprung als zusaetzlicher Hinweis auf den Leak. Test/Val nutzen jetzt die Train-
    Zuordnung: Wert-Anteile im Test = Klassen-Anteile im Test (Permutation aendert die
    Rand-, nicht die Zeilenzuordnung)."""
    info = prepare_dataset(dataset, str(tmp_path), neutralize_target=True,
                           inject_leakage=True, val_size=0.1)
    leak, target = info["leakage_column"], info["target_column"]
    value_of = {c: round(v, 6) for v, c in info["leakage_values"]}
    for part in ("test", "val"):
        df = pd.read_parquet(tmp_path / "holdout" / f"{part}.parquet")
        expected = df[target].astype(str).map(value_of).value_counts(normalize=True).sort_index()
        observed = df[leak].round(6).value_counts(normalize=True).sort_index()
        pd.testing.assert_series_equal(observed, expected, check_names=False)


def test_prepare_dataset_inject_leakage_wine(tmp_path):
    """Mehrklassen-Fall (3 Klassen) + dataset-spezifischer Tarnname."""
    info = prepare_dataset("wine", str(tmp_path), neutralize_target=True,
                           inject_leakage=True)
    leak = info["leakage_column"]
    assert leak == "proline_ratio"
    target = info["target_column"]
    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert train[leak].nunique() == 3
    assert (train.groupby(leak)[target].nunique() == 1).all()


def test_prepare_dataset_leakage_values_are_the_train_mapping(tmp_path):
    """leakage_values = [[Leak-Wert, Klasse], ...] so, wie die Spalte im TRAIN das Ziel
    kodiert — die Zuordnung, die ein Modell lernt, das dem Leak folgt. Grundlage der
    Leak-Messung auf dem Test (leakage.leak_reliance), für alle Arme gleich."""
    info = prepare_dataset("wine", str(tmp_path), neutralize_target=True,
                           inject_leakage=True)
    leak, target = info["leakage_column"], info["target_column"]
    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    observed = {round(float(v), 6): str(c)
                for v, c in train.groupby(leak)[target].first().items()}
    assert {round(v, 6): c for v, c in info["leakage_values"]} == observed
    assert len(info["leakage_values"]) == 3
    json.dumps(info["leakage_values"])               # landet im summary.json


def _nearest_center_accuracy(df, leak, target, leak_values):
    import numpy as np
    centers = np.array([v for v, _ in leak_values])
    labels = np.array([c for _, c in leak_values])
    decoded = labels[np.abs(df[leak].to_numpy()[:, None] - centers[None, :]).argmin(axis=1)]
    return float((decoded == df[target].astype(str).to_numpy()).mean())


def test_spurious_leak_is_strong_but_not_perfect_in_train(tmp_path):
    """Batch 2 (docs/leakage-runs.md §7): kontinuierlich, die Spalte allein trifft im Train
    ~85 % (kein CV = 1.0-Alarm), auf Validierung + Test nur Zufallsniveau."""
    info = prepare_dataset("breast_cancer", str(tmp_path), neutralize_target=True,
                           inject_leakage=True, leak_variant="spurious",
                           leakage_column="mean ice cream sales", val_size=0.1)
    leak, target = info["leakage_column"], info["target_column"]
    assert leak == "mean ice cream sales"
    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert train[leak].nunique() > 50                      # kontinuierlich, keine K Stufen
    acc = _nearest_center_accuracy(train, leak, target, info["leakage_values"])
    assert 0.78 < acc < 0.92
    for part in ("test", "val"):
        df = pd.read_parquet(tmp_path / "holdout" / f"{part}.parquet")
        assert _nearest_center_accuracy(df, leak, target, info["leakage_values"]) < 0.7


def test_spurious_centers_are_the_leakage_values(tmp_path):
    info = prepare_dataset("wine", str(tmp_path), neutralize_target=True,
                           inject_leakage=True, leak_variant="spurious",
                           leakage_column="rocket_launches")
    assert [round(v, 6) for v, _ in info["leakage_values"]] == [15.0, 20.53, 26.06]


def test_spurious_noise_targets_85_percent_for_any_class_count():
    from crispfsm.datasets import spurious_sd
    from scipy.stats import norm
    for k in (2, 3, 7, 10):
        a = 5.53 / (2 * spurious_sd(k))
        acc = (2 * (k - 1) * norm.cdf(a) - (k - 2)) / k
        assert acc == pytest.approx(0.85)


def test_unknown_leak_variant_raises(tmp_path):
    with pytest.raises(ValueError, match="leak_variant"):
        prepare_dataset("wine", str(tmp_path), inject_leakage=True, leak_variant="spuriuos")


def test_spurious_without_column_name_raises(tmp_path):
    with pytest.raises(ValueError, match="spurious"):
        prepare_dataset("wine", str(tmp_path), neutralize_target=True,
                        inject_leakage=True, leak_variant="spurious")


def test_prepare_from_manifest_spurious_uses_spurious_column(tmp_path):
    from crispfsm.datasets import prepare_from_manifest
    from crispfsm.manifests import load_manifest
    info = prepare_from_manifest(load_manifest("wine"), str(tmp_path),
                                 inject_leakage=True, leak_variant="spurious")
    assert info["leakage_column"] == "rocket_launches"


def test_prepare_dataset_no_leakage_by_default(tmp_path):
    info = prepare_dataset("breast_cancer", str(tmp_path), neutralize_target=True)
    assert info["leakage_column"] is None
    assert info["leakage_values"] is None
    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert "mean nuclear density" not in train.columns


# --- OpenML-Pfad (source="openml") — fetch_openml gemockt, kein Netzwerk ---

import types
import numpy as np
from crispfsm import datasets as ds


def _fake_openml_bunch():
    """Credit-g-artiges Mini-Set: mixed dtypes (Categorical + numerisch),
    ein Missing Value, semantisches Categorical-Target namens 'class'."""
    X = pd.DataFrame({
        "checking_status": pd.Categorical(["<0", "no checking"] * 30),
        "duration": np.linspace(4.0, 72.0, 60),
        "credit_amount": [1000.0 + i for i in range(60)],
    })
    X.loc[3, "duration"] = np.nan
    y = pd.Series(pd.Categorical(["good"] * 42 + ["bad"] * 18), name="class")
    return types.SimpleNamespace(data=X, target=y)


def test_prepare_dataset_openml(monkeypatch, tmp_path):
    called = {}
    def fake_fetch(data_id=None, as_frame=None):
        called["data_id"] = data_id
        return _fake_openml_bunch()
    monkeypatch.setattr(ds, "fetch_openml", fake_fetch)

    info = ds.prepare_dataset("credit_g", str(tmp_path), neutralize_target=True,
                              source="openml", openml_data_id=31)
    assert called["data_id"] == 31
    assert info["target_column"] == "col_3"
    assert info["n_classes"] == 2
    assert info["n_features"] == 3

    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    test = pd.read_parquet(tmp_path / "holdout" / "test.parquet")
    # Klassen-Labels bleiben semantisch, nur der Spaltenname ist neutralisiert
    assert set(train["col_3"].unique()) == {"good", "bad"}
    assert "class" not in train.columns
    # Missing Values bleiben erhalten (DU-Futter, keine stille Imputation)
    assert pd.concat([train, test])["duration"].isna().sum() == 1
    # Mixed dtypes überleben den Parquet-Roundtrip
    assert str(train["checking_status"].dtype) in ("category", "object")


def test_prepare_dataset_openml_stratified_split(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "fetch_openml", lambda **kw: _fake_openml_bunch())
    info = ds.prepare_dataset("credit_g", str(tmp_path), neutralize_target=True,
                              source="openml", openml_data_id=31, test_size=0.2)
    assert info["n_train"] == 48 and info["n_test"] == 12
    test = pd.read_parquet(tmp_path / "holdout" / "test.parquet")
    # Stratifizierung: 30 % 'bad' im Gesamtset -> auch im Test-Split (±1 Rundung)
    assert (test["col_3"] == "bad").sum() in (3, 4)  # 18 * 0.2 = 3.6


def test_prepare_dataset_openml_leakage_from_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "fetch_openml", lambda **kw: _fake_openml_bunch())
    info = ds.prepare_dataset("credit_g", str(tmp_path), neutralize_target=True,
                              source="openml", openml_data_id=31,
                              inject_leakage=True,
                              leakage_column="repayment_status")
    assert info["leakage_column"] == "repayment_status"
    target = info["target_column"]
    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert "repayment_status" in train.columns
    # Train perfekt korreliert, Test zufällig (gleiches Kontrakt wie Builtins)
    assert (train.groupby("repayment_status")[target].nunique() == 1).all()
    test = pd.read_parquet(tmp_path / "holdout" / "test.parquet")
    assert (test.groupby("repayment_status")[target].nunique() > 1).any()


def test_openml_without_data_id_raises(tmp_path):
    with pytest.raises(ValueError, match="openml_data_id"):
        ds.prepare_dataset("credit_g", str(tmp_path), source="openml")


def test_openml_leakage_without_column_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "fetch_openml", lambda **kw: _fake_openml_bunch())
    with pytest.raises(ValueError, match="leakage_column"):
        ds.prepare_dataset("credit_g", str(tmp_path), source="openml",
                           openml_data_id=31, inject_leakage=True)


def test_manifest_leakage_column_overrides_builtin_fallback(tmp_path):
    """Manifest-Feld gewinnt auch für sklearn-Builtins (datasets._LEAKAGE_COLUMNS
    ist nur Fallback)."""
    info = prepare_dataset("breast_cancer", str(tmp_path), neutralize_target=True,
                           inject_leakage=True, leakage_column="nuclear grade")
    assert info["leakage_column"] == "nuclear grade"
    train = pd.read_parquet(tmp_path / "work" / "data" / "train.parquet")
    assert "nuclear grade" in train.columns
    assert "mean nuclear density" not in train.columns


def test_load_frame_sklearn_semantic_labels():
    X, y = ds.load_frame("wine")
    assert X.shape == (178, 13)
    assert set(y.unique()) == {"class_0", "class_1", "class_2"}


def test_load_frame_openml(monkeypatch):
    monkeypatch.setattr(ds, "fetch_openml", lambda **kw: _fake_openml_bunch())
    X, y = ds.load_frame("credit_g", source="openml", openml_data_id=31)
    assert X.shape == (60, 3)
    # Target-Categorical ist auf plain strings normalisiert
    assert not isinstance(y.dtype, pd.CategoricalDtype)
    assert set(y.unique()) == {"good", "bad"}


# --- Kaggle-Pfad (Post-Cutoff-Kontrolle, Decker 14.09.) ---------------------

import hashlib
import crispfsm.datasets as ds_mod


def _write_kaggle_csv(root, comp="playground-series-s6e7", n=60):
    d = root / "kaggle" / comp
    d.mkdir(parents=True)
    rows = ["id,sleep_hours,diet,stress_level,health_condition"]
    labels = ["Healthy", "At Risk", "Unhealthy"]
    for i in range(n):
        rows.append(f"{i},{5 + i % 4}.5,{'veg' if i % 2 else 'mixed'},{i % 5},{labels[i % 3]}")
    p = d / "train.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_kaggle_cache_path_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CRISPFSM_DATA_DIR", str(tmp_path))
    assert ds_mod.kaggle_cache_path("playground-series-s6e7") == \
        str(tmp_path / "kaggle" / "playground-series-s6e7" / "train.csv")


def test_load_frame_kaggle_reads_cache_drops_id_and_splits_target(monkeypatch, tmp_path):
    monkeypatch.setenv("CRISPFSM_DATA_DIR", str(tmp_path))
    _write_kaggle_csv(tmp_path)
    X, y = ds_mod.load_frame("student_health", source="kaggle",
                             kaggle_competition="playground-series-s6e7",
                             source_target="health_condition", drop_columns=["id"])
    assert list(X.columns) == ["sleep_hours", "diet", "stress_level"]
    assert set(y.unique()) == {"Healthy", "At Risk", "Unhealthy"}
    assert all(isinstance(v, str) for v in y)        # plain strings wie OpenML-Pfad


def test_load_frame_kaggle_missing_cache_points_to_fetch_script(monkeypatch, tmp_path):
    monkeypatch.setenv("CRISPFSM_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_kaggle.py"):
        ds_mod.load_frame("x", source="kaggle", kaggle_competition="playground-series-s6e9",
                          source_target="Will_Buy_EV")


def test_load_frame_kaggle_verifies_sha256(monkeypatch, tmp_path):
    """Kaggle kann Daten nachtraeglich tauschen -> Manifest pinnt die Pruefsumme."""
    monkeypatch.setenv("CRISPFSM_DATA_DIR", str(tmp_path))
    p = _write_kaggle_csv(tmp_path)
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    ds_mod.load_frame("x", source="kaggle", kaggle_competition="playground-series-s6e7",
                      source_target="health_condition", drop_columns=["id"],
                      data_sha256=good)
    with pytest.raises(ValueError, match="sha256"):
        ds_mod.load_frame("x", source="kaggle", kaggle_competition="playground-series-s6e7",
                          source_target="health_condition", data_sha256="0" * 64)


def test_prepare_dataset_kaggle_neutralizes_target(monkeypatch, tmp_path):
    monkeypatch.setenv("CRISPFSM_DATA_DIR", str(tmp_path / "cache"))
    _write_kaggle_csv(tmp_path / "cache")
    info = prepare_dataset("student_health", str(tmp_path / "run"), neutralize_target=True,
                           source="kaggle", kaggle_competition="playground-series-s6e7",
                           source_target="health_condition", drop_columns=["id"])
    assert info["target_column"] == "col_3" and info["n_classes"] == 3
    assert info["n_train"] == 48 and info["n_test"] == 12
    train = pd.read_parquet(tmp_path / "run" / "work" / "data" / "train.parquet")
    assert "health_condition" not in train.columns and "id" not in train.columns
    assert set(train["col_3"].unique()) == {"Healthy", "At Risk", "Unhealthy"}


def test_prepare_dataset_kaggle_without_competition_raises(tmp_path):
    with pytest.raises(ValueError, match="kaggle_competition"):
        prepare_dataset("x", str(tmp_path), source="kaggle", source_target="y")


# --- Validierungsanteil des Harness (S1, 23.09.) ------------------------------

import json as _json


def _prep(tmp_path, name, val_size):
    d = tmp_path / f"{name}-{val_size}"
    info = prepare_dataset("breast_cancer", str(d), neutralize_target=True, val_size=val_size)
    return d, info


def test_val_split_keeps_test_rows_identical(tmp_path):
    """Test-Zeilen haengen nicht am Validierungsanteil (erst Test-, dann Val-Split)."""
    d0, _ = _prep(tmp_path, "a", 0.0)
    d1, _ = _prep(tmp_path, "b", 0.1)
    t0 = pd.read_parquet(d0 / "holdout" / "test.parquet")
    t1 = pd.read_parquet(d1 / "holdout" / "test.parquet")
    pd.testing.assert_frame_equal(t0, t1)


def test_val_split_sizes_and_disjointness(tmp_path):
    d, info = _prep(tmp_path, "c", 0.1)
    train = pd.read_parquet(d / "work" / "data" / "train.parquet")
    val = pd.read_parquet(d / "holdout" / "val.parquet")
    test = pd.read_parquet(d / "holdout" / "test.parquet")
    assert (info["n_train"], info["n_val"], info["n_test"]) == (len(train), len(val), len(test))
    assert len(train) + len(val) + len(test) == 569
    assert info["n_val"] == 57 and info["n_test"] == 114       # 10 % / 20 % von 569
    rows = pd.concat([train, val, test]).drop(columns=[info["target_column"]])
    assert not rows.duplicated().any()                          # drei disjunkte Teile
    assert not (d / "work" / "data" / "val.parquet").exists()   # Val-Labels nie im Workdir


def test_combined_features_and_index_reconstruct_both_parts(tmp_path):
    d, info = _prep(tmp_path, "d", 0.1)
    t = info["target_column"]
    feats = pd.read_parquet(d / "holdout" / "test_features.parquet")
    idx = _json.loads((d / "holdout" / "eval_index.json").read_text(encoding="utf-8"))
    assert t not in feats.columns
    assert len(feats) == info["n_val"] + info["n_test"]
    assert sorted(idx["val"] + idx["test"]) == list(range(len(feats)))
    for part in ("val", "test"):
        truth = pd.read_parquet(d / "holdout" / f"{part}.parquet").drop(columns=[t])
        pd.testing.assert_frame_equal(feats.iloc[idx[part]].reset_index(drop=True),
                                      truth.reset_index(drop=True))
    # gemischt, nicht "erst val, dann test" (Agent soll die Teile nicht erkennen)
    assert idx["val"] != list(range(info["n_val"]))


def test_leakage_column_random_in_val_like_test(tmp_path):
    d = tmp_path / "leak"
    info = prepare_dataset("breast_cancer", str(d), neutralize_target=True,
                           inject_leakage=True, val_size=0.1)
    t, leak = info["target_column"], info["leakage_column"]
    val = pd.read_parquet(d / "holdout" / "val.parquet")
    assert (val.groupby(leak)[t].nunique() > 1).any()          # nicht perfekt korreliert


def test_prepare_from_manifest_uses_val_fraction(tmp_path):
    from crispfsm.config import VAL_FRACTION
    from crispfsm.datasets import prepare_from_manifest
    from crispfsm.manifests import load_manifest
    assert VAL_FRACTION == 0.1
    info = prepare_from_manifest(load_manifest("breast_cancer"), str(tmp_path / "m"))
    assert info["n_val"] == 57
    assert (tmp_path / "m" / "holdout" / "eval_index.json").exists()
