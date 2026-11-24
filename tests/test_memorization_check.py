# tests/test_memorization_check.py
"""Ein Namens-Treffer zaehlt nur, wenn der Name NICHT schon im Task-Prompt steht —
sonst misst der Check Prompt-Echo statt Vorwissen."""
from __future__ import annotations
import json
import re

import pytest

import memorization_check as mc
from crispfsm.manifests import load_manifest
from main_runs import suite_datasets


@pytest.mark.parametrize("dataset", suite_datasets("manifests"))
def test_every_suite_dataset_has_patterns_none_matching_its_own_prompt(dataset):
    prompt = load_manifest(dataset).task_prompt
    assert dataset in mc.SPECIFIC
    kept = mc.specific_patterns(dataset, prompt)
    assert kept, f"{dataset}: alle Muster treffen den eigenen Prompt"
    assert not any(re.search(p, prompt, re.I) for p in kept)


def test_prompt_echo_is_filtered():
    assert mc.specific_patterns("wine", "Classify the wine using load_wine data.") == \
        [p for p in mc.SPECIFIC["wine"] if p != r"load_wine"]


def test_scan_run_finds_named_generic_and_external(tmp_path):
    run = tmp_path / "main-vehicle-naked_react-r1"
    (run / "work").mkdir(parents=True)
    (run / "events.jsonl").write_text(json.dumps(
        {"worker_text": "Done. This is the Statlog Vehicle Silhouettes dataset."}) + "\n",
        encoding="utf-8")
    (run / "work" / "solve.py").write_text(
        "from sklearn.datasets import fetch_openml\nX = fetch_openml('vehicle')\n", encoding="utf-8")
    r = mc.scan_run(str(run), mc.specific_patterns("vehicle", "Determine the type of vehicle."))
    assert "statlog" in r["named"]
    assert r["generic"]                                   # "this is the ... dataset"
    assert "fetch_openml" in r["external"]


def test_clean_run_has_no_hits(tmp_path):
    run = tmp_path / "main-cmc-crisp_react-r1"
    (run / "work").mkdir(parents=True)
    (run / "events.jsonl").write_text(json.dumps({"worker_text": "Modell trainiert."}) + "\n",
                                      encoding="utf-8")
    (run / "work" / "m.py").write_text("import pandas as pd\n", encoding="utf-8")
    r = mc.scan_run(str(run), mc.specific_patterns("cmc", load_manifest("cmc").task_prompt))
    assert r == {"named": [], "generic": [], "external": []}
