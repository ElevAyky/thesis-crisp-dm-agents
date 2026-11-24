# scripts/fetch_kaggle.py
"""Laedt train.csv einer Kaggle-Competition in den lokalen Daten-Cache
(Post-Cutoff-Kontrolle, Decker 14.09.; benchmarks.md §2).

Laeuft bewusst NICHT im Projekt-venv: das kaggle-Paket (+ Token in ~/.kaggle/
kaggle.json) braucht nur dieser einmalige Schritt, der Harness liest danach
nur noch den Cache (crispfsm.datasets.kaggle_cache_path).

    uvx --from kaggle python scripts/fetch_kaggle.py playground-series-s6e7 playground-series-s6e9

Voraussetzung: auf kaggle.com einmal pro Competition die Regeln akzeptiert
("Join Competition"), sonst 403. Ausgabe: sha256 fuers Manifest (data_sha256),
Zeilenzahl, Spalten, Klassenverteilung der letzten Spalte.
"""
from __future__ import annotations
import csv
import hashlib
import os
import sys
import zipfile
from collections import Counter


def cache_path(competition: str) -> str:
    # Spiegel von crispfsm.datasets.kaggle_cache_path (hier dupliziert, damit das
    # Script ohne Projekt-Import im uvx-Env laeuft).
    root = os.environ.get("CRISPFSM_DATA_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "crispfsm")
    return os.path.join(root, "kaggle", competition, "train.csv")


def fetch(competition: str) -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi

    target = cache_path(competition)
    folder = os.path.dirname(target)
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(target):
        api = KaggleApi()
        api.authenticate()
        api.competition_download_file(competition, "train.csv", path=folder, quiet=True)
        zipped = target + ".zip"
        if os.path.exists(zipped):
            with zipfile.ZipFile(zipped) as z:
                z.extract("train.csv", folder)
            os.remove(zipped)
    return target


def describe(path: str) -> None:
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        last = Counter(row[-1] for row in reader)
    print(f"  Datei       {path}")
    print(f"  data_sha256 {digest}")
    print(f"  Zeilen      {sum(last.values())}")
    print(f"  Spalten     {', '.join(header)}")
    print(f"  {header[-1]}: {dict(last.most_common())}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for comp in sys.argv[1:]:
        print(f"== {comp}")
        describe(fetch(comp))


if __name__ == "__main__":
    main()
