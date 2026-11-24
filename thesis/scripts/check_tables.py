# thesis/scripts/check_tables.py
"""Gegenprobe: jede Zahl in thesis/tables/tab5_*.tex bzw. tab6_*.tex muss im Faktenblatt
thesis/facts/kap5.md bzw. kap6.md vorkommen.

Geprueft werden Dezimalzahlen, Uhrzeiten (hh:mm) und ganze Zahlen >= 100 (kleine ganze Zahlen
wie Zaehler 6/6 sind zu unspezifisch). Tausendertrennzeichen und Vorzeichen werden ignoriert.
Exit 1, wenn eine Zahl fehlt.

Aufruf (Repo-Root): uv run python thesis/scripts/check_tables.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+:\d{2}|\d+(?:\.\d+)?")
TEXTTT = re.compile(r"\\texttt\{[^}]*\}|\\label\{[^}]*\}|\\autoref\{[^}]*\}|\\ref\{[^}]*\}|\\cite\{[^}]*\}")


def tokens(text: str) -> set[str]:
    out = set()
    for t in NUM.findall(text):
        t = t.replace(",", "")
        if "." in t or ":" in t or int(t) >= 100:
            out.add(t)
    return out


def main() -> int:
    bad = 0
    for chap in ("5", "6"):
        with open(os.path.join(REPO, "thesis", "facts", f"kap{chap}.md"), encoding="utf-8") as f:
            facts = tokens(f.read())
        for path in sorted(glob.glob(os.path.join(REPO, "thesis", "tables", f"tab{chap}_*.tex"))):
            with open(path, encoding="utf-8") as f:
                # Kommentare und die Spaltendefinition (\begin{tabular}{...p{2.5cm}...}) nicht pruefen
                body = "\n".join(line for line in f if not line.lstrip().startswith("%")
                                 and not line.lstrip().startswith("\\begin{tabular"))
            body = TEXTTT.sub(" ", body)
            missing = sorted(tokens(body) - facts)
            name = os.path.basename(path)
            if missing:
                bad += 1
                print(f"FEHLT in kap{chap}.md  {name}: {', '.join(missing)}")
            else:
                print(f"ok  {name}: {len(tokens(body))} Zahlen im Faktenblatt gefunden")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
