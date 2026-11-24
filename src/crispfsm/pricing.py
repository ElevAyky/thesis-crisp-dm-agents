# src/crispfsm/pricing.py
"""Preise pro 1 Mio. Tokens (USD). Stand 2026-09-09, verifiziert gegen die
offizielle Anthropic-Modelluebersicht (Standard-API-Raten).
Modellwahl + Begruendung: docs/modellwahl.md.

Korrektur 2026-09-09: Sonnet 5 stand hier auf 3/15 — mit der Begruendung, die
2/10 seien Intro-Preise bis 31.08.2026 und die Haupt-Runs liefen spaeter. Die
Annahme war falsch: 2/10 ist der Standardpreis (geprueft am 09.09., also nach
dem angenommenen Intro-Ende). Sonnet-5-Kosten aus Planung und Runs vor diesem
Datum sind damit um 50 % ueberschaetzt. Opus 4.8 war korrekt.

Achtung Historie: Bis 2026-08-08 stand hier fuer Opus 4.8 ein veralteter
Platzhalter (15/75 statt real 5/25) — cost_usd-Werte aus Runs vor diesem Datum
sind ~3x ueberschaetzt. Der Worker ist zur Laufzeit ueber das CLI-Flag --model
waehlbar; die Tabelle enthaelt alle im Projekt verwendeten Modelle, damit
cost_usd nach einem Modellwechsel weiter funktioniert."""
from __future__ import annotations
from crispfsm.backend import TokenUsage

PRICING: dict[str, dict[str, float]] = {
    # USD/Token; cache_read = 0.1x input, cache_creation = 1.25x input (5-Min-TTL).
    "claude-opus-4-8": {  # Hauptmodell (Pilot-Modell)
        "input": 5.0 / 1_000_000,
        "output": 25.0 / 1_000_000,
        "cache_read": 0.5 / 1_000_000,
        "cache_creation": 6.25 / 1_000_000,
    },
    "claude-sonnet-5": {  # Zweitmodell (Robustheits-Check) — 40 % des Opus-Preises
        "input": 2.0 / 1_000_000,
        "output": 10.0 / 1_000_000,
        "cache_read": 0.2 / 1_000_000,
        "cache_creation": 2.5 / 1_000_000,
    },
    "claude-opus-5": {  # nicht gewaehlt (Cutoff Mai 2026, s. modellwahl.md) —
        "input": 5.0 / 1_000_000,   # gelistet, damit ein Modellwechsel nicht
        "output": 25.0 / 1_000_000,  # still auf cost_usd = nan faellt
        "cache_read": 0.5 / 1_000_000,
        "cache_creation": 6.25 / 1_000_000,
    },
    "claude-sonnet-4-6": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
        "cache_read": 0.3 / 1_000_000,
        "cache_creation": 3.75 / 1_000_000,
    },
    "claude-haiku-4-5-20251001": {  # Dev-/Testlaeufe
        "input": 1.0 / 1_000_000,
        "output": 5.0 / 1_000_000,
        "cache_read": 0.1 / 1_000_000,
        "cache_creation": 1.25 / 1_000_000,
    },
}


def cost_usd(usage: TokenUsage, model: str) -> float:
    p = PRICING[model]
    return (usage.input_tokens * p["input"]
            + usage.output_tokens * p["output"]
            + usage.cache_read_tokens * p["cache_read"]
            + usage.cache_creation_tokens * p["cache_creation"])
