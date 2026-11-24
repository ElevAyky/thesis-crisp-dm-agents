# tests/test_pricing.py
import pytest

from crispfsm.backend import TokenUsage
from crispfsm.pricing import cost_usd, PRICING
from crispfsm.config import RunConfig
from crispfsm.state import new_state


def test_tokenusage_total():
    u = TokenUsage(input_tokens=100, output_tokens=50)
    assert u.total == 150


def test_tokenusage_total_excludes_cache_fields():
    # total ist input+output und darf die Cache-Felder NICHT mitzaehlen.
    u = TokenUsage(input_tokens=100, output_tokens=50,
                   cache_read_tokens=999, cache_creation_tokens=777)
    assert u.total == 150


def test_tokenusage_cache_fields_default_to_zero():
    u = TokenUsage(input_tokens=10, output_tokens=20)
    assert u.cache_read_tokens == 0
    assert u.cache_creation_tokens == 0


def test_tokenusage_as_dict():
    u = TokenUsage(input_tokens=10, output_tokens=20,
                   cache_read_tokens=30, cache_creation_tokens=40)
    d = u.as_dict()
    assert d == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_tokens": 30,
        "cache_creation_tokens": 40,
    }
    # genau diese 4 Keys, nicht mehr.
    assert set(d.keys()) == {
        "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_creation_tokens",
    }


def test_cost_usd_all_four_terms_manual():
    # Alle vier Terme mit nicht-null Cache-Werten; Erwartung von Hand gerechnet
    # fuer ein konkretes Modell (claude-sonnet-4-6).
    u = TokenUsage(input_tokens=1000, output_tokens=2000,
                   cache_read_tokens=3000, cache_creation_tokens=4000)
    # Sonnet-Raten (USD/Token): in 3.0/1e6, out 15.0/1e6,
    # cache_read 0.3/1e6, cache_creation 3.75/1e6.
    expected = (
        1000 * (3.0 / 1_000_000)
        + 2000 * (15.0 / 1_000_000)
        + 3000 * (0.3 / 1_000_000)
        + 4000 * (3.75 / 1_000_000)
    )
    assert abs(cost_usd(u, "claude-sonnet-4-6") - expected) < 1e-9


@pytest.mark.parametrize("model", list(PRICING.keys()))
def test_cost_usd_each_model_all_terms(model):
    u = TokenUsage(input_tokens=1234, output_tokens=5678,
                   cache_read_tokens=9012, cache_creation_tokens=3456)
    p = PRICING[model]
    expected = (
        u.input_tokens * p["input"]
        + u.output_tokens * p["output"]
        + u.cache_read_tokens * p["cache_read"]
        + u.cache_creation_tokens * p["cache_creation"]
    )
    assert abs(cost_usd(u, model) - expected) < 1e-9


def test_pricing_table_has_expected_models_and_rate_keys():
    assert set(PRICING.keys()) == {
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    }
    for model, rates in PRICING.items():
        assert set(rates.keys()) == {
            "input", "output", "cache_read", "cache_creation",
        }, model


def test_cost_usd_uses_pricing_table():
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000,
                   cache_read_tokens=0, cache_creation_tokens=0)
    model = next(iter(PRICING))            # irgendein bekanntes Modell
    p = PRICING[model]
    # je 1M Tokens: PRICING-Werte sind USD/Token, Usage hat je 1M Tokens.
    # (Plan-Vorlage hatte hier den Faktor 1_000_000 vergessen — korrigiert,
    #  damit die Assertion ihre eigene Intention "je 1M Tokens" prueft.)
    expected = (p["input"] + p["output"]) * 1_000_000
    assert abs(cost_usd(u, model) - expected) < 1e-9


def test_cost_usd_unknown_model_raises():
    u = TokenUsage(input_tokens=1, output_tokens=1)
    try:
        cost_usd(u, "does-not-exist")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_runconfig_defaults():
    cfg = RunConfig(run_id="r1", workdir="/tmp/r1")
    assert cfg.dataset == "breast_cancer"
    assert cfg.max_retries == 3
    assert cfg.model == "claude-opus-4-8"


def test_new_state_initial_values():
    cfg = RunConfig(run_id="r1", workdir="/tmp/r1")
    s = new_state(cfg)
    assert s["visited_phases"] == []
    assert s["attempts"] == {"bu": 0, "du": 0, "prep": 0, "model": 0, "eval": 0, "react": 0}
    assert s["success"] is False
    assert s["final_f1"] is None
    assert s["total_calls"] == 0


def test_sonnet_5_steht_auf_den_standardpreisen_2_10():
    """Regression: bis 09.09.2026 stand Sonnet 5 hier auf 3/15 — mit der Annahme,
    2/10 seien Intro-Preise bis 31.08.2026. Falsch: 2/10 ist der Standardpreis.
    Sonnet 5 ist das Zweitmodell des Robustheits-Checks, der Fehler ginge also
    direkt in die Kostenrechnung der Thesis."""
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000,
                   cache_read_tokens=0, cache_creation_tokens=0)
    assert cost_usd(u, "claude-sonnet-5") == pytest.approx(2.0 + 10.0)


def test_opus_familie_kostet_durchgaengig_5_25():
    """Innerhalb der Opus-Linie ist der Preis kein Auswahlkriterium — die
    Modellwahl 4.8 statt 5 haengt allein am Training-Cutoff (modellwahl.md)."""
    u = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000,
                   cache_read_tokens=0, cache_creation_tokens=0)
    for model in ("claude-opus-4-8", "claude-opus-5"):
        assert cost_usd(u, model) == pytest.approx(5.0 + 25.0)
