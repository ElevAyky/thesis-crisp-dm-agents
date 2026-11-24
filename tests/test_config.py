# tests/test_config.py
import pytest

from crispfsm.config import RunConfig


def test_runconfig_defaults():
    cfg = RunConfig(run_id="r1", workdir="/tmp/r1")
    assert cfg.run_id == "r1"
    assert cfg.workdir == "/tmp/r1"
    assert cfg.dataset == "breast_cancer"
    assert cfg.target == "target"
    assert cfg.model == "claude-opus-4-8"
    assert cfg.max_retries == 3
    assert cfg.safety_max_calls == 20
    assert cfg.budget_usd_per_call == 4.0
    assert cfg.max_turns == 60
    # S4: gleiches Gesamtkontingent fuer alle Arme
    assert (cfg.run_budget_usd, cfg.run_max_turns) == (20.0, 300)
    assert cfg.eval_f1_threshold == 0.5
    assert cfg.recursion_limit == 50


def test_runconfig_all_fields_overridden():
    cfg = RunConfig(
        run_id="run-42",
        workdir="/runs/run-42",
        dataset="titanic",
        target="survived",
        model="claude-opus-4-8",
        max_retries=7,
        safety_max_calls=99,
        budget_usd_per_call=5.5,
        max_turns=12,
        eval_f1_threshold=0.8,
        recursion_limit=100,
    )
    assert cfg.run_id == "run-42"
    assert cfg.workdir == "/runs/run-42"
    assert cfg.dataset == "titanic"
    assert cfg.target == "survived"
    assert cfg.model == "claude-opus-4-8"
    assert cfg.max_retries == 7
    assert cfg.safety_max_calls == 99
    assert cfg.budget_usd_per_call == 5.5
    assert cfg.max_turns == 12
    assert cfg.eval_f1_threshold == 0.8
    assert cfg.recursion_limit == 100


def test_runconfig_budget_can_be_none():
    cfg = RunConfig(run_id="r1", workdir="/tmp/r1", budget_usd_per_call=None)
    assert cfg.budget_usd_per_call is None


def test_runconfig_requires_run_id_and_workdir():
    # run_id und workdir sind required (keine Defaults) — Konstruktion ohne sie
    # muss fehlschlagen.
    with pytest.raises(TypeError):
        RunConfig()
    with pytest.raises(TypeError):
        RunConfig(run_id="r1")
    with pytest.raises(TypeError):
        RunConfig(workdir="/tmp/r1")


def test_config_has_bu_defaults():
    cfg = RunConfig(run_id="r", workdir="/tmp/x")
    assert cfg.task_prompt == ""
    assert cfg.eval_metric == "f1"


def test_config_arm_default_and_react_turns():
    from crispfsm.config import FSM_PHASES
    cfg = RunConfig(run_id="r", workdir="w")
    assert cfg.arm == "hard_fsm"
    assert len(FSM_PHASES) == 5
    # S4: C/D bekommen das gemeinsame Run-Kontingent in ihrer einen Session
    assert cfg.react_max_turns == cfg.run_max_turns == 300
    # Default-Werte so gewaehlt, dass fuenf volle Phasen-Calls genau das Run-Budget sind
    assert cfg.run_max_turns == len(FSM_PHASES) * cfg.max_turns
    assert cfg.run_budget_usd == len(FSM_PHASES) * cfg.budget_usd_per_call

    cfg2 = RunConfig(run_id="r", workdir="w", run_max_turns=50)
    assert cfg2.react_max_turns == 50

    cfg3 = RunConfig(run_id="r", workdir="w", react_max_turns=7)
    assert cfg3.react_max_turns == 7                   # explizit gewinnt


def test_config_accepts_all_four_arms():
    """4-Arm-Gradient (experiment-design §1): alle Arm-Kennungen konstruierbar."""
    for arm in ("hard_fsm", "soft_fsm", "crisp_react", "naked_react"):
        assert RunConfig(run_id="r", workdir="w", arm=arm).arm == arm


def test_config_rejects_unknown_arm():
    """Tippfehler-Arm darf nicht still als hard_fsm laufen (falsch gelabelte Daten)."""
    with pytest.raises(ValueError, match="naked-react"):
        RunConfig(run_id="r", workdir="w", arm="naked-react")


def test_config_rejects_unknown_eval_metric():
    """Review-Auflage: eval_metric ist Harness-Config — Tippfehler sollen beim
    Konstruieren knallen, nicht erst als nutzloses Gate-Feedback an einen Worker."""
    with pytest.raises(ValueError, match="bogus"):
        RunConfig(run_id="r", workdir="w", eval_metric="bogus")
