# tests/test_state.py
from crispfsm.config import RunConfig
from crispfsm.state import new_state


def _cfg():
    return RunConfig(run_id="run-1", workdir="/runs/run-1",
                     dataset="iris", target="species")


def test_new_state_copies_cfg_fields():
    cfg = _cfg()
    s = new_state(cfg)
    assert s["run_id"] == "run-1"
    assert s["dataset"] == "iris"
    assert s["target"] == "species"
    assert s["workdir"] == "/runs/run-1"


def test_new_state_seeds_initial_values():
    s = new_state(_cfg())
    assert s["current_phase"] == ""
    assert s["visited_phases"] == []
    assert s["session_ids"] == {}
    assert s["attempts"] == {"bu": 0, "du": 0, "prep": 0, "model": 0, "eval": 0, "react": 0}
    assert s["gate_passed"] is False
    assert s["gate_metrics"] == {}
    assert s["gate_feedback"] == ""
    assert s["artifacts"] == {}
    assert s["usage_total"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    assert s["usage_per_phase"] == {}
    assert s["total_calls"] == 0
    assert s["success"] is False
    assert s["final_f1"] is None


def test_new_state_usage_total_token_keys():
    s = new_state(_cfg())
    assert set(s["usage_total"].keys()) == {
        "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_creation_tokens",
    }
    assert all(v == 0 for v in s["usage_total"].values())


def test_new_state_returns_independent_dicts():
    cfg = _cfg()
    s1 = new_state(cfg)
    s2 = new_state(cfg)
    # Mutationen an verschachtelten Containern duerfen nicht durchschlagen.
    s1["visited_phases"].append("prep")
    s1["attempts"]["prep"] += 1
    s1["session_ids"]["prep"] = "abc"
    s1["usage_total"]["input_tokens"] = 999
    s1["artifacts"]["x"] = 1

    assert s2["visited_phases"] == []
    assert s2["attempts"] == {"bu": 0, "du": 0, "prep": 0, "model": 0, "eval": 0, "react": 0}
    assert s2["session_ids"] == {}
    assert s2["usage_total"]["input_tokens"] == 0
    assert s2["artifacts"] == {}


def test_new_state_has_bu_fields():
    cfg = RunConfig(run_id="r", workdir="/tmp/x")
    s = new_state(cfg)
    assert s["attempts"]["bu"] == 0
    assert s["task_spec"] is None
    assert s["abstained"] is False
    assert s["bu_correct"] is None


def test_new_state_has_react_and_target_correct():
    s = new_state(RunConfig(run_id="r", workdir="w"))
    assert s["attempts"]["react"] == 0
    assert s["target_correct"] is None


def test_accumulate_usage_totals_and_per_phase():
    from crispfsm.config import RunConfig
    from crispfsm.state import accumulate_usage, new_state
    s = new_state(RunConfig(run_id="r", workdir="w"))
    u = {"input_tokens": 10, "output_tokens": 2,
         "cache_read_tokens": 1, "cache_creation_tokens": 0}
    total, per_phase = accumulate_usage(s, "react", u)
    assert total["input_tokens"] == 10
    assert per_phase["react"]["output_tokens"] == 2
    # zweiter Call auf Basis des Updates akkumuliert
    s2 = {**s, "usage_total": total, "usage_per_phase": per_phase}
    total2, per_phase2 = accumulate_usage(s2, "react", u)
    assert total2["input_tokens"] == 20
    assert per_phase2["react"]["input_tokens"] == 20


def test_new_state_has_du_phase_fields(tmp_path):
    from crispfsm.config import RunConfig
    from crispfsm.state import new_state
    s = new_state(RunConfig(run_id="r", workdir=str(tmp_path)))
    assert s["attempts"]["du"] == 0
    assert s["du_report"] is None
