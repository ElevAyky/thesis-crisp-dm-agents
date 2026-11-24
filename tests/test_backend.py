# tests/test_backend.py
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from crispfsm.backend import ClaudeCliBackend, WorkerResult, TokenUsage

FAKE_JSON = json.dumps({
    "result": "done",
    "usage": {"input_tokens": 100, "output_tokens": 20,
              "cache_read_input_tokens": 5, "cache_creation_input_tokens": 7},
    "duration_api_ms": 1234,
})


def _run_capture(backend, **overrides):
    """Helper: run backend.run with sane defaults under a mocked subprocess,
    returning (result_or_exc, cmd, kwargs). Override any run() kwarg or the
    mocked MagicMock fields (stdout/stderr/returncode) via overrides."""
    defaults = dict(
        prompt="do prep", cwd="/tmp/run", session_id="sess-1", resume=False,
        max_turns=10, budget_usd=2.0, allowed_tools=["Bash", "Write"],
    )
    mock_fields = dict(
        stdout=overrides.pop("stdout", FAKE_JSON),
        stderr=overrides.pop("stderr", ""),
        returncode=overrides.pop("returncode", 0),
    )
    defaults.update(overrides)
    prompt = defaults.pop("prompt")
    with patch("crispfsm.backend.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(**mock_fields)
        res = backend.run(prompt, **defaults)
    args, kwargs = mock_run.call_args
    return res, args[0], kwargs


def test_rate_limit_in_json_raises_rate_limit_error():
    """Abo-Kontingent erschöpft: claude -p liefert is_error + Limit-Text. Das ist
    kein Agent-Verhalten → RateLimitError, der Run wird nicht gewertet (run.py)
    und vom Runner wiederholt, sobald wieder Kontingent da ist."""
    from crispfsm.backend import RateLimitError
    payload = json.dumps({"type": "result", "subtype": "error", "is_error": True,
                          "result": "You've hit your usage limit. Your limit will reset at 3pm.",
                          "usage": {}})
    with pytest.raises(RateLimitError, match="usage limit"):
        _run_capture(ClaudeCliBackend(model="m"), stdout=payload)


def test_session_limit_message_raises_rate_limit_error():
    """Abo-Meldung 09/2026 ("hit your session limit") ist Kontingent-Ende, kein
    Crash — sonst bricht der Runner nach 5 "Fehlschlaegen" ab statt zu warten."""
    from crispfsm.backend import RateLimitError
    payload = json.dumps({"type": "result", "subtype": "success", "is_error": True,
                          "result": "You've hit your session limit · resets 6:30pm (Europe/Berlin)",
                          "usage": {}})
    with pytest.raises(RateLimitError, match="session limit"):
        _run_capture(ClaudeCliBackend(model="m"), stdout=payload)


def test_rate_limit_without_json_raises_rate_limit_error():
    from crispfsm.backend import RateLimitError
    with pytest.raises(RateLimitError):
        _run_capture(ClaudeCliBackend(model="m"), stdout="",
                     stderr="Error: rate limit exceeded (429)", returncode=1)


def test_other_is_error_raises_backend_error():
    """Sonstige CLI-Fehler (is_error ohne Limit-Text) sind Harness-Fehler → ERROR/crash."""
    from crispfsm.backend import BackendError, RateLimitError
    payload = json.dumps({"type": "result", "subtype": "error_during_execution",
                          "is_error": True, "result": "Internal server error", "usage": {}})
    with pytest.raises(BackendError) as exc:
        _run_capture(ClaudeCliBackend(model="m"), stdout=payload)
    assert not isinstance(exc.value, RateLimitError)


def test_max_turns_or_budget_subtype_is_normal_result():
    """error_max_turns / error_max_budget_usd = der Worker hat sein Budget verbraucht.
    Das ist Agent-Verhalten (Gate-Fail folgt), kein Harness-Fehler."""
    for subtype in ("error_max_turns", "error_max_budget_usd"):
        payload = json.dumps({"type": "result", "subtype": subtype, "is_error": True,
                              "result": "", "usage": {"input_tokens": 1, "output_tokens": 1}})
        res, _, _ = _run_capture(ClaudeCliBackend(model="m"), stdout=payload)
        assert isinstance(res, WorkerResult)
        assert res.raw["subtype"] == subtype


def _pair(cmd, flag):
    """Value that immediately follows `flag` in the cmd list, or None."""
    if flag not in cmd:
        return None
    return cmd[cmd.index(flag) + 1]

def test_run_builds_cmd_and_parses_usage(tmp_path):
    backend = ClaudeCliBackend(model="claude-sonnet-4-6")
    with patch("crispfsm.backend.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=FAKE_JSON, stderr="", returncode=0)
        res = backend.run("do prep", cwd=str(tmp_path), session_id="sess-1",
                          resume=False, max_turns=10, budget_usd=2.0,
                          allowed_tools=["Bash", "Write"])

    args, kwargs = mock_run.call_args
    cmd = args[0]
    # cmd[0] ist der via shutil.which aufgelöste Pfad (Windows: claude.cmd-Shim)
    assert os.path.basename(cmd[0]).startswith("claude")
    assert cmd[1] == "-p"
    assert "--session-id" in cmd and "sess-1" in cmd
    assert "--resume" not in cmd                      # erster Call: keine Resume
    assert "--bare" not in cmd                         # --bare bricht Abo-Auth (>=2.1.x) — bewusst weggelassen
    assert "--allowedTools" in cmd
    # Prompt geht via STDIN, NICHT als argv — der Windows-.cmd-Shim re-parst die
    # Kommandozeile und verschluckt bei Quotes im Prompt nachfolgende Flags.
    assert kwargs["input"] == "do prep"
    assert "do prep" not in cmd
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["cwd"] == str(tmp_path)
    assert isinstance(res, WorkerResult)
    assert res.usage.input_tokens == 100
    assert res.usage.cache_creation_tokens == 7
    assert res.latency_ms == 1234
    assert res.session_id == "sess-1"

def test_run_resume_uses_resume_flag(tmp_path):
    backend = ClaudeCliBackend(model="claude-sonnet-4-6")
    with patch("crispfsm.backend.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=FAKE_JSON, stderr="", returncode=0)
        backend.run("retry", cwd=str(tmp_path), session_id="sess-1",
                    resume=True, max_turns=10, budget_usd=None, allowed_tools=["Bash"])
    cmd = mock_run.call_args[0][0]
    assert "--resume" in cmd
    assert "--session-id" not in cmd                  # beim Resume keine neue ID setzen


def test_cmd_flags_carry_expected_values(tmp_path):
    backend = ClaudeCliBackend(model="claude-opus-4-6")
    _, cmd, _ = _run_capture(
        backend, cwd=str(tmp_path), max_turns=7,
        allowed_tools=["Bash", "Write", "Read"],
    )
    # Each flag must be immediately followed by its value (order matters).
    assert _pair(cmd, "--model") == "claude-opus-4-6"
    assert _pair(cmd, "--max-turns") == "7" == str(7)
    assert _pair(cmd, "--allowedTools") == "Bash,Write,Read"
    assert _pair(cmd, "--output-format") == "json"
    assert "--bare" not in cmd                         # --bare bricht Abo-Auth — bewusst weg


def test_cmd_isolates_worker_context(tmp_path):
    """Harness-Isolation (docs/harness-overhead.md §5, Decker 14.09.): der Worker
    bekommt exakt dasselbe Tool-Set und keinen Account-Kontext (Plugins, Skills,
    MCP, Chrome) — unabhängig davon, was auf dem Betreiber-Account aktiv ist.
    --allowedTools filtert nur die Auto-Freigabe; erst --tools begrenzt die
    gesendeten Schemas."""
    backend = ClaudeCliBackend(model="claude-opus-4-8")
    _, cmd, _ = _run_capture(
        backend, cwd=str(tmp_path), max_turns=3,
        allowed_tools=["Read", "Write", "Bash"],
    )
    assert _pair(cmd, "--tools") == _pair(cmd, "--allowedTools") == "Read,Write,Bash"
    assert _pair(cmd, "--setting-sources") == ""
    assert "--strict-mcp-config" in cmd
    assert _pair(cmd, "--mcp-config") == '{"mcpServers":{}}'
    assert "--no-chrome" in cmd


def test_cmd_disables_auto_memory(tmp_path):
    """Befund 24.10. (docs/befund-auto-memory.md): ohne Sperre teilen sich alle Runs im
    runs-Repo ein Claude-Code-Auto-Memory — spaetere Runs lesen Lehren frueherer. Die
    Isolations-Flags oben schalten das NICHT ab. Doppelt gesperrt (Setting + Env), beide
    im Kanarien-Test wirksam."""
    backend = ClaudeCliBackend(model="claude-opus-4-8")
    _, cmd, kwargs = _run_capture(backend, cwd=str(tmp_path))
    assert json.loads(_pair(cmd, "--settings")) == {"autoMemoryEnabled": False}
    assert kwargs["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    # restliche Umgebung bleibt erhalten (PATH, Abo-Login-Pfade)
    assert kwargs["env"]["PATH"] == os.environ["PATH"]


def test_cmd_disables_git_context(tmp_path):
    """Review 24.10. (K1): Claude Code haengt sonst Git-Status + 'Recent commits' des
    runs-Repos an — mit Ergebnis-Commits wie 'Leak 0/39 ausgenutzt'."""
    _, _, kwargs = _run_capture(ClaudeCliBackend(model="m"), cwd=str(tmp_path))
    assert kwargs["env"]["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"] == "1"


def test_worker_cannot_discover_runs_repo_via_git(tmp_path):
    """Befund 27.10. (docs/befund-auto-memory.md, K-git): Der Arbeitsordner liegt im
    runs-Repo. Ein Worker rief selbst `git log`/`git status` auf und sah Ergebnis-Commits
    und Nachbar-Runs. GIT_CEILING_DIRECTORIES sperrt den Aufstieg in jeden Elternordner."""
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git nicht installiert")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    work = tmp_path / "main-x-hard_fsm-r1" / "work"
    work.mkdir(parents=True)
    _, _, kwargs = _run_capture(ClaudeCliBackend(model="m"), cwd=str(work))
    probe = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=work, env=kwargs["env"],
                           capture_output=True, text=True)
    assert probe.returncode != 0 and "not a git repository" in probe.stderr
    # Gegenprobe: ohne Sperre findet git das Repo
    assert subprocess.run(["git", "rev-parse", "--git-dir"], cwd=work,
                          capture_output=True).returncode == 0


def test_worker_env_drops_inherited_claude_session_vars(tmp_path, monkeypatch):
    """Review 24.10. (G1): aus einer Desktop-/SDK-Session erben Worker sonst ~30
    Claude-Variablen (Session-IDs, Messaging-Socket, Nutzer-Mail, Base-URL). Die
    Worker-Umgebung darf nicht davon abhaengen, wer den Runner startet."""
    for k in ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_USER_EMAIL", "CLAUDE_CODE_MESSAGING_TOKEN",
              "CLAUDE_CODE_ENTRYPOINT", "CLAUDECODE", "ANTHROPIC_BASE_URL"):
        monkeypatch.setenv(k, "x")
    _, _, kwargs = _run_capture(ClaudeCliBackend(model="m"), cwd=str(tmp_path))
    env = kwargs["env"]
    assert not [k for k in env if k.startswith(("CLAUDE", "ANTHROPIC_"))
                and k not in ("CLAUDE_CODE_DISABLE_AUTO_MEMORY",
                              "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS")]
    assert env["PATH"] == os.environ["PATH"]


def test_cmd_pins_effort(tmp_path):
    """Review 24.10. (M1): Effort nicht dem CLI-/Server-Default ueberlassen."""
    _, cmd, _ = _run_capture(ClaudeCliBackend(model="m"), cwd=str(tmp_path))
    assert _pair(cmd, "--effort") == "high"


def test_max_budget_present_when_number():
    backend = ClaudeCliBackend(model="m")
    _, cmd, _ = _run_capture(backend, budget_usd=3.5)
    assert "--max-budget-usd" in cmd
    assert _pair(cmd, "--max-budget-usd") == "3.5" == str(3.5)


def test_max_budget_absent_when_none():
    backend = ClaudeCliBackend(model="m")
    _, cmd, _ = _run_capture(backend, budget_usd=None)
    assert "--max-budget-usd" not in cmd


def test_non_json_stdout_raises_runtimeerror_with_rc_and_stderr():
    backend = ClaudeCliBackend(model="m")
    with pytest.raises(RuntimeError) as exc:
        _run_capture(backend, stdout="this is not json",
                     returncode=2, stderr="kaboom")
    msg = str(exc.value)
    assert "2" in msg                                  # returncode surfaced
    assert "kaboom" in msg                             # (truncated) stderr surfaced


def test_stderr_is_truncated_to_500_chars():
    backend = ClaudeCliBackend(model="m")
    long_err = "x" * 1000
    with pytest.raises(RuntimeError) as exc:
        _run_capture(backend, stdout="nope", returncode=1, stderr=long_err)
    msg = str(exc.value)
    assert "x" * 500 in msg
    assert "x" * 501 not in msg                         # capped at 500


def test_missing_usage_key_yields_zero_token_counts():
    backend = ClaudeCliBackend(model="m")
    payload = json.dumps({"result": "ok", "duration_api_ms": 42})  # no "usage"
    res, _, _ = _run_capture(backend, stdout=payload)
    assert res.usage.input_tokens == 0
    assert res.usage.output_tokens == 0
    assert res.usage.cache_read_tokens == 0
    assert res.usage.cache_creation_tokens == 0


def test_missing_result_key_yields_empty_text():
    backend = ClaudeCliBackend(model="m")
    payload = json.dumps({"usage": {}, "duration_api_ms": 42})  # no "result"
    res, _, _ = _run_capture(backend, stdout=payload)
    assert res.text == ""


def test_missing_duration_yields_zero_latency():
    backend = ClaudeCliBackend(model="m")
    payload = json.dumps({"result": "ok", "usage": {}})  # no "duration_api_ms"
    res, _, _ = _run_capture(backend, stdout=payload)
    assert res.latency_ms == 0.0
    assert isinstance(res.latency_ms, float)


def test_worker_result_raw_and_session_id():
    backend = ClaudeCliBackend(model="m")
    res, _, _ = _run_capture(backend, session_id="sess-XYZ")
    assert res.raw == json.loads(FAKE_JSON)            # raw == parsed dict
    assert res.session_id == "sess-XYZ"                # passthrough


def test_token_usage_as_dict_round_trips():
    backend = ClaudeCliBackend(model="m")
    res, _, _ = _run_capture(backend)
    src = json.loads(FAKE_JSON)["usage"]
    d = res.usage.as_dict()
    assert d["input_tokens"] == src["input_tokens"]
    assert d["output_tokens"] == src["output_tokens"]
    assert d["cache_read_tokens"] == src["cache_read_input_tokens"]
    assert d["cache_creation_tokens"] == src["cache_creation_input_tokens"]


def test_subprocess_called_with_passed_cwd(tmp_path):
    backend = ClaudeCliBackend(model="m")
    _, _, kwargs = _run_capture(backend, cwd=str(tmp_path))
    assert kwargs["cwd"] == str(tmp_path)
