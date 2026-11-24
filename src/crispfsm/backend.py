# src/crispfsm/backend.py
from __future__ import annotations
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field

# Auto-Memory der Worker gesperrt (Befund 24.10., docs/befund-auto-memory.md) — doppelt,
# weil --setting-sources "" Settings-Dateien kappt, ein Env-Wert aber immer greift.
# Git-Kontext gesperrt (Review 24.10., K1): sonst Git-Status + "Recent commits" des
# runs-Repos im Worker-Kontext, inkl. Ergebnis-Commits frueherer Runs.
WORKER_SETTINGS = {"autoMemoryEnabled": False}
WORKER_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
              "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1"}
WORKER_EFFORT = "high"


def _ancestors(path: str) -> list[str]:
    out, cur = [], os.path.dirname(os.path.abspath(path))
    while cur and cur not in out:
        out.append(cur)
        cur = os.path.dirname(cur)
    return out


def worker_env(cwd: str | None = None) -> dict:
    """Umgebung des Workers: die des Runners OHNE geerbte Claude-/Anthropic-Variablen
    (Review 24.10., G1: aus einer Desktop-/SDK-Session kamen Session-IDs, Messaging-Socket,
    Nutzer-Mail, Base-URL mit) plus WORKER_ENV. Abo-Login liegt in ~/.claude, nicht im Env.
    Mit cwd zusaetzlich GIT_CEILING_DIRECTORIES = alle Elternordner (Befund 27.10., K-git:
    Worker rief selbst `git log` im runs-Repo auf und sah Ergebnis-Commits + Nachbar-Runs)."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC_"))}
    env = {**base, **WORKER_ENV}
    if cwd:
        env["GIT_CEILING_DIRECTORIES"] = os.pathsep.join(_ancestors(cwd))
    return env


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }


@dataclass
class WorkerResult:
    text: str
    usage: TokenUsage
    latency_ms: float
    session_id: str
    raw: dict = field(repr=False, default_factory=dict)


class BackendError(RuntimeError):
    """Harness-/CLI-Fehler (kein JSON, is_error ohne Agent-Bezug) → run.py: ERROR/crash."""


class RateLimitError(BackendError):
    """Abo-Kontingent erschöpft / Rate-Limit. Kein Agent-Verhalten: der Run wird
    nicht gewertet (ERROR/rate_limited) und vom Runner wiederholt, sobald wieder
    Kontingent da ist (Entscheidung Johannes 18.09.)."""


# Textmuster, an denen die CLI ein Kontingent-/Rate-Limit meldet. Bewusst breit:
# ein falsch als rate_limited erkannter Crash kostet nur eine Wartezeit, ein
# übersehenes Limit würde als FAIL in die SUCCESS-Rate laufen.
_RATE_LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "limit will reset",
                       "hit your limit", "too many requests", "429", "overloaded",
                       "quota",
                       # Abo-Meldung seit 09/2026: "You've hit your session limit · resets 6:30pm"
                       "session limit", "weekly limit")

# Subtypes, bei denen is_error=True AGENT-Verhalten ist (Budget/Turns des Workers
# verbraucht → unvollständiges Artefakt → regulärer Gate-Fail), kein Harness-Fehler.
_AGENT_LIMIT_SUBTYPES = ("error_max_turns", "error_max_budget_usd")


def _looks_rate_limited(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _RATE_LIMIT_MARKERS)


class ClaudeCliBackend:
    """Agentischer Worker via `claude -p`. Token-Counts exakt aus dem JSON;
    Kosten werden separat über pricing.py gerechnet (architecture.md Decision 4)."""

    def __init__(self, model: str):
        self.name = "claude_cli"
        self.model = model

    def run(self, prompt: str, *, cwd: str, session_id: str, resume: bool,
            max_turns: int, budget_usd: float | None,
            allowed_tools: list[str]) -> WorkerResult:
        # NOTE: --bare bewusst NICHT gesetzt. In Claude Code >=2.1.x überspringt --bare
        # das Laden der Abo-Credentials (~/.claude/.credentials.json) → "Not logged in".
        # Der saubere Worker-Context (kein Projekt-CLAUDE.md/Hooks) wird stattdessen über
        # ein Run-Workdir AUSSERHALB jedes CLAUDE.md-Baums erreicht (--runs-dir außerhalb
        # des Repos). Mit dem Docker/API-Backend (Folge-Spec) kommt echte Isolation zurück.
        # shutil.which: löst auf Windows den npm-Shim (claude.cmd) auf — nackte
        # ["claude", ...] findet subprocess dort nicht (WinError 2).
        # Prompt via STDIN statt argv: der .cmd-Shim re-parst die Kommandozeile
        # durch cmd.exe — doppelte Anführungszeichen im Prompt (JSON-Beispiele!)
        # verschieben dabei die Argumentgrenzen und verschlucken Flags wie
        # --output-format/--allowedTools (Worker lief dann als Text-Chat ohne Tools).
        # Kontext-Isolation (docs/harness-overhead.md, gemessen 18.09., Decker 14.09.):
        # --allowedTools steuert nur die Auto-Freigabe — die CLI schickt trotzdem
        # ALLE Built-in-Tool-Schemas; erst --tools begrenzt sie. --setting-sources ""
        # kappt User-Plugins/Skills/Settings, --strict-mcp-config + leere Config
        # kappt Account-MCP-Server, --no-chrome die Chrome-Integration. Ergebnis:
        # identischer Worker-Kontext unabhängig vom Betreiber-Account (Störvariable),
        # Sockel ~42k → ~15k Cache-Tokens/Call. Basis-Systemprompt bleibt (Limitation).
        # Auto-Memory (Befund 24.10., docs/befund-auto-memory.md): die Flags oben schalten
        # es NICHT ab — alle Runs im runs-Repo teilten sich ein Projekt-Gedaechtnis, spaetere
        # Runs lasen Lehren frueherer. Doppelt gesperrt: Setting + Env (Kanarien-Test).
        claude_bin = shutil.which("claude") or "claude"
        tools = ",".join(allowed_tools)
        cmd = [claude_bin, "-p",
               "--output-format", "json",
               "--model", self.model,
               "--max-turns", str(max_turns),
               "--tools", tools,
               "--allowedTools", tools,
               "--setting-sources", "",
               "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--no-chrome",
               "--settings", json.dumps(WORKER_SETTINGS),
               "--effort", WORKER_EFFORT]
        if resume:
            cmd += ["--resume", session_id]
        else:
            cmd += ["--session-id", session_id]
        if budget_usd is not None:
            cmd += ["--max-budget-usd", str(budget_usd)]

        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", input=prompt,
                              env=worker_env(cwd))
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            blob = f"{proc.stderr[:500]} {proc.stdout[:500]}"
            if _looks_rate_limited(blob):
                raise RateLimitError(
                    f"claude -p: Kontingent/Rate-Limit (rc={proc.returncode}): "
                    f"{blob.strip()[:300]}") from e
            raise BackendError(
                f"claude -p lieferte kein JSON (rc={proc.returncode}): "
                f"{proc.stderr[:500]}") from e

        # is_error: Harness-Fehler, außer der Worker hat sein eigenes Turn-/USD-
        # Budget verbraucht (Agent-Verhalten → Gate entscheidet). Kontingent-Ende
        # separat, damit run.py/Runner warten + wiederholen statt werten.
        if data.get("is_error") and data.get("subtype") not in _AGENT_LIMIT_SUBTYPES:
            msg = f"{data.get('result', '')} {proc.stderr[:300]}".strip()
            if _looks_rate_limited(msg):
                raise RateLimitError(f"claude -p: Kontingent/Rate-Limit: {msg[:300]}")
            raise BackendError(
                f"claude -p is_error (subtype={data.get('subtype')}): {msg[:300]}")

        u = data.get("usage", {})
        usage = TokenUsage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_input_tokens", 0),
            cache_creation_tokens=u.get("cache_creation_input_tokens", 0),
        )
        return WorkerResult(
            text=data.get("result", ""),
            usage=usage,
            latency_ms=float(data.get("duration_api_ms", 0.0)),
            session_id=session_id,
            raw=data,
        )
