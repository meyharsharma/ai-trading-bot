"""
ClaudeCodeStrategist — drop-in replacement for `LLMStrategist` that shells
out to the local `claude` CLI binary instead of calling the Anthropic SDK.

Why this exists
---------------
The hackathon doesn't require us to manage a separate Anthropic API key. The
user is already signed in to Claude Code on this machine, so we just spawn
`claude --print` per cycle and reuse those credentials. No API key, no
billing setup, no extra environment variables.

Design
------
* Same `decide()` / `critique()` interface as `LLMStrategist`, so
  `SelfCritiquingStrategist` accepts it without any changes (duck typing).
* Sync method that uses `subprocess.run()`. Cycle latency is ~1–3s per call;
  for a 5-minute loop interval that's negligible. Backtests use `LLMStrategist`
  with a fake client, so the subprocess overhead never enters the hot path.
* Uses `claude --json-schema <schema>` to force the model to emit a JSON
  payload that matches the existing `DECISION_TOOL["input_schema"]` (and
  `CRITIQUE_TOOL["input_schema"]` for the second pass). This is the same
  schema enforcement we got from anthropic SDK tool-use; we just trade
  `tool_use` blocks for an out-of-process validated JSON string.
* We deliberately do NOT use `--bare`: that flag forces auth strictly through
  `ANTHROPIC_API_KEY`/apiKeyHelper and refuses to read the user's existing
  Claude Code OAuth login, which is the whole point of this strategist. We
  instead pass `--no-session-persistence` (no clutter in `--resume` history)
  and `--disable-slash-commands` (no skill auto-loading inside cycle calls).
* On any failure (CLI not found, timeout, malformed JSON, schema mismatch)
  we return a conservative HOLD with the failure reason in `reasoning`.
  This matches `LLMStrategist`'s pattern: HOLD is always safe and the loop
  keeps running.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agent.brain.llm_strategist import Critique
from agent.brain.prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_TOOL,
    DECISION_TOOL,
    SYSTEM_PROMPT,
    render_critique_prompt,
    render_user_prompt,
)
from agent.state import Decision, Symbol, utcnow

log = logging.getLogger(__name__)


# 90s ceiling per call. The model usually finishes in 1–3s; longer than 90s
# means something is wrong (network hiccup, hung CLI) and we'd rather HOLD
# this cycle than block the loop indefinitely.
DEFAULT_TIMEOUT_S = 90.0

# System-prompt suffix that overrides the tool-use instruction baked into
# `SYSTEM_PROMPT` / `CRITIQUE_SYSTEM_PROMPT`. The shared prompts tell the
# model "call the submit_decision tool" — but in CLI mode there is no tool,
# only a `--json-schema` constraint, and without this override the model
# happily produces a natural-language summary like "Submitted BUY decision
# for BTC/USD: 10% size, 4% stop". This suffix replaces that instruction.
JSON_OUTPUT_OVERRIDE = """\

OUTPUT FORMAT (CLI mode override):
Ignore any instruction in the prompt above that asks you to call a tool. There
is no tool. Instead, respond with a single JSON object that conforms to the
schema enforced by --json-schema. Output ONLY the JSON object — no preamble,
no markdown code fence, no explanation, no trailing text.
"""

# Trailer appended to the *user* prompt. The shared `render_user_prompt`
# template ends with "Call `submit_decision`..." which is the wrong contract
# for CLI mode. We can't (and shouldn't) edit the shared template — it's used
# by `LLMStrategist` against the real anthropic SDK where the tool exists. So
# we strip that line and append a JSON-mode instruction instead.
USER_PROMPT_JSON_TRAILER = (
    "\n\nRespond now with the JSON object that matches the --json-schema. "
    "Output ONLY the JSON. No prose, no markdown, no code fence."
)


def _strip_tool_call_line(prompt: str) -> str:
    """Drop the trailing 'Call `submit_decision`/`submit_critique`...' line."""
    lines = prompt.rstrip().splitlines()
    while lines and (
        "submit_decision" in lines[-1]
        or "submit_critique" in lines[-1]
        or lines[-1].strip() == ""
    ):
        lines.pop()
    return "\n".join(lines)

# Flag combo we always pass. Documented at top of file. Order matters only
# for readability; the CLI accepts them anywhere before the trailing prompt.
BASE_FLAGS: tuple[str, ...] = (
    "--print",
    "--no-session-persistence",
    "--disable-slash-commands",
    "--max-turns", "3",
)
# Back-compat alias kept so older test imports keep working.
BARE_FLAGS = BASE_FLAGS


class ClaudeCodeUnavailableError(RuntimeError):
    """Raised by `is_available()` callers if the binary can't be found."""


@dataclass
class ClaudeCodeConfig:
    binary: str = "claude"
    model: str = "claude-opus-4-6"
    timeout_s: float = DEFAULT_TIMEOUT_S
    extra_args: tuple[str, ...] = ()


class ClaudeCodeStrategist:
    """LLMStrategist-shaped wrapper around the local `claude` CLI."""

    def __init__(self, config: ClaudeCodeConfig | None = None) -> None:
        self._config = config or ClaudeCodeConfig()
        self.model = self._config.model

    # ---------- public ----------

    def decide(
        self,
        *,
        symbol: Symbol,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
        recent_decisions: list[dict[str, Any]] | None = None,
    ) -> Decision:
        user_prompt = render_user_prompt(
            symbol=symbol,
            price=price,
            signals=signals,
            portfolio=portfolio,
            recent_decisions=recent_decisions,
        )
        user_prompt = _strip_tool_call_line(user_prompt) + USER_PROMPT_JSON_TRAILER
        try:
            payload = self._invoke(
                system_prompt=SYSTEM_PROMPT + JSON_OUTPUT_OVERRIDE,
                user_prompt=user_prompt,
                schema=DECISION_TOOL["input_schema"],
            )
        except _ClaudeCallError as exc:
            log.warning("claude CLI decide failed: %s", exc)
            return self._hold(symbol, signals, str(exc))

        try:
            return Decision(
                timestamp=utcnow(),
                symbol=payload["symbol"],
                action=payload["action"],
                size_pct=float(payload["size_pct"]),
                stop_loss_pct=float(payload["stop_loss_pct"]),
                take_profit_pct=float(payload["take_profit_pct"]),
                reasoning=str(payload["reasoning"]),
                signals={k: float(v) for k, v in (payload.get("signals") or {}).items()},
                model=self.model,
            )
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            log.warning("claude CLI decision failed schema validation: %s", exc)
            return self._hold(symbol, signals, f"Schema validation failed: {exc}")

    def critique(
        self,
        decision: Decision,
        *,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
    ) -> Critique:
        prompt = render_critique_prompt(
            decision=decision.model_dump(mode="json"),
            price=price,
            signals=signals,
            portfolio=portfolio,
        )
        prompt = _strip_tool_call_line(prompt) + USER_PROMPT_JSON_TRAILER
        try:
            payload = self._invoke(
                system_prompt=CRITIQUE_SYSTEM_PROMPT + JSON_OUTPUT_OVERRIDE,
                user_prompt=prompt,
                schema=CRITIQUE_TOOL["input_schema"],
            )
        except _ClaudeCallError as exc:
            log.warning("claude CLI critique failed, defaulting to ACCEPT: %s", exc)
            return Critique(verdict="ACCEPT", weakness=f"critique unavailable: {exc}")

        verdict = payload.get("verdict")
        weakness = payload.get("weakness", "")
        if verdict not in ("ACCEPT", "REJECT") or not isinstance(weakness, str) or not weakness:
            return Critique(verdict="ACCEPT", weakness="critic returned malformed payload")
        return Critique(verdict=verdict, weakness=weakness)  # type: ignore[arg-type]

    # ---------- internals ----------

    def _invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the CLI once. Returns the parsed model payload (a dict)."""
        cmd = [
            self._config.binary,
            *BASE_FLAGS,
            "--model", self._config.model,
            "--system-prompt", system_prompt,
            "--output-format", "json",
            "--json-schema", json.dumps(schema, separators=(",", ":")),
            *self._config.extra_args,
            user_prompt,
        ]
        log.debug("invoking claude CLI: %d arg(s)", len(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise _ClaudeCallError(f"binary not found: {self._config.binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise _ClaudeCallError(
                f"claude CLI timed out after {self._config.timeout_s:.0f}s"
            ) from exc

        # IMPORTANT: parse the envelope BEFORE checking returncode. The CLI
        # often exits non-zero AND emits a perfectly good JSON envelope on
        # stdout (e.g. "Not logged in · Please run /login"). The envelope's
        # `result` field is the human-readable message we want to surface.
        envelope_text = (result.stdout or "").strip()
        envelope: dict[str, Any] | None = None
        if envelope_text:
            try:
                parsed = json.loads(envelope_text)
                if isinstance(parsed, dict):
                    envelope = parsed
            except json.JSONDecodeError:
                envelope = None

        if envelope is not None and envelope.get("is_error"):
            human = envelope.get("result") or envelope.get("subtype") or "unknown error"
            raise _ClaudeCallError(f"claude CLI: {human}")

        if result.returncode != 0:
            stderr = (result.stderr or "").strip().splitlines()
            tail = "; ".join(stderr[-3:]) or "no stderr"
            raise _ClaudeCallError(f"claude CLI exit {result.returncode}: {tail}")

        if envelope is None:
            raise _ClaudeCallError(
                f"claude CLI produced unparseable stdout: {envelope_text[:200]!r}"
            )

        # When `--json-schema` is honored, the CLI already parsed the model's
        # JSON for us and put it in `structured_output`. Prefer that — it's
        # always a dict and skips a redundant parse.
        structured = envelope.get("structured_output")
        if isinstance(structured, dict):
            return structured

        # Fallback for older CLI builds: the `result` field is a string that
        # contains the JSON document conforming to our schema.
        result_str = envelope.get("result")
        if not isinstance(result_str, str) or not result_str.strip():
            raise _ClaudeCallError("envelope has neither structured_output nor result")

        payload = _extract_json_object(result_str)
        if payload is None:
            raise _ClaudeCallError(
                f"could not extract JSON object from result: {result_str[:200]!r}"
            )
        return payload

    def _hold(
        self,
        symbol: Symbol,
        signals: dict[str, float],
        reason: str,
    ) -> Decision:
        return Decision(
            timestamp=utcnow(),
            symbol=symbol,
            action="HOLD",
            size_pct=0.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            reasoning=f"Defaulted to HOLD: {reason}",
            signals={k: float(v) for k, v in signals.items()},
            model=self.model,
        )


# ---------- helpers ----------

class _ClaudeCallError(RuntimeError):
    """Internal: any failure path the strategist should map to HOLD."""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Defensive parser for the model's JSON output.

    `--json-schema` should give us a clean JSON document, but the CLI has
    historically wrapped the response in a markdown code fence on some
    versions. We try the cheap path first, then strip a fence, then try a
    last-resort brace-balanced extraction.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Strip a leading code fence (```json ... ``` or ``` ... ```).
    if text.startswith("```"):
        body = text[3:]
        if body.lower().startswith("json"):
            body = body[4:]
        body = body.lstrip("\n")
        if body.endswith("```"):
            body = body[:-3]
        try:
            parsed = json.loads(body.strip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    # Last resort: find the first balanced {...} block.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start : i + 1]
                try:
                    parsed = json.loads(snippet)
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def is_available(binary: str = "claude") -> bool:
    """Cheap check used by `build_runtime` to pick this strategist."""
    return shutil.which(binary) is not None or os.path.isabs(binary) and os.access(binary, os.X_OK)
