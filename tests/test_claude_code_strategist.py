"""
Unit tests for ClaudeCodeStrategist.

We never actually invoke the `claude` binary — every test patches
`subprocess.run` so the assertions stay hermetic and fast.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from agent.brain.claude_code_strategist import (
    BARE_FLAGS,
    ClaudeCodeConfig,
    ClaudeCodeStrategist,
    _extract_json_object,
)
from agent.brain.llm_strategist import Critique
from agent.brain.prompts import DECISION_TOOL
from agent.state import Decision


# ---------- helpers ----------

DECISION_PAYLOAD = {
    "symbol": "BTC/USD",
    "action": "BUY",
    "size_pct": 0.1,
    "stop_loss_pct": 0.03,
    "take_profit_pct": 0.06,
    "reasoning": "MA fast 65000 > MA slow 64000 with RSI 58.",
    "signals": {"ma_fast": 65000.0, "ma_slow": 64000.0, "rsi": 58.0},
}


def _envelope(result_obj: dict, *, is_error: bool = False, returncode: int = 0):
    """Build a fake `subprocess.run` CompletedProcess that mimics the CLI.

    Real envelopes (when --json-schema is honored) carry both `structured_output`
    (already-parsed dict, preferred) and `result` (stringified JSON, fallback).
    We populate both so the strategist exercises whichever path it prefers.
    """
    envelope = {
        "type": "result",
        "subtype": "success" if not is_error else "error",
        "is_error": is_error,
        "result": json.dumps(result_obj),
        "structured_output": result_obj if not is_error else None,
        "session_id": "fake-session",
        "total_cost_usd": 0.0001,
    }
    return subprocess.CompletedProcess(
        args=["claude"],
        returncode=returncode,
        stdout=json.dumps(envelope),
        stderr="",
    )


def _strategist() -> ClaudeCodeStrategist:
    return ClaudeCodeStrategist(ClaudeCodeConfig(binary="claude", timeout_s=5.0))


def _decide_kwargs():
    return dict(
        symbol="BTC/USD",
        price=65000.0,
        signals={"ma_fast": 65000.0, "ma_slow": 64000.0, "rsi": 58.0},
        portfolio={
            "cash_usd": 1000.0,
            "equity_usd": 1000.0,
            "positions": [],
            "open_positions": [],
        },
    )


# ---------- decide() ----------

def test_decide_happy_path():
    s = _strategist()
    with patch("agent.brain.claude_code_strategist.subprocess.run", return_value=_envelope(DECISION_PAYLOAD)) as mock_run:
        decision = s.decide(**_decide_kwargs())

    assert isinstance(decision, Decision)
    assert decision.action == "BUY"
    assert decision.symbol == "BTC/USD"
    assert decision.size_pct == 0.1
    assert decision.model == "claude-opus-4-6"

    # The CLI was called with the right shape: --bare flags, --json-schema set,
    # --output-format json, --print, our prompt as the trailing arg.
    cmd = mock_run.call_args.args[0]
    for flag in BARE_FLAGS:
        assert flag in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--json-schema" in cmd
    assert "--system-prompt" in cmd
    # The user prompt is the last positional arg.
    assert "BTC/USD" in cmd[-1]


def test_decide_holds_on_nonzero_exit():
    s = _strategist()
    failed = subprocess.CompletedProcess(
        args=["claude"], returncode=2, stdout="", stderr="boom\n"
    )
    with patch("agent.brain.claude_code_strategist.subprocess.run", return_value=failed):
        decision = s.decide(**_decide_kwargs())
    assert decision.action == "HOLD"
    assert "exit 2" in decision.reasoning


def test_decide_holds_on_timeout():
    s = _strategist()
    with patch(
        "agent.brain.claude_code_strategist.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5),
    ):
        decision = s.decide(**_decide_kwargs())
    assert decision.action == "HOLD"
    assert "timed out" in decision.reasoning


def test_decide_holds_on_missing_binary():
    s = _strategist()
    with patch(
        "agent.brain.claude_code_strategist.subprocess.run",
        side_effect=FileNotFoundError("no such file"),
    ):
        decision = s.decide(**_decide_kwargs())
    assert decision.action == "HOLD"
    assert "binary not found" in decision.reasoning


def test_decide_holds_on_envelope_error_flag():
    """A non-success envelope must surface the human-readable `result` field
    so the operator sees the actual error (e.g. 'Not logged in')."""
    s = _strategist()
    # Override the envelope so `result` carries a sentinel string we can grep.
    error_env = subprocess.CompletedProcess(
        args=["claude"],
        returncode=1,
        stdout=json.dumps({
            "type": "result",
            "subtype": "error",
            "is_error": True,
            "result": "Not logged in · Please run /login",
        }),
        stderr="",
    )
    with patch("agent.brain.claude_code_strategist.subprocess.run", return_value=error_env):
        decision = s.decide(**_decide_kwargs())
    assert decision.action == "HOLD"
    assert "Not logged in" in decision.reasoning


def test_decide_holds_on_malformed_payload():
    s = _strategist()
    # Missing required `reasoning` field.
    bad_payload = {**DECISION_PAYLOAD}
    bad_payload.pop("reasoning")
    with patch("agent.brain.claude_code_strategist.subprocess.run", return_value=_envelope(bad_payload)):
        decision = s.decide(**_decide_kwargs())
    assert decision.action == "HOLD"
    assert "Schema validation" in decision.reasoning


def test_decide_passes_schema_to_cli():
    """The schema we hand to --json-schema must be the existing DECISION_TOOL schema."""
    s = _strategist()
    with patch("agent.brain.claude_code_strategist.subprocess.run", return_value=_envelope(DECISION_PAYLOAD)) as mock_run:
        s.decide(**_decide_kwargs())
    cmd = mock_run.call_args.args[0]
    schema_idx = cmd.index("--json-schema")
    schema = json.loads(cmd[schema_idx + 1])
    assert schema == DECISION_TOOL["input_schema"]


# ---------- critique() ----------

def test_critique_happy_path():
    s = _strategist()
    decision = Decision(
        timestamp=__import__("datetime").datetime(2026, 4, 7, tzinfo=__import__("datetime").timezone.utc),
        symbol="BTC/USD",
        action="BUY",
        size_pct=0.1,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        reasoning="trend+rsi",
        signals={"rsi": 58.0},
        model="claude-opus-4-6",
    )
    payload = {"verdict": "REJECT", "weakness": "MA crossover too recent to confirm"}
    with patch("agent.brain.claude_code_strategist.subprocess.run", return_value=_envelope(payload)):
        critique = s.critique(
            decision,
            price=65000.0,
            signals={"rsi": 58.0},
            portfolio={"cash_usd": 1000.0, "equity_usd": 1000.0, "positions": []},
        )
    assert isinstance(critique, Critique)
    assert critique.verdict == "REJECT"
    assert "too recent" in critique.weakness


def test_critique_defaults_to_accept_on_failure():
    s = _strategist()
    failed = subprocess.CompletedProcess(args=["claude"], returncode=1, stdout="", stderr="x")
    with patch("agent.brain.claude_code_strategist.subprocess.run", return_value=failed):
        critique = s.critique(
            Decision(
                timestamp=__import__("datetime").datetime(2026, 4, 7, tzinfo=__import__("datetime").timezone.utc),
                symbol="BTC/USD",
                action="BUY",
                size_pct=0.1,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                reasoning="x",
                signals={},
                model="claude-opus-4-6",
            ),
            price=65000.0,
            signals={},
            portfolio={"cash_usd": 1000.0, "equity_usd": 1000.0, "positions": []},
        )
    assert critique.verdict == "ACCEPT"
    assert "critique unavailable" in critique.weakness


# ---------- _extract_json_object ----------

def test_extract_json_object_plain():
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_fenced():
    fenced = "```json\n{\"a\": 1}\n```"
    assert _extract_json_object(fenced) == {"a": 1}


def test_extract_json_object_brace_extraction():
    noisy = 'Here is your answer: {"a": 1, "b": [2, 3]}\nthanks!'
    assert _extract_json_object(noisy) == {"a": 1, "b": [2, 3]}


def test_extract_json_object_returns_none_on_garbage():
    assert _extract_json_object("not json at all") is None
    assert _extract_json_object("[1, 2, 3]") is None  # array, not object


# ---------- works as a SelfCritiquingStrategist inner ----------

def test_works_with_self_critiquing_wrapper():
    """SelfCritiquingStrategist should accept ClaudeCodeStrategist via duck typing."""
    from agent.brain.llm_strategist import SelfCritiquingStrategist

    s = _strategist()
    wrapped = SelfCritiquingStrategist(s)  # type: ignore[arg-type]

    decide_envelope = _envelope(DECISION_PAYLOAD)
    critique_envelope = _envelope({"verdict": "ACCEPT", "weakness": "thesis is sound"})
    with patch(
        "agent.brain.claude_code_strategist.subprocess.run",
        side_effect=[decide_envelope, critique_envelope],
    ):
        decision = wrapped.decide(**_decide_kwargs())

    assert decision.action == "BUY"
    # Reasoning should have the [critic ACCEPT] marker appended.
    assert "[critic ACCEPT]" in decision.reasoning
