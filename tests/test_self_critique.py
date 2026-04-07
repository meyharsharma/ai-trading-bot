"""Self-critique pass: ACCEPT must keep the original action; REJECT must
downgrade to HOLD with the critique embedded in reasoning."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.brain.llm_strategist import (
    Critique,
    LLMStrategist,
    SelfCritiquingStrategist,
)
from agent.brain.prompts import CRITIQUE_TOOL, DECISION_TOOL


class _ScriptedMessages:
    """Replays a list of pre-baked responses, one per .create() call."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("ScriptedMessages exhausted")
        return self._responses.pop(0)


class _ScriptedClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _ScriptedMessages(responses)


def _decision_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name=DECISION_TOOL["name"],
                input={
                    "symbol": "BTC/USD",
                    "action": "BUY",
                    "size_pct": 0.10,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "reasoning": "MA up-cross with healthy RSI.",
                    "signals": {"ma_fast": 65100.0, "ma_slow": 64800.0, "rsi": 55.0},
                },
            )
        ]
    )


def _critique_response(verdict: str, weakness: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name=CRITIQUE_TOOL["name"],
                input={"verdict": verdict, "weakness": weakness},
            )
        ]
    )


def _ctx() -> dict[str, Any]:
    return {
        "symbol": "BTC/USD",
        "price": 65000.0,
        "signals": {"ma_fast": 65100.0, "ma_slow": 64800.0, "rsi": 55.0},
        "portfolio": {"cash_usd": 1000.0, "equity_usd": 1000.0, "positions": []},
    }


def test_critique_accept_preserves_action() -> None:
    client = _ScriptedClient(
        [_decision_response(), _critique_response("ACCEPT", "trend is shallow but valid")]
    )
    strat = SelfCritiquingStrategist(LLMStrategist(client))
    d = strat.decide(**_ctx())
    assert d.action == "BUY"
    assert d.size_pct == pytest.approx(0.10)
    assert "[critic ACCEPT]" in d.reasoning
    assert "shallow" in d.reasoning


def test_critique_reject_downgrades_to_hold() -> None:
    client = _ScriptedClient(
        [_decision_response(), _critique_response("REJECT", "MA gap is razor-thin and RSI mid-range")]
    )
    strat = SelfCritiquingStrategist(LLMStrategist(client))
    d = strat.decide(**_ctx())
    assert d.action == "HOLD"
    assert d.size_pct == 0.0
    assert "REJECTED" in d.reasoning
    assert "razor-thin" in d.reasoning
    # Original reasoning preserved for the audit trail.
    assert "MA up-cross" in d.reasoning


def test_critique_skipped_for_hold_decision() -> None:
    hold_response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name=DECISION_TOOL["name"],
                input={
                    "symbol": "BTC/USD",
                    "action": "HOLD",
                    "size_pct": 0.0,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "reasoning": "no edge",
                    "signals": {"rsi": 50.0},
                },
            )
        ]
    )
    client = _ScriptedClient([hold_response])
    strat = SelfCritiquingStrategist(LLMStrategist(client))
    d = strat.decide(**_ctx())
    assert d.action == "HOLD"
    # Only one LLM call should have been made — no critique.
    assert len(client.messages.calls) == 1


def test_critique_failure_defaults_to_accept() -> None:
    """If the critic call crashes, we keep the original decision (don't kill all trades)."""

    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return _decision_response()
            raise RuntimeError("network down")

    client = SimpleNamespace(messages=_Flaky())
    strat = SelfCritiquingStrategist(LLMStrategist(client))
    d = strat.decide(**_ctx())
    assert d.action == "BUY"
    assert "[critic ACCEPT]" in d.reasoning
    assert "critique unavailable" in d.reasoning


def test_critique_dataclass_is_frozen() -> None:
    c = Critique(verdict="REJECT", weakness="bad")
    with pytest.raises(Exception):
        c.verdict = "ACCEPT"  # type: ignore[misc]
