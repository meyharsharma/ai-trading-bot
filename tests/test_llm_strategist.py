"""LLM strategist tests using a fake Anthropic client.

We don't hit the network. We feed canned responses (in both SDK-object and
plain-dict shapes) and verify that the strategist correctly:
  - parses a successful tool_use into a `Decision`
  - falls back to HOLD on missing tool calls, schema errors, and exceptions
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.brain.llm_strategist import LLMStrategist
from agent.brain.prompts import DECISION_TOOL, render_user_prompt
from agent.state import Decision


# ---------- fakes ----------

class _FakeMessages:
    def __init__(self, response: Any | Exception) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeAnthropic:
    def __init__(self, response: Any | Exception) -> None:
        self.messages = _FakeMessages(response)


def _tool_use_response(input_dict: dict[str, Any]) -> SimpleNamespace:
    """Mimic the SDK's response.content[*] tool_use block as a SimpleNamespace."""
    block = SimpleNamespace(
        type="tool_use",
        name=DECISION_TOOL["name"],
        input=input_dict,
    )
    return SimpleNamespace(content=[block])


def _ctx() -> dict[str, Any]:
    return {
        "symbol": "BTC/USD",
        "price": 65000.0,
        "signals": {"ma_fast": 65100.0, "ma_slow": 64800.0, "rsi": 58.0},
        "portfolio": {"cash_usd": 1000.0, "equity_usd": 1000.0, "positions": []},
    }


# ---------- happy path ----------

def test_decide_parses_tool_use_into_decision() -> None:
    response = _tool_use_response(
        {
            "symbol": "BTC/USD",
            "action": "BUY",
            "size_pct": 0.10,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "reasoning": "MA fast crossed above slow with RSI 58.",
            "signals": {"ma_fast": 65100.0, "ma_slow": 64800.0, "rsi": 58.0},
        }
    )
    client = _FakeAnthropic(response)
    strat = LLMStrategist(client, model="claude-test")

    d = strat.decide(**_ctx())
    assert isinstance(d, Decision)
    assert d.action == "BUY"
    assert d.size_pct == pytest.approx(0.10)
    assert d.model == "claude-test"
    assert "MA fast" in d.reasoning

    # Confirm the strategist used tool-forced output and the right system prompt.
    kwargs = client.messages.last_kwargs
    assert kwargs is not None
    assert kwargs["tools"][0]["name"] == DECISION_TOOL["name"]
    assert kwargs["tool_choice"]["type"] == "tool"
    assert kwargs["model"] == "claude-test"


def test_decide_accepts_dict_shaped_response() -> None:
    response = {
        "content": [
            {
                "type": "tool_use",
                "name": DECISION_TOOL["name"],
                "input": {
                    "symbol": "ETH/USD",
                    "action": "HOLD",
                    "size_pct": 0.0,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "reasoning": "No edge.",
                    "signals": {"rsi": 50.0},
                },
            }
        ]
    }
    strat = LLMStrategist(_FakeAnthropic(response))
    d = strat.decide(**_ctx())
    assert d.action == "HOLD"
    assert d.symbol == "ETH/USD"


# ---------- failure modes ----------

def test_missing_tool_call_falls_back_to_hold() -> None:
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="just a chat")])
    strat = LLMStrategist(_FakeAnthropic(response))
    d = strat.decide(**_ctx())
    assert d.action == "HOLD"
    assert "no tool call" in d.reasoning.lower()


def test_schema_violation_falls_back_to_hold() -> None:
    bad = _tool_use_response(
        {
            "symbol": "BTC/USD",
            "action": "BUY",
            "size_pct": 5.0,  # > 1.0 violates the Decision schema
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "reasoning": "oops",
            "signals": {},
        }
    )
    strat = LLMStrategist(_FakeAnthropic(bad))
    d = strat.decide(**_ctx())
    assert d.action == "HOLD"
    assert "schema" in d.reasoning.lower() or "validation" in d.reasoning.lower()


def test_network_exception_falls_back_to_hold() -> None:
    strat = LLMStrategist(_FakeAnthropic(RuntimeError("boom")))
    d = strat.decide(**_ctx())
    assert d.action == "HOLD"
    assert "boom" in d.reasoning


def test_render_user_prompt_includes_signals_and_positions() -> None:
    text = render_user_prompt(
        symbol="BTC/USD",
        price=65000.0,
        signals={"ma_fast": 65100.0, "rsi": 58.0},
        portfolio={
            "cash_usd": 100.0,
            "equity_usd": 1000.0,
            "positions": [
                {
                    "symbol": "BTC/USD",
                    "quantity": 0.001,
                    "avg_entry_price": 60000.0,
                    "stop_loss_price": 58000.0,
                }
            ],
        },
    )
    assert "BTC/USD" in text
    assert "ma_fast" in text
    assert "65000" in text
    assert "submit_decision" in text
