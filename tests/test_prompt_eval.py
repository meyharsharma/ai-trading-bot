"""Prompt eval harness — measure decision variance across repeat runs."""
from __future__ import annotations

from typing import Any

import pytest

from agent.state import Decision, utcnow

from backtest.prompt_eval import (
    Snapshot,
    evaluate_many,
    evaluate_snapshot,
    format_eval,
    overall_consistency,
)


class _DeterministicStrategist:
    """Returns the same decision every call — perfect consistency."""

    def decide(self, *, symbol, price, signals, portfolio, recent_decisions=None):
        return Decision(
            timestamp=utcnow(),
            symbol=symbol,
            action="BUY",
            size_pct=0.10,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            reasoning="always buy",
            signals={k: float(v) for k, v in signals.items()},
            model="det",
        )


class _AlternatingStrategist:
    """Flip-flops BUY/HOLD on every call — 50% consistency."""

    def __init__(self) -> None:
        self.i = 0

    def decide(self, *, symbol, price, signals, portfolio, recent_decisions=None):
        action = "BUY" if self.i % 2 == 0 else "HOLD"
        size = 0.10 if action == "BUY" else 0.0
        self.i += 1
        return Decision(
            timestamp=utcnow(),
            symbol=symbol,
            action=action,  # type: ignore[arg-type]
            size_pct=size,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            reasoning="alt",
            signals={},
            model="alt",
        )


def _snapshot(label: str = "btc-uptrend") -> Snapshot:
    return Snapshot(
        label=label,
        symbol="BTC/USD",
        price=65000.0,
        signals={"ma_fast": 65100.0, "ma_slow": 64800.0, "rsi": 58.0},
        portfolio={"cash_usd": 1000.0, "equity_usd": 1000.0, "positions": []},
    )


def test_deterministic_strategist_has_full_consistency() -> None:
    rep = evaluate_snapshot(_DeterministicStrategist(), _snapshot(), n_runs=10)
    assert rep.n == 10
    assert rep.dominant_action == "BUY"
    assert rep.action_consistency == pytest.approx(1.0)
    assert rep.size_pct_std == pytest.approx(0.0)
    assert rep.action_counts == {"BUY": 10}


def test_alternating_strategist_is_half_consistent() -> None:
    rep = evaluate_snapshot(_AlternatingStrategist(), _snapshot(), n_runs=10)
    assert rep.action_consistency == pytest.approx(0.5)
    assert set(rep.action_counts.keys()) == {"BUY", "HOLD"}
    # Size mean = (5*0.10 + 5*0.0) / 10 = 0.05
    assert rep.size_pct_mean == pytest.approx(0.05)
    assert rep.size_pct_std > 0.0


def test_evaluate_many_and_overall_consistency() -> None:
    snaps = [_snapshot("a"), _snapshot("b"), _snapshot("c")]
    reports = evaluate_many(_DeterministicStrategist(), snaps, n_runs=5)
    assert len(reports) == 3
    assert overall_consistency(reports) == pytest.approx(1.0)


def test_format_eval_includes_labels() -> None:
    reports = evaluate_many(_DeterministicStrategist(), [_snapshot("btc-up")], n_runs=4)
    text = format_eval(reports)
    assert "btc-up" in text
    assert "BUY" in text


def test_evaluate_snapshot_rejects_zero_runs() -> None:
    with pytest.raises(ValueError):
        evaluate_snapshot(_DeterministicStrategist(), _snapshot(), n_runs=0)


def test_overall_consistency_empty_returns_zero() -> None:
    assert overall_consistency([]) == 0.0
