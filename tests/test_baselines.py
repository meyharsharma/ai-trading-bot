"""Baseline strategies + comparator."""
from __future__ import annotations

import pytest

from agent.risk.gate import RiskConfig

from backtest.baselines import (
    BuyAndHoldStrategist,
    MACrossOnlyStrategist,
    compare_strategies,
    format_comparison,
)
from backtest.runner import MACrossStrategist, synthetic_bars


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_risk_per_trade_pct=0.02,
        max_open_positions=3,
        max_position_size_pct=0.25,
        default_stop_loss_pct=0.03,
        max_stop_loss_pct=0.08,
        max_take_profit_pct=0.20,
        allow_leverage=False,
        allow_shorts=False,
    )


def test_buy_and_hold_makes_one_buy(risk_config: RiskConfig) -> None:
    bars = synthetic_bars(n=200, drift=0.001, amplitude=0.02, period=80)
    rows = compare_strategies(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        strategists={"buy-hold": BuyAndHoldStrategist()},
        ma_fast=5,
        ma_slow=20,
    )
    row = rows[0]
    # Exactly one BUY across the run, no SELLs (we never close).
    buys = sum(1 for d in row.result.decisions if d.action == "BUY")
    assert buys == 1


def test_compare_strategies_runs_all(risk_config: RiskConfig) -> None:
    bars = synthetic_bars(n=400, drift=0.0005, amplitude=0.05, period=120)
    rows = compare_strategies(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        strategists={
            "buy-hold": BuyAndHoldStrategist(),
            "ma-cross-only": MACrossOnlyStrategist(),
            "ma-cross+rsi": MACrossStrategist(rsi_overbought=95.0),
        },
        ma_fast=5,
        ma_slow=20,
        bars_per_year=365 * 24 * 12,
    )
    assert len(rows) == 3
    names = {r.name for r in rows}
    assert names == {"buy-hold", "ma-cross-only", "ma-cross+rsi"}
    for r in rows:
        assert r.metrics is not None
        assert 0.0 <= r.metrics.max_drawdown < 1.0


def test_format_comparison_prints_table(risk_config: RiskConfig) -> None:
    bars = synthetic_bars(n=200)
    rows = compare_strategies(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        strategists={"buy-hold": BuyAndHoldStrategist()},
        ma_fast=5,
        ma_slow=20,
    )
    text = format_comparison(rows)
    assert "buy-hold" in text
    assert "return" in text
    assert "maxDD" in text


def test_ma_cross_only_skips_when_no_history(risk_config: RiskConfig) -> None:
    """Before warmup, the strategist must HOLD (not crash on missing signals)."""
    s = MACrossOnlyStrategist()
    d = s.decide(
        symbol="BTC/USD",
        price=100.0,
        signals={},
        portfolio={"cash_usd": 1000.0, "equity_usd": 1000.0, "positions": []},
    )
    assert d.action == "HOLD"
    assert d.size_pct == 0.0
