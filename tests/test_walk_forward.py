"""Walk-forward backtest harness."""
from __future__ import annotations

import pytest

from agent.risk.gate import RiskConfig

from backtest.baselines import BuyAndHoldStrategist
from backtest.runner import MACrossStrategist, synthetic_bars
from backtest.walk_forward import format_walk_forward, run_walk_forward


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


def test_walk_forward_produces_expected_window_count(risk_config: RiskConfig) -> None:
    bars = synthetic_bars(n=600, drift=0.0005, amplitude=0.05, period=120)
    result = run_walk_forward(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        train_bars=200,
        test_bars=100,
        strategist_factory=lambda: MACrossStrategist(rsi_overbought=95.0),
        ma_fast=5,
        ma_slow=20,
    )
    # 600 bars, 200 train + 100 test, step 100 (default = test_bars)
    # windows at i=0 (covers 0..300), i=100 (100..400), i=200 (200..500), i=300 (300..600)
    assert len(result.windows) == 4
    assert result.aggregate is not None
    for w in result.windows:
        assert w.num_test_bars >= 100
        assert w.metrics is not None
        assert w.test_end > w.test_start


def test_walk_forward_factory_resets_state(risk_config: RiskConfig) -> None:
    """BuyAndHoldStrategist tracks an internal _bought set; the factory must
    re-instantiate per window so each test window gets a fresh entry."""
    bars = synthetic_bars(n=500)
    result = run_walk_forward(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        train_bars=100,
        test_bars=50,
        strategist_factory=BuyAndHoldStrategist,
        ma_fast=5,
        ma_slow=20,
    )
    # Buy-and-hold positions may not *close* within a 50-bar test window
    # (no stop/TP hit), so num_trades stays 0. Instead we verify state reset
    # by confirming each window's equity curve actually moves — it can only
    # move if a position was opened, and that requires a fresh _bought set.
    assert len(result.windows) >= 4
    moving_windows = sum(
        1 for w in result.windows if abs(w.metrics.total_return) > 1e-9
    )
    # If state weren't reset, only the very first window would open a position.
    assert moving_windows >= 2


def test_walk_forward_rejects_bad_inputs(risk_config: RiskConfig) -> None:
    bars = synthetic_bars(n=100)
    with pytest.raises(ValueError):
        run_walk_forward(
            symbol="BTC/USD",
            bars=bars,
            risk_config=risk_config,
            train_bars=0,
            test_bars=10,
            strategist_factory=MACrossStrategist,
        )
    with pytest.raises(ValueError):
        run_walk_forward(
            symbol="BTC/USD",
            bars=[],
            risk_config=risk_config,
            train_bars=10,
            test_bars=10,
            strategist_factory=MACrossStrategist,
        )


def test_walk_forward_handles_too_few_bars(risk_config: RiskConfig) -> None:
    """If history is shorter than train+test, we get zero windows but no crash."""
    bars = synthetic_bars(n=50)
    result = run_walk_forward(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        train_bars=100,
        test_bars=50,
        strategist_factory=MACrossStrategist,
    )
    assert result.windows == []
    assert result.aggregate is None


def test_format_walk_forward(risk_config: RiskConfig) -> None:
    bars = synthetic_bars(n=400)
    result = run_walk_forward(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        train_bars=100,
        test_bars=50,
        strategist_factory=lambda: MACrossStrategist(rsi_overbought=95.0),
        ma_fast=5,
        ma_slow=20,
    )
    text = format_walk_forward(result)
    assert "window" in text
    assert "return" in text


def test_walk_forward_as_dict_serializable(risk_config: RiskConfig) -> None:
    import json

    bars = synthetic_bars(n=400)
    result = run_walk_forward(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        train_bars=100,
        test_bars=50,
        strategist_factory=lambda: MACrossStrategist(rsi_overbought=95.0),
        ma_fast=5,
        ma_slow=20,
    )
    # Must be JSON-serializable for the demo / artifact pipeline.
    json.dumps(result.as_dict())
