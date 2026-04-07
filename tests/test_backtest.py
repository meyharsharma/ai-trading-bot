"""End-to-end backtest tests using the deterministic MA-cross strategist
and synthetic bars. Verifies the runner wires brain → risk → stub-exec
correctly and the metrics module produces sensible numbers."""
from __future__ import annotations

import pytest

from agent.risk.gate import RiskConfig
from agent.state import Decision, utcnow

from backtest.metrics import (
    BacktestMetrics,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)
from backtest.runner import (
    Bar,
    MACrossStrategist,
    StubExecutionAdapter,
    run_backtest,
    synthetic_bars,
)


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


# ---------- metrics ----------

def test_total_return_basic() -> None:
    assert total_return([100, 110]) == pytest.approx(0.10)
    assert total_return([100]) == 0.0
    assert total_return([]) == 0.0


def test_max_drawdown() -> None:
    assert max_drawdown([100, 120, 90, 110]) == pytest.approx((120 - 90) / 120)
    assert max_drawdown([100, 100, 100]) == 0.0


def test_win_rate() -> None:
    assert win_rate([1, -1, 2, -3, 0]) == pytest.approx(2 / 5)
    assert win_rate([]) == 0.0


def test_sharpe_handles_constant_curve() -> None:
    assert sharpe_ratio([100, 100, 100, 100]) == 0.0


def test_sharpe_positive_for_uptrend() -> None:
    curve = [100 * (1.01 ** i) for i in range(50)]
    assert sharpe_ratio(curve, bars_per_year=252) > 0


def test_compute_metrics_round_trip() -> None:
    m = compute_metrics([100, 105, 110], [5, -2, 3], bars_per_year=252)
    assert isinstance(m, BacktestMetrics)
    assert m.num_trades == 3
    assert m.final_equity == 110
    assert m.total_return == pytest.approx(0.10)


# ---------- stub adapter ----------

def test_stub_adapter_open_close_pnl() -> None:
    a = StubExecutionAdapter(starting_cash_usd=1000.0, fee_bps=0.0)
    fill = a.open_long(
        symbol="BTC/USD",
        notional_usd=500.0,
        price=100.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        ts=utcnow(),
    )
    assert fill is not None
    assert "BTC/USD" in a.positions
    assert a.cash_usd == pytest.approx(500.0)

    close_fill = a.close(symbol="BTC/USD", price=110.0, ts=utcnow())
    assert close_fill is not None
    assert "BTC/USD" not in a.positions
    # PnL = (110-100)*5 = 50
    assert a.realized_pnl_usd == pytest.approx(50.0)
    assert a.closed_trade_pnls == [pytest.approx(50.0)]
    assert a.cash_usd == pytest.approx(1050.0)


def test_stub_adapter_fee_deducted() -> None:
    a = StubExecutionAdapter(starting_cash_usd=1000.0, fee_bps=100.0)  # 1%
    a.open_long(
        symbol="BTC/USD",
        notional_usd=500.0,
        price=100.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        ts=utcnow(),
    )
    # Fee = 5, cash should be 1000 - 500 - 5 = 495
    assert a.cash_usd == pytest.approx(495.0)


# ---------- runner ----------

def test_run_backtest_executes_pipeline(risk_config: RiskConfig) -> None:
    # Long sinusoid period so MA(slow=20) can still swing across MA(fast=5).
    bars = synthetic_bars(n=400, start_price=100.0, drift=0.0005, amplitude=0.05, period=120)
    result = run_backtest(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        ma_fast=5,
        ma_slow=20,
        rsi_period=14,
        # rsi_overbought=95 — synthetic price is too smooth, RSI peaks ~70 at
        # every up-cross which would block every entry. The threshold relaxation
        # lets the test exercise the trade path; production default stays 70.
        strategist=MACrossStrategist(
            size_pct=0.20,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            rsi_overbought=95.0,
        ),
        starting_cash_usd=1000.0,
        fee_bps=10.0,
        bars_per_year=365 * 24 * 12,
    )
    assert result.metrics is not None
    assert len(result.equity_curve) == len(bars)
    # Strategist should have produced some decisions after warmup.
    assert len(result.decisions) > 0
    # MA-cross + sinusoid should generate at least one closed trade.
    assert result.metrics.num_trades >= 1
    # Equity must remain bounded — drawdown < 100% sanity check.
    assert 0.0 <= result.metrics.max_drawdown < 1.0


def test_run_backtest_with_holds_only_strategist(risk_config: RiskConfig) -> None:
    """A strategist that always HOLDs leaves equity flat at starting cash."""

    class AlwaysHold:
        def decide(self, *, symbol, price, signals, portfolio, recent_decisions=None):
            return Decision(
                timestamp=utcnow(),
                symbol=symbol,
                action="HOLD",
                size_pct=0.0,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                reasoning="never trade",
                signals={},
                model="hold",
            )

    bars = synthetic_bars(n=120)
    result = run_backtest(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        strategist=AlwaysHold(),
        starting_cash_usd=1000.0,
        fee_bps=10.0,
    )
    assert result.metrics is not None
    assert result.metrics.num_trades == 0
    assert result.equity_curve[-1] == pytest.approx(1000.0)
    assert result.metrics.max_drawdown == 0.0


def test_run_backtest_respects_risk_gate(risk_config: RiskConfig) -> None:
    """A strategist asking for an oversized BUY gets clamped, not honored raw."""

    class OversizeBuy(MACrossStrategist):
        def decide(self, **kw):  # type: ignore[override]
            d = super().decide(**kw)
            if d.action == "BUY":
                # Try to buy 90% of capital — must be clamped to 25%.
                return d.model_copy(update={"size_pct": 0.90})
            return d

    bars = synthetic_bars(n=400, drift=0.0005, amplitude=0.05, period=120)
    result = run_backtest(
        symbol="BTC/USD",
        bars=bars,
        risk_config=risk_config,
        ma_fast=5,
        ma_slow=20,
        strategist=OversizeBuy(
            size_pct=0.20,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            rsi_overbought=95.0,
        ),
        starting_cash_usd=1000.0,
        fee_bps=10.0,
    )
    # Every passed BUY must have been clamped to <= max_position_size_pct.
    buys = [r for r in result.risked if r.passed and r.decision.action == "BUY"]
    assert buys, "expected at least one BUY in this synthetic series"
    for r in buys:
        assert r.final_size_pct <= 0.25 + 1e-9
        assert r.clamped


def test_runner_requires_bars(risk_config: RiskConfig) -> None:
    with pytest.raises(ValueError):
        run_backtest(symbol="BTC/USD", bars=[], risk_config=risk_config)
