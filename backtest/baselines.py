"""
Baseline strategies for side-by-side comparison.

The whole point of these is to have something dumb to lose to. "Our LLM beat
buy-and-hold by X% with Y% lower max drawdown" is a much stronger pitch than
a standalone Sharpe number — judges instinctively distrust unanchored metrics.

Available baselines:
    BuyAndHoldStrategist     — buys once on the first warm bar and never sells
    MACrossOnlyStrategist    — pure MA(fast) vs MA(slow) cross, no RSI filter

Use `compare_strategies(...)` to run a dict of named strategists across the
same bars and get back a dict of `BacktestMetrics` for printing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.risk.gate import RiskConfig
from agent.state import Decision, Symbol, utcnow

from backtest.metrics import BacktestMetrics
from backtest.runner import Bar, BacktestResult, Strategist, run_backtest


# ---------- buy and hold ----------

class BuyAndHoldStrategist:
    """The hardest baseline to beat in a bull market. Buys on first call, holds forever."""

    def __init__(
        self,
        *,
        size_pct: float = 0.25,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20,
        model_name: str = "buy-and-hold",
    ) -> None:
        self.size_pct = size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.model_name = model_name
        self._bought: set[str] = set()

    def decide(
        self,
        *,
        symbol: Symbol,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
        recent_decisions: list[dict[str, Any]] | None = None,
    ) -> Decision:
        holding = symbol in {p.get("symbol") for p in portfolio.get("positions", []) or []}
        if symbol in self._bought or holding:
            action = "HOLD"
            reasoning = "Already long, holding."
            size = 0.0
        else:
            self._bought.add(symbol)
            action = "BUY"
            reasoning = f"Initial buy-and-hold entry at {price:.2f}."
            size = self.size_pct
        return Decision(
            timestamp=utcnow(),
            symbol=symbol,
            action=action,  # type: ignore[arg-type]
            size_pct=size,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            reasoning=reasoning,
            signals={k: float(v) for k, v in signals.items()},
            model=self.model_name,
        )


# ---------- MA-cross only ----------

class MACrossOnlyStrategist:
    """
    Dumb baseline: BUY on MA up-cross, SELL on MA down-cross. No RSI filter,
    no momentum check, no overbought guard. The point is to give the LLM
    something obvious to outperform.
    """

    def __init__(
        self,
        *,
        size_pct: float = 0.20,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        model_name: str = "ma-cross-only",
    ) -> None:
        self.size_pct = size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.model_name = model_name

    def decide(
        self,
        *,
        symbol: Symbol,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
        recent_decisions: list[dict[str, Any]] | None = None,
    ) -> Decision:
        ma_fast = signals.get("ma_fast")
        ma_slow = signals.get("ma_slow")
        prev_fast = signals.get("ma_fast_prev")
        prev_slow = signals.get("ma_slow_prev")

        action: str = "HOLD"
        reasoning = "Insufficient indicator history."
        size = 0.0
        if None not in (ma_fast, ma_slow, prev_fast, prev_slow):
            crossed_up = prev_fast <= prev_slow and ma_fast > ma_slow  # type: ignore[operator]
            crossed_down = prev_fast >= prev_slow and ma_fast < ma_slow  # type: ignore[operator]
            holding = symbol in {p.get("symbol") for p in portfolio.get("positions", []) or []}
            if crossed_up and not holding:
                action = "BUY"
                size = self.size_pct
                reasoning = f"MA up-cross: fast {ma_fast:.2f} > slow {ma_slow:.2f}."
            elif crossed_down and holding:
                action = "SELL"
                reasoning = f"MA down-cross: fast {ma_fast:.2f} < slow {ma_slow:.2f}."

        return Decision(
            timestamp=utcnow(),
            symbol=symbol,
            action=action,  # type: ignore[arg-type]
            size_pct=size,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            reasoning=reasoning,
            signals={k: float(v) for k, v in signals.items() if v is not None},
            model=self.model_name,
        )


# ---------- comparator ----------

@dataclass
class ComparisonRow:
    name: str
    metrics: BacktestMetrics
    result: BacktestResult


def compare_strategies(
    *,
    symbol: Symbol,
    bars: list[Bar],
    risk_config: RiskConfig,
    strategists: dict[str, Strategist],
    starting_cash_usd: float = 1000.0,
    fee_bps: float = 26.0,
    ma_fast: int = 20,
    ma_slow: int = 50,
    rsi_period: int = 14,
    bars_per_year: int = 252,
) -> list[ComparisonRow]:
    """Run each strategist on the same bars and return rows for printing."""
    rows: list[ComparisonRow] = []
    for name, strat in strategists.items():
        result = run_backtest(
            symbol=symbol,
            bars=bars,
            risk_config=risk_config,
            strategist=strat,
            starting_cash_usd=starting_cash_usd,
            fee_bps=fee_bps,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            rsi_period=rsi_period,
            bars_per_year=bars_per_year,
        )
        assert result.metrics is not None  # run_backtest always populates this
        rows.append(ComparisonRow(name=name, metrics=result.metrics, result=result))
    return rows


def format_comparison(rows: list[ComparisonRow]) -> str:
    """Pretty-print a comparison table for the README / demo video."""
    header = f"{'strategy':<24} {'return':>10} {'maxDD':>10} {'sharpe':>10} {'trades':>8} {'winrate':>10}"
    lines = [header, "-" * len(header)]
    for row in rows:
        m = row.metrics
        lines.append(
            f"{row.name:<24} "
            f"{m.total_return * 100:>9.2f}% "
            f"{m.max_drawdown * 100:>9.2f}% "
            f"{m.sharpe_ratio:>10.2f} "
            f"{m.num_trades:>8d} "
            f"{m.win_rate * 100:>9.1f}%"
        )
    return "\n".join(lines)
