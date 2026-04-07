"""
Backtest performance metrics.

Pure-Python (no pandas / quantstats dep) so tests stay fast and the numbers
are easy to audit. The four metrics that matter for the hackathon judging
rubric are:

    total_return     — final / initial - 1
    max_drawdown     — worst peak-to-trough decline on the equity curve
    win_rate         — closed trades with PnL > 0 / total closed trades
    sharpe_ratio     — mean(returns) / std(returns) * sqrt(annualization)

`bars_per_year` defaults to 252 (daily). For 5m crypto bars use 365*24*12.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    num_trades: int
    final_equity: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "num_trades": self.num_trades,
            "final_equity": self.final_equity,
        }


def total_return(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0
    return equity_curve[-1] / equity_curve[0] - 1.0


def max_drawdown(equity_curve: list[float]) -> float:
    """Returns a non-negative fraction (0.15 == 15% drawdown)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > worst:
                worst = dd
    return worst


def win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def sharpe_ratio(equity_curve: list[float], bars_per_year: int = 252) -> float:
    """
    Annualized Sharpe of per-bar simple returns. Risk-free rate = 0.

    Returns 0.0 for degenerate curves (constant equity, <2 bars). The intent
    is to be defensive in tests, not statistically rigorous.
    """
    if len(equity_curve) < 3:
        return 0.0
    rets: list[float] = []
    for prev, curr in zip(equity_curve[:-1], equity_curve[1:]):
        if prev <= 0:
            return 0.0
        rets.append(curr / prev - 1.0)
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(bars_per_year)


def compute_metrics(
    equity_curve: list[float],
    trade_pnls: list[float],
    *,
    bars_per_year: int = 252,
) -> BacktestMetrics:
    return BacktestMetrics(
        total_return=total_return(equity_curve),
        max_drawdown=max_drawdown(equity_curve),
        win_rate=win_rate(trade_pnls),
        sharpe_ratio=sharpe_ratio(equity_curve, bars_per_year=bars_per_year),
        num_trades=len(trade_pnls),
        final_equity=equity_curve[-1] if equity_curve else 0.0,
    )
