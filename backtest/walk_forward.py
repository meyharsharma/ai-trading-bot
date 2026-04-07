"""
Walk-forward backtest harness.

Single-period backtests are how teams accidentally overfit and get crushed
in live trading. Walk-forward splits history into rolling train/test windows
and reports metrics on each *out-of-sample* test window. The train window
provides indicator warmup only — no decisions or trades fire during it.

For a non-parametric strategy (like our LLM brain or the rule baselines),
"training" means "let the strategy see this much history before we start
scoring it". For a parametric strategy, the train window is where you'd fit
parameters; we leave that hook implicit by accepting any `Strategist`.

Window layout (default 14d train / 7d test):

    | <-- 14d train --><-- 7d test --> |   window 0
                       | <-- 14d train --><-- 7d test --> |   window 1
                                          | ...

The runner re-runs `run_backtest` once per window with `warmup_bars` set to
the train length, then slices the equity curve to the test portion before
computing metrics. Each window starts with the same `starting_cash_usd` —
windows are independent so a single bad window can't poison the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.risk.gate import RiskConfig

from backtest.metrics import BacktestMetrics, compute_metrics
from backtest.runner import Bar, Strategist, run_backtest


@dataclass
class WindowResult:
    index: int
    train_start: datetime
    test_start: datetime
    test_end: datetime
    metrics: BacktestMetrics
    num_test_bars: int


@dataclass
class WalkForwardResult:
    windows: list[WindowResult] = field(default_factory=list)
    aggregate: BacktestMetrics | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_windows": len(self.windows),
            "aggregate": self.aggregate.as_dict() if self.aggregate else None,
            "windows": [
                {
                    "index": w.index,
                    "train_start": w.train_start.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    "metrics": w.metrics.as_dict(),
                    "num_test_bars": w.num_test_bars,
                }
                for w in self.windows
            ],
        }


def run_walk_forward(
    *,
    symbol: str,
    bars: list[Bar],
    risk_config: RiskConfig,
    train_bars: int,
    test_bars: int,
    strategist_factory,
    step_bars: int | None = None,
    starting_cash_usd: float = 1000.0,
    fee_bps: float = 26.0,
    ma_fast: int = 20,
    ma_slow: int = 50,
    rsi_period: int = 14,
    bars_per_year: int = 252,
) -> WalkForwardResult:
    """
    Replay `bars` through rolling train/test windows.

    `strategist_factory` is a zero-arg callable that returns a fresh
    `Strategist`. We re-instantiate per window so any internal state (e.g.
    `BuyAndHoldStrategist._bought`) is reset between windows.

    `step_bars` defaults to `test_bars` (non-overlapping test windows).
    """
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    if not bars:
        raise ValueError("walk-forward requires at least one bar")

    step = step_bars if step_bars is not None else test_bars
    if step <= 0:
        raise ValueError("step_bars must be positive")

    result = WalkForwardResult()
    all_test_curves: list[float] = []
    all_test_pnls: list[float] = []

    i = 0
    window_idx = 0
    while i + train_bars + test_bars <= len(bars):
        slice_bars = bars[i : i + train_bars + test_bars]
        strat = strategist_factory()

        sub = run_backtest(
            symbol=symbol,  # type: ignore[arg-type]
            bars=slice_bars,
            risk_config=risk_config,
            strategist=strat,
            starting_cash_usd=starting_cash_usd,
            fee_bps=fee_bps,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            rsi_period=rsi_period,
            bars_per_year=bars_per_year,
            warmup_bars=train_bars,
        )

        # Slice equity curve to the test portion. Prepend the equity at the
        # train→test boundary so per-window total_return is well-defined
        # (return measured from the bar entering the test window).
        test_curve = sub.equity_curve[train_bars - 1 : train_bars + test_bars]
        # Trades all closed within the test window because no fills happen
        # during warmup. Safe to take the whole list.
        test_pnls = sub.closed_trade_pnls

        window_metrics = compute_metrics(
            test_curve, test_pnls, bars_per_year=bars_per_year
        )

        result.windows.append(
            WindowResult(
                index=window_idx,
                train_start=slice_bars[0].timestamp,
                test_start=slice_bars[train_bars].timestamp,
                test_end=slice_bars[-1].timestamp,
                metrics=window_metrics,
                num_test_bars=len(test_curve),
            )
        )

        # Aggregate by stitching test curves end-to-end normalized to the
        # previous endpoint. This gives an "if you ran it continuously"
        # equity curve we can compute one big set of metrics on.
        if not all_test_curves:
            all_test_curves.extend(test_curve)
        else:
            anchor = all_test_curves[-1]
            base = test_curve[0] if test_curve and test_curve[0] != 0 else 1.0
            scale = anchor / base
            all_test_curves.extend(eq * scale for eq in test_curve[1:])
        all_test_pnls.extend(test_pnls)

        i += step
        window_idx += 1

    if all_test_curves:
        result.aggregate = compute_metrics(
            all_test_curves, all_test_pnls, bars_per_year=bars_per_year
        )

    return result


def format_walk_forward(result: WalkForwardResult) -> str:
    """Pretty-print walk-forward results for a README / demo."""
    if not result.windows:
        return "(no walk-forward windows produced)"
    header = f"{'window':>6} {'test_start':<19} {'return':>10} {'maxDD':>10} {'sharpe':>10} {'trades':>8}"
    lines = [header, "-" * len(header)]
    for w in result.windows:
        m = w.metrics
        lines.append(
            f"{w.index:>6d} {w.test_start.isoformat()[:19]:<19} "
            f"{m.total_return * 100:>9.2f}% "
            f"{m.max_drawdown * 100:>9.2f}% "
            f"{m.sharpe_ratio:>10.2f} "
            f"{m.num_trades:>8d}"
        )
    if result.aggregate:
        a = result.aggregate
        lines.append("-" * len(header))
        lines.append(
            f"{'AGG':>6} {'':<19} "
            f"{a.total_return * 100:>9.2f}% "
            f"{a.max_drawdown * 100:>9.2f}% "
            f"{a.sharpe_ratio:>10.2f} "
            f"{a.num_trades:>8d}"
        )
    return "\n".join(lines)
