"""
Backtest runner.

Replays a sequence of OHLCV bars through the full Brain → Risk Gate → Exec
pipeline using a stub `ExecutionAdapter` that lives in this file. The real
exec layer is owned by another worktree; we keep a self-contained stub here
so the backtest never depends on it.

Pipeline per bar:
    1. Append the new bar to history.
    2. Check the open position (if any) for stop-loss / take-profit hits at
       the bar's high/low. Close it if so.
    3. Compute indicators (MA fast, MA slow, RSI) on the closed history.
    4. Build a `PortfolioSnapshot` from the stub adapter's current state.
    5. Hand context to a `Strategist` callable. The strategist returns a
       `Decision`. The default strategist is a deterministic MA-cross + RSI
       rule so the backtest is reproducible without an LLM.
    6. Pass the `Decision` through the `RiskGate`.
    7. If the gate passes, route the resulting order to the stub adapter
       which fills at the bar's close (minus fee_bps).
    8. Mark equity to market on the bar's close and append to the curve.

Output is a `BacktestResult` containing equity_curve, closed-trade PnLs,
the list of `Decision`s, and the computed `BacktestMetrics`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

from agent.risk.gate import RiskConfig, RiskGate
from agent.state import (
    Decision,
    Fill,
    Order,
    PortfolioSnapshot,
    Position,
    RiskedDecision,
    Symbol,
    utcnow,
)

from backtest.metrics import BacktestMetrics, compute_metrics


# ---------- bar type ----------

@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# ---------- stub execution adapter ----------

@dataclass
class _OpenPosition:
    symbol: Symbol
    quantity: float
    avg_entry_price: float
    stop_loss_price: float
    take_profit_price: float
    opened_at: datetime


class StubExecutionAdapter:
    """
    Minimal in-memory paper executor used by the backtest.

    Spot-only, one position per symbol at a time. Fills happen instantly at
    the price the runner passes in (typically the bar close). A flat fee in
    basis points is deducted from cash on every fill.
    """

    def __init__(self, *, starting_cash_usd: float = 1000.0, fee_bps: float = 26.0) -> None:
        self.cash_usd = starting_cash_usd
        self.fee_bps = fee_bps
        self.positions: dict[Symbol, _OpenPosition] = {}
        self.fills: list[Fill] = []
        self.realized_pnl_usd: float = 0.0
        self.closed_trade_pnls: list[float] = []

    # ----- queries -----

    def equity(self, marks: dict[Symbol, float]) -> float:
        eq = self.cash_usd
        for sym, pos in self.positions.items():
            eq += pos.quantity * marks.get(sym, pos.avg_entry_price)
        return eq

    def snapshot(self, *, marks: dict[Symbol, float], ts: datetime) -> PortfolioSnapshot:
        equity = self.equity(marks)
        unreal = 0.0
        positions: list[Position] = []
        for sym, pos in self.positions.items():
            mark = marks.get(sym, pos.avg_entry_price)
            unreal += (mark - pos.avg_entry_price) * pos.quantity
            positions.append(
                Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    avg_entry_price=pos.avg_entry_price,
                    stop_loss_price=pos.stop_loss_price,
                    take_profit_price=pos.take_profit_price,
                    opened_at=pos.opened_at,
                )
            )
        return PortfolioSnapshot(
            timestamp=ts,
            cash_usd=self.cash_usd,
            positions=tuple(positions),
            equity_usd=equity,
            realized_pnl_usd=self.realized_pnl_usd,
            unrealized_pnl_usd=unreal,
        )

    # ----- mutations -----

    def open_long(
        self,
        *,
        symbol: Symbol,
        notional_usd: float,
        price: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        ts: datetime,
    ) -> Fill | None:
        if symbol in self.positions or notional_usd <= 0 or price <= 0:
            return None
        notional_usd = min(notional_usd, self.cash_usd)
        if notional_usd <= 0:
            return None
        qty = notional_usd / price
        fee = notional_usd * self.fee_bps / 10_000.0
        if fee + notional_usd > self.cash_usd:
            # Shrink to fit available cash including fee.
            notional_usd = self.cash_usd / (1.0 + self.fee_bps / 10_000.0)
            qty = notional_usd / price
            fee = notional_usd * self.fee_bps / 10_000.0
        self.cash_usd -= notional_usd + fee
        self.positions[symbol] = _OpenPosition(
            symbol=symbol,
            quantity=qty,
            avg_entry_price=price,
            stop_loss_price=price * (1.0 - stop_loss_pct),
            take_profit_price=price * (1.0 + take_profit_pct),
            opened_at=ts,
        )
        order = Order(symbol=symbol, side="BUY", quantity=qty)
        fill = Fill(
            order=order,
            filled_at=ts,
            fill_price=price,
            fee_usd=fee,
            venue="paper",
        )
        self.fills.append(fill)
        return fill

    def close(self, *, symbol: Symbol, price: float, ts: datetime) -> Fill | None:
        pos = self.positions.pop(symbol, None)
        if pos is None or price <= 0:
            return None
        notional = pos.quantity * price
        fee = notional * self.fee_bps / 10_000.0
        self.cash_usd += notional - fee
        pnl = (price - pos.avg_entry_price) * pos.quantity - fee
        self.realized_pnl_usd += pnl
        self.closed_trade_pnls.append(pnl)
        order = Order(symbol=symbol, side="SELL", quantity=pos.quantity)
        fill = Fill(
            order=order,
            filled_at=ts,
            fill_price=price,
            fee_usd=fee,
            venue="paper",
        )
        self.fills.append(fill)
        return fill


# ---------- strategist protocol ----------

class Strategist(Protocol):
    def decide(
        self,
        *,
        symbol: Symbol,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
        recent_decisions: list[dict[str, Any]] | None = None,
    ) -> Decision: ...


# ---------- default deterministic strategist ----------

class MACrossStrategist:
    """
    Deterministic baseline used in tests and as the default backtest brain.

    Rules:
        - BUY when MA(fast) crosses above MA(slow) and RSI < rsi_overbought.
        - SELL (close) when MA(fast) crosses below MA(slow) or RSI > rsi_overbought.
        - HOLD otherwise.

    The point isn't to be a great strategy — it's to give the backtest harness
    something reproducible to run end-to-end without an LLM.
    """

    def __init__(
        self,
        *,
        size_pct: float = 0.20,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        rsi_overbought: float = 70.0,
        model_name: str = "ma-cross-baseline",
    ) -> None:
        self.size_pct = size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.rsi_overbought = rsi_overbought
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
        rsi = signals.get("rsi", 50.0)

        action: str = "HOLD"
        reasoning = "Insufficient indicator history."
        if None not in (ma_fast, ma_slow, prev_fast, prev_slow):
            crossed_up = prev_fast <= prev_slow and ma_fast > ma_slow  # type: ignore[operator]
            crossed_down = prev_fast >= prev_slow and ma_fast < ma_slow  # type: ignore[operator]
            holding = symbol in {p.get("symbol") for p in portfolio.get("positions", []) or []}
            if crossed_up and rsi < self.rsi_overbought and not holding:
                action = "BUY"
                reasoning = (
                    f"MA fast {ma_fast:.2f} crossed above slow {ma_slow:.2f}, "
                    f"RSI {rsi:.1f} < {self.rsi_overbought}."
                )
            elif (crossed_down or rsi > self.rsi_overbought) and holding:
                action = "SELL"
                reasoning = (
                    f"Exit signal: ma_fast={ma_fast:.2f} ma_slow={ma_slow:.2f} rsi={rsi:.1f}."
                )

        return Decision(
            timestamp=utcnow(),
            symbol=symbol,
            action=action,  # type: ignore[arg-type]
            size_pct=self.size_pct if action == "BUY" else 0.0,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            reasoning=reasoning,
            signals={k: float(v) for k, v in signals.items() if v is not None},
            model=self.model_name,
        )


# ---------- indicators (small, dependency-free) ----------

def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for prev, curr in zip(values[-period - 1 : -1], values[-period:]):
        change = curr - prev
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_signals(
    closes: list[float],
    *,
    ma_fast: int = 20,
    ma_slow: int = 50,
    rsi_period: int = 14,
) -> dict[str, float]:
    """Compute indicator snapshot from a closing-price history."""
    out: dict[str, float] = {}
    fast = _sma(closes, ma_fast)
    slow = _sma(closes, ma_slow)
    fast_prev = _sma(closes[:-1], ma_fast) if len(closes) > 1 else None
    slow_prev = _sma(closes[:-1], ma_slow) if len(closes) > 1 else None
    rsi = _rsi(closes, rsi_period)
    if fast is not None:
        out["ma_fast"] = fast
    if slow is not None:
        out["ma_slow"] = slow
    if fast_prev is not None:
        out["ma_fast_prev"] = fast_prev
    if slow_prev is not None:
        out["ma_slow_prev"] = slow_prev
    if rsi is not None:
        out["rsi"] = rsi
    return out


# ---------- result ----------

@dataclass
class BacktestResult:
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    closed_trade_pnls: list[float] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    risked: list[RiskedDecision] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    metrics: BacktestMetrics | None = None


# ---------- runner ----------

def run_backtest(
    *,
    symbol: Symbol,
    bars: list[Bar],
    risk_config: RiskConfig,
    strategist: Strategist | None = None,
    starting_cash_usd: float = 1000.0,
    fee_bps: float = 26.0,
    ma_fast: int = 20,
    ma_slow: int = 50,
    rsi_period: int = 14,
    bars_per_year: int = 252,
    warmup_bars: int | None = None,
) -> BacktestResult:
    """Replay `bars` through the brain → risk → stub-exec pipeline."""
    if not bars:
        raise ValueError("backtest requires at least one bar")

    gate = RiskGate(risk_config)
    adapter = StubExecutionAdapter(starting_cash_usd=starting_cash_usd, fee_bps=fee_bps)
    strategist = strategist or MACrossStrategist()
    warmup = warmup_bars if warmup_bars is not None else max(ma_slow, rsi_period + 1)

    closes: list[float] = []
    result = BacktestResult()

    for i, bar in enumerate(bars):
        # 1. Append history.
        closes.append(bar.close)

        # 2. Stop-loss / take-profit checks for any open position.
        pos = adapter.positions.get(symbol)
        if pos is not None:
            if bar.low <= pos.stop_loss_price:
                adapter.close(symbol=symbol, price=pos.stop_loss_price, ts=bar.timestamp)
            elif bar.high >= pos.take_profit_price:
                adapter.close(symbol=symbol, price=pos.take_profit_price, ts=bar.timestamp)

        # 3. Skip until indicators are warm.
        if i < warmup:
            equity = adapter.equity({symbol: bar.close})
            result.equity_curve.append(equity)
            result.timestamps.append(bar.timestamp)
            continue

        # 4. Build context for the strategist.
        signals = compute_signals(
            closes, ma_fast=ma_fast, ma_slow=ma_slow, rsi_period=rsi_period
        )
        snap = adapter.snapshot(marks={symbol: bar.close}, ts=bar.timestamp)
        portfolio_dict = snap.model_dump(mode="json")

        # 5. Strategist → Decision.
        decision = strategist.decide(
            symbol=symbol,
            price=bar.close,
            signals=signals,
            portfolio=portfolio_dict,
        )
        result.decisions.append(decision)

        # 6. Risk gate.
        risked = gate.evaluate(decision, snap)
        result.risked.append(risked)

        # 7. Route to stub adapter if approved.
        if risked.passed and risked.final_size_pct > 0:
            if decision.action == "BUY":
                fill = adapter.open_long(
                    symbol=symbol,
                    notional_usd=risked.final_size_pct * snap.equity_usd,
                    price=bar.close,
                    stop_loss_pct=risked.final_stop_loss_pct,
                    take_profit_pct=decision.take_profit_pct,
                    ts=bar.timestamp,
                )
                if fill is not None:
                    result.fills.append(fill)
            elif decision.action == "SELL":
                fill = adapter.close(symbol=symbol, price=bar.close, ts=bar.timestamp)
                if fill is not None:
                    result.fills.append(fill)

        # 8. Mark to market.
        result.equity_curve.append(adapter.equity({symbol: bar.close}))
        result.timestamps.append(bar.timestamp)

    result.closed_trade_pnls = list(adapter.closed_trade_pnls)
    result.metrics = compute_metrics(
        result.equity_curve,
        result.closed_trade_pnls,
        bars_per_year=bars_per_year,
    )
    return result


# ---------- convenience: synthetic bar generator ----------

def synthetic_bars(
    *,
    n: int,
    start_price: float = 100.0,
    drift: float = 0.001,
    amplitude: float = 0.02,
    period: int = 30,
    start: datetime | None = None,
    bar_minutes: int = 5,
) -> list[Bar]:
    """
    Deterministic sinusoidal-with-drift price series. Useful for tests so the
    backtest harness has a reliable, MA-crossable signal without bundling
    historical CSVs in the repo.
    """
    import math

    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars: list[Bar] = []
    for i in range(n):
        trend = start_price * (1 + drift) ** i
        price = trend * (1 + amplitude * math.sin(2 * math.pi * i / period))
        prev_close = bars[-1].close if bars else price
        high = max(prev_close, price) * 1.001
        low = min(prev_close, price) * 0.999
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=bar_minutes * i),
                open=prev_close,
                high=high,
                low=low,
                close=price,
            )
        )
    return bars
