"""
Technical indicators (SMA, RSI, ATR) used by the Brain layer.

Implemented in pure Python over sequences so the module is dependency-free at
import time, fast in unit tests, and trivial to reason about. The `ta` and
`pandas` libraries are intentionally not imported here — the indicator surface
the brain actually consumes is small and pandas-on-import is slow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Snapshot of the indicators we feed to the LLM brain."""
    ma_fast: float
    ma_slow: float
    rsi: float
    atr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ma_fast": self.ma_fast,
            "ma_slow": self.ma_slow,
            "rsi": self.rsi,
            "atr": self.atr,
        }


def sma(values: Sequence[float], period: int) -> float:
    if period <= 0 or len(values) < period:
        return 0.0
    window = values[-period:]
    return sum(window) / period


def rsi(values: Sequence[float], period: int = 14) -> float:
    """Wilder-style RSI on the trailing ``period`` deltas.

    Returns 0.0 if we don't have enough data — the brain treats 0 as
    "indicator unavailable" and de-weights it.
    """
    if period <= 0 or len(values) < period + 1:
        return 0.0
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float:
    """Average True Range over the trailing ``period`` bars."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return 0.0
    trs: list[float] = []
    for i in range(n - period, n):
        h, l, prev_c = highs[i], lows[i], closes[i - 1]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    return sum(trs) / len(trs)


def compute_indicators(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    ma_fast: int = 20,
    ma_slow: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ma_fast=sma(closes, ma_fast),
        ma_slow=sma(closes, ma_slow),
        rsi=rsi(closes, rsi_period),
        atr=atr(highs, lows, closes, atr_period),
    )
