"""
Live market data feed via the Kraken CLI MCP server.

Wraps ``KrakenTools`` to produce ``MarketSnapshot`` and
``MultiTimeframeSnapshot`` objects that the Brain layer consumes. This is the
only module the Brain needs to import from the Data layer.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from agent.data.indicators import IndicatorSnapshot, compute_indicators
from agent.kraken_mcp.tools import OHLCVBar, Quote


class _ToolsProto(Protocol):
    async def get_ohlcv(
        self, symbol: str, interval: str = ..., limit: int = ...
    ) -> list[OHLCVBar]: ...
    async def get_ticker(self, symbol: str) -> dict[str, float]: ...


@dataclass
class MarketSnapshot:
    """Bars + indicators + live ticker for a single symbol/timeframe."""
    symbol: str
    interval: str
    bars: list[OHLCVBar]
    indicators: IndicatorSnapshot
    last_price: float
    bid: float = 0.0
    ask: float = 0.0

    @property
    def mid_price(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last_price


@dataclass
class MultiTimeframeSnapshot:
    """Concurrent multi-timeframe view of one symbol.

    The brain prompt is much stronger when it can see short-term action
    (1m) and structural trend (1h) in the same call. ``timeframes`` preserves
    insertion order so the brain can render them top-down.
    """
    symbol: str
    quote: Quote
    timeframes: dict[str, MarketSnapshot] = field(default_factory=dict)

    @property
    def intervals(self) -> list[str]:
        return list(self.timeframes.keys())

    def snapshot(self, interval: str) -> MarketSnapshot:
        return self.timeframes[interval]


class KrakenFeed:
    def __init__(
        self,
        tools: _ToolsProto,
        *,
        ma_fast: int = 20,
        ma_slow: int = 50,
        rsi_period: int = 14,
    ):
        self._tools = tools
        self._ma_fast = ma_fast
        self._ma_slow = ma_slow
        self._rsi_period = rsi_period

    # ---------------- single timeframe ----------------

    async def get_snapshot(
        self,
        symbol: str,
        *,
        interval: str = "5m",
        lookback: int = 200,
    ) -> MarketSnapshot:
        bars = await self._tools.get_ohlcv(symbol, interval=interval, limit=lookback)
        if not bars:
            raise RuntimeError(f"no OHLCV bars returned for {symbol} {interval}")
        ticker = await self._tools.get_ticker(symbol)
        return self._build_snapshot(symbol, interval, bars, ticker)

    # ---------------- quotes ----------------

    async def get_quote(self, symbol: str) -> Quote:
        """Top-of-book quote. Used by the paper exec to fill realistically."""
        ticker = await self._tools.get_ticker(symbol)
        return Quote(
            bid=float(ticker.get("bid", 0.0)),
            ask=float(ticker.get("ask", 0.0)),
            last=float(ticker.get("last", 0.0)),
        )

    async def get_mid_price(self, symbol: str) -> float:
        """Live mid for paper-fill pricing. Falls back to last if no book."""
        ticker = await self._tools.get_ticker(symbol)
        mid = float(ticker.get("mid") or 0.0)
        if mid:
            return mid
        return float(ticker.get("last") or 0.0)

    # ---------------- multi timeframe ----------------

    async def get_multi_snapshot(
        self,
        symbol: str,
        *,
        intervals: Sequence[str] = ("1m", "5m", "1h"),
        lookback: int = 200,
    ) -> MultiTimeframeSnapshot:
        """Pull all requested timeframes + a single quote concurrently.

        One round trip per timeframe + one ticker call, all in flight at
        once. The shared quote is reused across timeframes so each
        ``MarketSnapshot.bid``/``ask`` reflects the same instant.
        """
        ticker_task = asyncio.create_task(self._tools.get_ticker(symbol))
        bar_tasks = [
            asyncio.create_task(
                self._tools.get_ohlcv(symbol, interval=iv, limit=lookback)
            )
            for iv in intervals
        ]
        ticker = await ticker_task
        bar_results = await asyncio.gather(*bar_tasks)

        timeframes: dict[str, MarketSnapshot] = {}
        for interval, bars in zip(intervals, bar_results):
            if not bars:
                raise RuntimeError(
                    f"no OHLCV bars returned for {symbol} {interval}"
                )
            timeframes[interval] = self._build_snapshot(symbol, interval, bars, ticker)

        quote = Quote(
            bid=float(ticker.get("bid", 0.0)),
            ask=float(ticker.get("ask", 0.0)),
            last=float(ticker.get("last", 0.0)),
        )
        return MultiTimeframeSnapshot(symbol=symbol, quote=quote, timeframes=timeframes)

    # ---------------- internals ----------------

    def _build_snapshot(
        self,
        symbol: str,
        interval: str,
        bars: list[OHLCVBar],
        ticker: dict[str, float],
    ) -> MarketSnapshot:
        ind = compute_indicators(
            [b.close for b in bars],
            [b.high for b in bars],
            [b.low for b in bars],
            ma_fast=self._ma_fast,
            ma_slow=self._ma_slow,
            rsi_period=self._rsi_period,
        )
        return MarketSnapshot(
            symbol=symbol,
            interval=interval,
            bars=bars,
            indicators=ind,
            last_price=float(ticker.get("last") or bars[-1].close),
            bid=float(ticker.get("bid", 0.0)),
            ask=float(ticker.get("ask", 0.0)),
        )
