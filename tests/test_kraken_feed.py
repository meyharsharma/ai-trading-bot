"""Tests for KrakenFeed using a fake KrakenTools double."""
from datetime import datetime, timezone

import pytest

from agent.data.kraken_feed import KrakenFeed, MultiTimeframeSnapshot
from agent.kraken_mcp.tools import OHLCVBar, Quote


class FakeTools:
    def __init__(self, bars_by_interval, ticker):
        # Accepts either a single bar list (legacy) or a dict per interval.
        if isinstance(bars_by_interval, list):
            self._bars_by_interval = {"5m": bars_by_interval}
            self._default_bars = bars_by_interval
        else:
            self._bars_by_interval = dict(bars_by_interval)
            self._default_bars = next(iter(bars_by_interval.values()))
        self._ticker = ticker
        self.ohlcv_calls: list[tuple] = []
        self.ticker_calls: list[str] = []

    async def get_ohlcv(self, symbol, interval="5m", limit=200):
        self.ohlcv_calls.append((symbol, interval, limit))
        return self._bars_by_interval.get(interval, self._default_bars)

    async def get_ticker(self, symbol):
        self.ticker_calls.append(symbol)
        return self._ticker


def _bars(n: int = 60, base: float = 100.0) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            timestamp=datetime.fromtimestamp(1_700_000_000 + i * 300, tz=timezone.utc),
            open=base + i,
            high=base + 1 + i,
            low=base - 1 + i,
            close=base + i,
            volume=1.0,
        )
        for i in range(n)
    ]


# ---------------- single-timeframe ----------------

async def test_get_snapshot_packs_bars_and_indicators():
    tools = FakeTools(
        _bars(60),
        {"bid": 158.0, "ask": 160.0, "last": 159.0, "mid": 159.0},
    )
    feed = KrakenFeed(tools)
    snap = await feed.get_snapshot("BTC/USD", interval="5m", lookback=60)

    assert len(snap.bars) == 60
    assert snap.indicators.ma_fast > snap.indicators.ma_slow  # rising series
    assert snap.last_price == 159.0
    assert snap.bid == 158.0
    assert snap.ask == 160.0
    assert snap.mid_price == 159.0
    assert tools.ohlcv_calls == [("BTC/USD", "5m", 60)]
    assert tools.ticker_calls == ["BTC/USD"]


async def test_get_snapshot_falls_back_to_last_close_when_ticker_blank():
    bars = _bars(60)
    tools = FakeTools(bars, {"bid": 0.0, "ask": 0.0, "last": 0.0, "mid": 0.0})
    feed = KrakenFeed(tools)
    snap = await feed.get_snapshot("BTC/USD")
    assert snap.last_price == bars[-1].close
    assert snap.mid_price == bars[-1].close


async def test_get_snapshot_raises_on_empty_bars():
    tools = FakeTools([], {"bid": 0, "ask": 0, "last": 0, "mid": 0})
    feed = KrakenFeed(tools)
    with pytest.raises(RuntimeError, match="no OHLCV"):
        await feed.get_snapshot("BTC/USD")


async def test_get_mid_price_uses_ticker_mid_then_last():
    tools = FakeTools(_bars(2), {"bid": 0, "ask": 0, "last": 50.0, "mid": 0})
    feed = KrakenFeed(tools)
    assert await feed.get_mid_price("BTC/USD") == 50.0


# ---------------- quotes ----------------

async def test_get_quote_returns_typed_quote():
    tools = FakeTools(_bars(2), {"bid": 99.0, "ask": 101.0, "last": 100.0})
    feed = KrakenFeed(tools)
    q = await feed.get_quote("BTC/USD")
    assert isinstance(q, Quote)
    assert q.bid == 99.0
    assert q.ask == 101.0
    assert q.mid == 100.0
    assert q.spread == 2.0
    assert q.spread_bps == pytest.approx(200.0)


async def test_quote_falls_back_to_last_when_no_book():
    tools = FakeTools(_bars(2), {"bid": 0.0, "ask": 0.0, "last": 50.0})
    feed = KrakenFeed(tools)
    q = await feed.get_quote("BTC/USD")
    assert q.mid == 50.0
    assert q.spread == 0.0


# ---------------- multi-timeframe ----------------

async def test_get_multi_snapshot_returns_all_intervals():
    tools = FakeTools(
        {
            "1m": _bars(60, base=100),
            "5m": _bars(60, base=200),
            "1h": _bars(60, base=300),
        },
        {"bid": 158.0, "ask": 160.0, "last": 159.0},
    )
    feed = KrakenFeed(tools)
    multi = await feed.get_multi_snapshot("BTC/USD", intervals=("1m", "5m", "1h"), lookback=60)

    assert isinstance(multi, MultiTimeframeSnapshot)
    assert multi.symbol == "BTC/USD"
    assert multi.intervals == ["1m", "5m", "1h"]
    assert multi.snapshot("1m").bars[-1].close == 159  # base 100 + 59
    assert multi.snapshot("5m").bars[-1].close == 259
    assert multi.snapshot("1h").bars[-1].close == 359
    # Ticker reused across timeframes — exactly one ticker call.
    assert tools.ticker_calls == ["BTC/USD"]
    # One ohlcv call per interval.
    intervals_called = [c[1] for c in tools.ohlcv_calls]
    assert intervals_called == ["1m", "5m", "1h"]
    assert multi.quote.mid == 159.0


async def test_get_multi_snapshot_propagates_empty_bars_error():
    tools = FakeTools(
        {"1m": _bars(60), "5m": [], "1h": _bars(60)},
        {"bid": 1, "ask": 2, "last": 1.5},
    )
    feed = KrakenFeed(tools)
    with pytest.raises(RuntimeError, match="no OHLCV"):
        await feed.get_multi_snapshot("BTC/USD", intervals=("1m", "5m", "1h"), lookback=60)
