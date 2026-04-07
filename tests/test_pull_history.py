"""Tests for the pull_history pagination + dataframe helpers.

We test the pure logic — pagination, dedupe, dataframe shape — against a
fake fetch function. The Kraken MCP wiring is exercised by the script's
``__main__`` path which only runs against a real binary.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.kraken_mcp.tools import OHLCVBar


# scripts/ is not a package — load it as a module by path.
_SPEC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pull_history.py"
_spec = importlib.util.spec_from_file_location("pull_history", _SPEC_PATH)
pull_history = importlib.util.module_from_spec(_spec)
sys.modules["pull_history"] = pull_history
_spec.loader.exec_module(pull_history)  # type: ignore[union-attr]


def _bar(epoch: int, close: float = 100.0) -> OHLCVBar:
    return OHLCVBar(
        timestamp=datetime.fromtimestamp(epoch, tz=timezone.utc),
        open=close, high=close, low=close, close=close, volume=1.0,
    )


# ---------------- pagination ----------------

async def test_paginate_walks_since_cursor_forward():
    now = datetime(2026, 4, 7, 0, 0, tzinfo=timezone.utc)
    step = 3600  # 1h
    start = int((now - timedelta(days=2)).timestamp())
    pages = [
        [_bar(start + i * step) for i in range(3)],
        [_bar(start + (3 + i) * step) for i in range(3)],
        [_bar(start + (6 + i) * step) for i in range(3)],
        [],  # exhausted
    ]
    calls: list[dict] = []

    async def fake_fetch(symbol, *, interval, limit, since):
        calls.append({"symbol": symbol, "interval": interval, "limit": limit, "since": since})
        if not pages:
            return []
        return pages.pop(0)

    bars = await pull_history.paginate_ohlcv(
        fake_fetch, "BTC/USD", "1h", days=2, now=now, max_pages=10
    )
    # 9 unique bars across the three non-empty pages.
    assert len(bars) == 9
    # `since` cursor advanced each call.
    sinces = [c["since"] for c in calls]
    assert sinces == sorted(sinces)
    assert len(set(sinces)) == len(sinces)


async def test_paginate_dedupes_overlapping_pages():
    now = datetime(2026, 4, 7, 0, 0, tzinfo=timezone.utc)
    step = 3600
    start = int((now - timedelta(days=1)).timestamp())
    bar0 = _bar(start)
    bar1 = _bar(start + step)
    bar2 = _bar(start + 2 * step)
    bar3 = _bar(start + 3 * step)
    pages = [[bar0, bar1, bar2], [bar2, bar3]]  # bar2 repeats

    async def fake_fetch(symbol, *, interval, limit, since):
        if pages:
            return pages.pop(0)
        return []

    bars = await pull_history.paginate_ohlcv(
        fake_fetch, "BTC/USD", "1h", days=1, now=now, max_pages=10
    )
    epochs = [int(b.timestamp.timestamp()) for b in bars]
    assert epochs == sorted(set(epochs))
    assert len(bars) == 4


async def test_paginate_stops_when_no_progress():
    now = datetime(2026, 4, 7, 0, 0, tzinfo=timezone.utc)
    same = [_bar(int((now - timedelta(hours=1)).timestamp()))]

    call_count = {"n": 0}

    async def fake_fetch(symbol, *, interval, limit, since):
        call_count["n"] += 1
        return list(same)  # always returns the same single bar

    bars = await pull_history.paginate_ohlcv(
        fake_fetch, "BTC/USD", "1h", days=1, now=now, max_pages=10
    )
    assert len(bars) == 1
    assert call_count["n"] <= 2  # second call returns same → stops


async def test_paginate_rejects_unknown_interval():
    async def noop(*a, **kw):
        return []

    with pytest.raises(ValueError, match="unsupported interval"):
        await pull_history.paginate_ohlcv(noop, "BTC/USD", "7m", days=1)


# ---------------- dataframe shape ----------------

def test_bars_to_dataframe_shape_and_dedupe():
    bars = [_bar(1, 10), _bar(2, 11), _bar(2, 11), _bar(3, 12)]
    df = pull_history.bars_to_dataframe(bars)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 3  # duplicate timestamp collapsed
    assert df["close"].tolist() == [10, 11, 12]


def test_bars_to_dataframe_handles_empty():
    df = pull_history.bars_to_dataframe([])
    assert df.empty
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_output_path_format(tmp_path):
    p = pull_history.output_path(tmp_path, "BTC/USD", "1h", 90)
    assert p.parent == tmp_path
    assert p.name == "BTC-USD_1h_90d.parquet"


# ---------------- argparse ----------------

def test_parse_args_defaults():
    args = pull_history.parse_args([])
    assert args.days == 90
    assert args.interval == "1h"
    assert args.symbols == ["BTC/USD", "ETH/USD"]


def test_parse_args_overrides():
    args = pull_history.parse_args(
        ["--days", "30", "--interval", "5m", "--symbols", "BTC/USD"]
    )
    assert args.days == 30
    assert args.interval == "5m"
    assert args.symbols == ["BTC/USD"]
