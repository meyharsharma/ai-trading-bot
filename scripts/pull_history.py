"""
Cache historical OHLCV for the agent's trading universe to local parquet.

The Brain worktree's backtest runner consumes these files. We pull them here
because Kraken access lives in this layer — the brain shouldn't be blocked
on data plumbing.

Usage
-----
    python scripts/pull_history.py                       # 90d, 1h, BTC/ETH
    python scripts/pull_history.py --days 30 --interval 5m
    python scripts/pull_history.py --symbols BTC/USD --out data/cache

Pagination
----------
Kraken's OHLC endpoint returns at most ~720 bars per call. We walk forward
from ``now - days`` using the timestamp of the last returned bar as the next
``since`` cursor, deduping on bar timestamp. This keeps the script honest
when the MCP server passes ``since`` through to the underlying REST call.

Output
------
``{out_dir}/{SYMBOL}_{interval}_{days}d.parquet`` with columns:
``timestamp, open, high, low, close, volume``.

Requires ``pyarrow`` (or ``fastparquet``) for parquet writes — install with
``pip install pyarrow`` if not already present.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import pandas as pd

from agent.kraken_mcp import KrakenMCPClient, KrakenTools
from agent.kraken_mcp.tools import OHLCVBar


DEFAULT_SYMBOLS: tuple[str, ...] = ("BTC/USD", "ETH/USD")

# Kraken OHLC supported intervals → seconds.
INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14_400,
    "1d": 86_400,
}

# Most callers — including the Kraken REST shape — cap a single response at
# this many bars. We re-paginate via ``since`` until we cover the window.
PER_CALL_LIMIT = 720

# A defensive cap so a misbehaving server can never spin us forever.
MAX_PAGES = 60


# A "fetch one page" function so the script's pagination loop is testable
# without a live Kraken connection. Signature mirrors KrakenTools.get_ohlcv.
FetchFn = Callable[..., Awaitable[list[OHLCVBar]]]


async def paginate_ohlcv(
    fetch: FetchFn,
    symbol: str,
    interval: str,
    days: int,
    *,
    per_call_limit: int = PER_CALL_LIMIT,
    max_pages: int = MAX_PAGES,
    now: datetime | None = None,
) -> list[OHLCVBar]:
    """Walk the ``since`` cursor forward until we cover ``days`` of history.

    The result is sorted by timestamp ascending and deduped — the same bar
    appearing in two pages is collapsed to one entry.
    """
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"unsupported interval {interval!r}")
    step = INTERVAL_SECONDS[interval]
    now = now or datetime.now(timezone.utc)
    target_start = now - timedelta(days=days)
    since: int = int(target_start.timestamp())

    seen: dict[int, OHLCVBar] = {}
    last_cursor = -1
    for _ in range(max_pages):
        bars = await fetch(symbol, interval=interval, limit=per_call_limit, since=since)
        if not bars:
            break

        added = 0
        for b in bars:
            key = int(b.timestamp.timestamp())
            if key not in seen:
                seen[key] = b
                added += 1

        latest_key = max(int(b.timestamp.timestamp()) for b in bars)
        # No forward progress → stop, otherwise we'd loop forever.
        if latest_key <= last_cursor or added == 0:
            break
        last_cursor = latest_key
        since = latest_key + step

        # Caught up to roughly "now" → done.
        if latest_key >= int(now.timestamp()) - step:
            break

    return [seen[k] for k in sorted(seen)]


def bars_to_dataframe(bars: Iterable[OHLCVBar]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": b.timestamp,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def output_path(out_dir: Path, symbol: str, interval: str, days: int) -> Path:
    safe = symbol.replace("/", "-")
    return out_dir / f"{safe}_{interval}_{days}d.parquet"


async def pull_all(
    fetch: FetchFn,
    symbols: Iterable[str],
    interval: str,
    days: int,
    out_dir: Path,
    *,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for symbol in symbols:
        log(f"pulling {symbol} {interval} for {days}d ...")
        bars = await paginate_ohlcv(fetch, symbol, interval, days)
        df = bars_to_dataframe(bars)
        path = output_path(out_dir, symbol, interval, days)
        df.to_parquet(path, index=False)
        log(f"  → {len(df)} bars → {path}")
        written[symbol] = path
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pull_history.py", description=__doc__)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--interval", default="1h", choices=sorted(INTERVAL_SECONDS))
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--out", default="data/cache", type=Path)
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    async with KrakenMCPClient() as client:
        tools = KrakenTools(client)
        await pull_all(tools.get_ohlcv, args.symbols, args.interval, args.days, args.out)


def main() -> None:
    args = parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
