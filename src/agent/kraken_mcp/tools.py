"""
Typed Python wrappers around the Kraken CLI MCP server's tools.

The exact tool names exposed by Kraken CLI's MCP server are not fully
standardized across versions (per the Day-1 open question in the build plan).
We resolve them lazily: each *logical* operation has an ordered list of
candidate tool names; on first use we pick the first one that the running
server actually advertises. Override via the ``KRAKEN_MCP_TOOLNAMES`` env
variable (a JSON object mapping logical → server tool name) when needed.

This module is the seam between Kraken-specific JSON shapes and the agent's
internal types. Everything above it speaks plain Python; everything below it
is Kraken JSON.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from agent.kraken_mcp.client import KrakenMCPClient, KrakenMCPError


# Logical operation → ordered list of candidate tool names. The first one that
# the live server advertises (via list_tools) wins. If list_tools returned
# nothing, we use the first candidate as-is.
#
# IMPORTANT: kraken-cli v0.3.0 runs in `guarded` mode by default and does NOT
# expose `kraken_add_order` / `kraken_cancel_order`. The only writable surface
# in guarded mode is the `kraken_paper_*` family. We map our `add_order` /
# `cancel_order` logical names to the paper variants so the loop runs out of
# the box. If you ever flip Kraken CLI into unguarded mode (real money), the
# alias resolver will pick the live names automatically because they appear
# first in each list.
DEFAULT_TOOL_ALIASES: dict[str, list[str]] = {
    "get_ohlcv":          ["kraken_ohlc", "get_ohlcv", "ohlcv", "get_ohlc", "ohlc"],
    "get_ticker":         ["kraken_ticker", "get_ticker", "ticker"],
    "get_balance":        ["kraken_balance", "get_balance", "balance"],
    "add_order":          ["kraken_add_order", "kraken_paper_buy", "add_order", "place_order", "create_order"],
    "cancel_order":       ["kraken_cancel_order", "kraken_paper_cancel", "cancel_order"],
    "get_open_orders":    ["kraken_open_orders", "get_open_orders", "open_orders"],
    "get_open_positions": ["kraken_positions", "get_open_positions", "open_positions", "kraken_open_positions"],
    # Paper-trading-specific surface (always present in guarded mode).
    "paper_init":         ["kraken_paper_init"],
    "paper_buy":          ["kraken_paper_buy"],
    "paper_sell":         ["kraken_paper_sell"],
    "paper_balance":      ["kraken_paper_balance"],
    "paper_status":       ["kraken_paper_status"],
    "paper_cancel":       ["kraken_paper_cancel"],
    "paper_reset":        ["kraken_paper_reset"],
}


@dataclass(frozen=True)
class OHLCVBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Quote:
    """Top-of-book snapshot. ``bid``/``ask`` may be 0 if the venue only
    returned a last-trade price; consumers should fall back to ``mid``."""
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float:
        if self.bid and self.ask:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0:
            return 0.0
        return (self.spread / m) * 10_000.0


_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "1w": 10080, "2w": 21600,
}


def _normalize_interval(interval: str | int) -> str:
    """Convert ``"5m"``/``"1h"``/``"1d"`` (or an int) into the integer-string-of-minutes
    that the kraken_ohlc MCP tool expects (``"5"``, ``"60"``, ``"1440"``)."""
    if isinstance(interval, int):
        return str(interval)
    s = interval.strip().lower()
    if s in _INTERVAL_MINUTES:
        return str(_INTERVAL_MINUTES[s])
    # Already a string of digits? Pass it through.
    if s.isdigit():
        return s
    raise KrakenMCPError(f"unsupported interval {interval!r}")


def _kraken_pair(symbol: str) -> str:
    """Translate ``BTC/USD`` → ``BTCUSD``.

    The Kraken CLI MCP server accepts the human-readable form (`BTCUSD`,
    `ETHUSD`) directly and handles the asset-code mapping internally. We
    deliberately do NOT translate to `XBTUSD` here — that's the legacy REST
    pair name and it's not what the MCP tools expect.
    """
    return symbol.replace("/", "")


def _from_kraken_pair(pair: str) -> str | None:
    """Best-effort reverse translation. Returns ``None`` if unrecognized."""
    if "/" in pair:
        return pair
    p = pair.upper().replace("XXBT", "BTC").replace("XBT", "BTC")
    p = p.replace("ZUSD", "USD")
    if p in ("BTCUSD", "ETHUSD"):
        return f"{p[:3]}/USD"
    return None


class KrakenTools:
    """Logical-layer wrappers around Kraken MCP tools."""

    def __init__(
        self,
        client: KrakenMCPClient,
        aliases: dict[str, list[str]] | None = None,
    ):
        self._client = client
        self._aliases = dict(aliases or DEFAULT_TOOL_ALIASES)
        env_overrides = os.getenv("KRAKEN_MCP_TOOLNAMES")
        if env_overrides:
            for k, v in json.loads(env_overrides).items():
                self._aliases[k] = [v] if isinstance(v, str) else list(v)
        self._resolved: dict[str, str] = {}

    def _resolve(self, logical: str) -> str:
        if logical in self._resolved:
            return self._resolved[logical]
        candidates = self._aliases.get(logical, [logical])
        available = self._client.tool_names
        for cand in candidates:
            if not available or cand in available:
                self._resolved[logical] = cand
                return cand
        raise KrakenMCPError(
            f"no Kraken MCP tool matches logical name {logical!r}; "
            f"tried {candidates}; server advertises {sorted(available)}"
        )

    async def call(self, logical: str, **kwargs: Any) -> Any:
        return await self._client.call_tool(self._resolve(logical), kwargs)

    # ---------------- typed helpers ----------------

    async def get_ticker(self, symbol: str) -> dict[str, float]:
        # kraken_ticker takes `pairs` (array). We always ask for one symbol.
        raw = await self.call("get_ticker", pairs=[_kraken_pair(symbol)])
        return _normalize_ticker(raw)

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 200,
        since: int | None = None,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars.

        ``interval`` is normalized from human-friendly forms (``"5m"``, ``"1h"``,
        ``"1d"``) into the integer-string-of-minutes that the MCP tool expects
        (``"5"``, ``"60"``, ``"1440"``).

        ``limit`` is **not supported** by the kraken_ohlc MCP tool — the server
        returns a fixed-size window from the most recent data (or from
        ``since``). We accept the parameter for backwards compatibility with
        callers but ignore it; downstream slicing happens in Python if needed.

        ``since`` is a unix-seconds cursor used by ``scripts/pull_history.py``
        to paginate past Kraken's per-call cap.
        """
        kwargs: dict[str, Any] = {
            "pair": _kraken_pair(symbol),
            "interval": _normalize_interval(interval),
        }
        if since is not None:
            kwargs["since"] = str(int(since))
        raw = await self.call("get_ohlcv", **kwargs)
        bars = _normalize_ohlcv(raw)
        if limit and len(bars) > limit:
            bars = bars[-limit:]
        return bars

    async def get_balance(self) -> dict[str, float]:
        raw = await self.call("get_balance")
        if isinstance(raw, dict):
            out: dict[str, float] = {}
            for k, v in raw.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    continue
            return out
        return {}

    async def add_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: float | None = None,
    ) -> dict[str, Any]:
        """
        Place an order via the Kraken CLI MCP server.

        In `guarded` mode (the default) `add_order` resolves to
        `kraken_paper_buy` or `kraken_paper_sell` — there is no real-money
        order tool exposed. We pick the buy/sell tool by side and only pass
        the args those schemas accept (`pair`, `volume`, optional `type` /
        `price`). All numeric args are passed as strings; the schemas reject
        bare numbers.
        """
        side_l = side.lower()
        if side_l not in ("buy", "sell"):
            raise KrakenMCPError(f"side must be buy|sell, got {side!r}")

        # Prefer the paper-specific logical name when in guarded mode. This
        # gives the alias resolver a tool that actually exists.
        logical = "paper_buy" if side_l == "buy" else "paper_sell"
        if logical not in self._aliases:
            logical = "add_order"

        args: dict[str, Any] = {
            "pair": _kraken_pair(symbol),
            "volume": str(quantity),
            "type": order_type.lower(),
        }
        if price is not None:
            args["price"] = str(price)
        result = await self.call(logical, **args)
        return result if isinstance(result, dict) else {"raw": result}

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        result = await self.call("cancel_order", txid=order_id)
        return result if isinstance(result, dict) else {"raw": result}

    async def get_open_positions(self) -> list[dict[str, Any]]:
        raw = await self.call("get_open_positions")
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
        if isinstance(raw, dict):
            return [v for v in raw.values() if isinstance(v, dict)]
        return []


# ---------------- normalizers ----------------
#
# Kraken JSON shapes vary across CLI versions and across REST vs streaming
# endpoints. These helpers paper over the differences so the rest of the
# codebase only sees clean Python.

def _normalize_ticker(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise KrakenMCPError(f"unexpected ticker payload: {type(raw).__name__}")
    payload = raw
    # Kraken REST style: {"result": {"XXBTZUSD": {...}}, "error": []}.
    if "result" in payload and isinstance(payload["result"], dict) and payload["result"]:
        payload = next(iter(payload["result"].values()))
    # Kraken CLI MCP style: result envelope already stripped, payload is
    # `{"XXBTZUSD": {a, b, c, ...}}`. Descend into the first dict child.
    elif "b" not in payload and "bid" not in payload:
        for value in payload.values():
            if isinstance(value, dict):
                payload = value
                break
    bid = _first_number(payload, ["b", "bid", "best_bid"])
    ask = _first_number(payload, ["a", "ask", "best_ask"])
    last = _first_number(payload, ["c", "last", "last_price", "close"])
    mid = (bid + ask) / 2 if bid and ask else last
    return {"bid": bid, "ask": ask, "last": last, "mid": mid}


def _first_number(payload: dict[str, Any], keys: Sequence[str]) -> float:
    for k in keys:
        if k not in payload:
            continue
        v = payload[k]
        if isinstance(v, list) and v:
            v = v[0]
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _normalize_ohlcv(raw: Any) -> list[OHLCVBar]:
    rows: list[Any] = []
    if isinstance(raw, dict):
        if "result" in raw and isinstance(raw["result"], dict) and raw["result"]:
            # Kraken REST: {"result": {"XXBTZUSD": [[ts,o,h,l,c,vwap,vol,count],...], "last": ...}}
            for key, val in raw["result"].items():
                if key == "last":
                    continue
                if isinstance(val, list):
                    rows = val
                    break
        elif "candles" in raw and isinstance(raw["candles"], list):
            rows = raw["candles"]
        elif "ohlc" in raw and isinstance(raw["ohlc"], list):
            rows = raw["ohlc"]
        elif "bars" in raw and isinstance(raw["bars"], list):
            rows = raw["bars"]
        else:
            # Kraken CLI MCP shape (result envelope already stripped):
            #   {"XXBTZUSD": [[ts,o,h,l,c,vwap,vol,count], ...], "last": <ts>}
            for key, val in raw.items():
                if key == "last":
                    continue
                if isinstance(val, list):
                    rows = val
                    break
    elif isinstance(raw, list):
        rows = raw
    return [_row_to_bar(r) for r in rows]


def _row_to_bar(row: Any) -> OHLCVBar:
    if isinstance(row, dict):
        ts = row.get("timestamp") or row.get("time") or row.get("t")
        return OHLCVBar(
            timestamp=_parse_ts(ts),
            open=float(row.get("open", row.get("o", 0.0))),
            high=float(row.get("high", row.get("h", 0.0))),
            low=float(row.get("low", row.get("l", 0.0))),
            close=float(row.get("close", row.get("c", 0.0))),
            volume=float(row.get("volume", row.get("v", 0.0))),
        )
    # Kraken REST list shape: [time, open, high, low, close, vwap, volume, count]
    return OHLCVBar(
        timestamp=_parse_ts(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[6]) if len(row) > 6 else float(row[5]),
    )


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        secs = value / 1000.0 if value > 1e12 else float(value)
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except ValueError:
                pass
    return datetime.now(tz=timezone.utc)
