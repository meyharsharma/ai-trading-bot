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
DEFAULT_TOOL_ALIASES: dict[str, list[str]] = {
    "get_ohlcv":          ["get_ohlcv", "ohlcv", "get_ohlc", "ohlc", "kraken_ohlc"],
    "get_ticker":         ["get_ticker", "ticker", "kraken_ticker"],
    "get_balance":        ["get_balance", "balance", "kraken_balance"],
    "add_order":          ["add_order", "place_order", "create_order", "kraken_add_order"],
    "cancel_order":       ["cancel_order", "kraken_cancel_order"],
    "get_open_orders":    ["get_open_orders", "open_orders", "kraken_open_orders"],
    "get_open_positions": ["get_open_positions", "open_positions", "kraken_open_positions"],
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


def _kraken_pair(symbol: str) -> str:
    """Translate ``BTC/USD`` → ``XBTUSD`` (Kraken's preferred pair form)."""
    base, quote = symbol.split("/")
    if base == "BTC":
        base = "XBT"
    return f"{base}{quote}"


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
        raw = await self.call("get_ticker", pair=_kraken_pair(symbol))
        return _normalize_ticker(raw)

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 200,
        since: int | None = None,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars. ``since`` is a unix-seconds cursor used by
        ``scripts/pull_history.py`` to paginate past Kraken's per-call cap."""
        kwargs: dict[str, Any] = {
            "pair": _kraken_pair(symbol),
            "interval": interval,
            "limit": limit,
        }
        if since is not None:
            kwargs["since"] = int(since)
        raw = await self.call("get_ohlcv", **kwargs)
        return _normalize_ohlcv(raw)

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
        args: dict[str, Any] = {
            "pair": _kraken_pair(symbol),
            "type": side.lower(),
            "ordertype": order_type.lower(),
            "volume": str(quantity),
        }
        if price is not None:
            args["price"] = str(price)
        result = await self.call("add_order", **args)
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
    # Kraken REST style: {"result": {"XXBTZUSD": {...}}, "error": []}
    if "result" in payload and isinstance(payload["result"], dict) and payload["result"]:
        payload = next(iter(payload["result"].values()))
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
