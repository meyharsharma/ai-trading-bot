"""
Live Kraken execution adapter — submits real orders via the Kraken CLI MCP
server. Default OFF.

To enable, set ``KRAKEN_LIVE_TRADING=1`` (or pass ``allow_live=True``) AND
flip ``execution.mode: live`` in ``config/strategy.yaml``. The dual gate is
intentional — we never want a config typo to send real orders.

Swapping the default ``PaperExecutionAdapter`` for this class is the only
change required to go live; the rest of the system speaks ``ExecutionAdapter``.
"""
from __future__ import annotations

import os
from typing import Any

from agent.exec.base import ExecutionAdapter
from agent.kraken_mcp.tools import KrakenTools, _from_kraken_pair
from agent.state.models import Fill, Order, PortfolioSnapshot, Position, utcnow


class KrakenLiveExecutionAdapter(ExecutionAdapter):
    venue = "kraken"

    def __init__(self, tools: KrakenTools, *, allow_live: bool | None = None):
        self._tools = tools
        if allow_live is None:
            allow_live = os.getenv("KRAKEN_LIVE_TRADING") == "1"
        self._allow_live = allow_live

    async def submit_order(self, order: Order) -> Fill:
        if not self._allow_live:
            raise RuntimeError(
                "live Kraken trading is disabled — set KRAKEN_LIVE_TRADING=1 "
                "or pass allow_live=True to enable"
            )
        result = await self._tools.add_order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type.lower(),
            price=order.limit_price,
        )
        fill_price = _extract_fill_price(result)
        if fill_price is None or fill_price <= 0:
            fill_price = order.limit_price or 0.0
        if fill_price <= 0:
            ticker = await self._tools.get_ticker(order.symbol)
            fill_price = float(ticker.get("last") or ticker.get("mid") or 0.0)

        return Fill(
            order=order,
            filled_at=utcnow(),
            fill_price=fill_price,
            fee_usd=_extract_fee(result),
            venue="kraken",
            venue_order_id=_extract_order_id(result),
        )

    async def get_portfolio(self) -> PortfolioSnapshot:
        balances = await self._tools.get_balance()
        cash = float(balances.get("ZUSD", balances.get("USD", 0.0)))

        positions: list[Position] = []
        for raw in await self._tools.get_open_positions():
            pos = _position_from_raw(raw)
            if pos is not None:
                positions.append(pos)

        equity = cash
        unrealized = 0.0
        for pos in positions:
            ticker = await self._tools.get_ticker(pos.symbol)
            mark = float(ticker.get("mid") or ticker.get("last") or pos.avg_entry_price)
            equity += mark * pos.quantity
            unrealized += (mark - pos.avg_entry_price) * pos.quantity

        return PortfolioSnapshot(
            timestamp=utcnow(),
            cash_usd=cash,
            positions=tuple(positions),
            equity_usd=equity,
            realized_pnl_usd=0.0,  # live realized PnL must come from the State store, not Kraken
            unrealized_pnl_usd=unrealized,
        )

    async def get_mark_price(self, symbol: str) -> float:
        t = await self._tools.get_ticker(symbol)
        return float(t.get("mid") or t.get("last") or 0.0)


# ---------------- response parsers ----------------
#
# Kraken's add_order / open_positions JSON shapes vary across CLI versions.
# We extract best-effort and let the caller fall back to ticker mid when fields
# are missing.

def _extract_fill_price(result: Any) -> float | None:
    if not isinstance(result, dict):
        return None
    for key in ("price", "fill_price", "avg_price", "average_price"):
        if key in result:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                continue
    desc = result.get("descr")
    if isinstance(desc, dict) and "price" in desc:
        try:
            return float(desc["price"])
        except (TypeError, ValueError):
            pass
    return None


def _extract_fee(result: Any) -> float:
    if not isinstance(result, dict):
        return 0.0
    for key in ("fee", "fee_usd", "cost"):
        if key in result:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _extract_order_id(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    for key in ("txid", "order_id", "id"):
        v = result.get(key)
        if isinstance(v, list) and v:
            return str(v[0])
        if isinstance(v, str):
            return v
    return None


def _position_from_raw(raw: dict[str, Any]) -> Position | None:
    sym_raw = raw.get("symbol") or raw.get("pair")
    if not isinstance(sym_raw, str):
        return None
    sym = _from_kraken_pair(sym_raw)
    if sym not in ("BTC/USD", "ETH/USD"):
        return None
    try:
        return Position(
            symbol=sym,  # type: ignore[arg-type]
            quantity=float(raw.get("quantity", raw.get("vol", 0.0))),
            avg_entry_price=float(raw.get("avg_entry_price", raw.get("cost_basis", 0.0))),
            stop_loss_price=float(raw.get("stop_loss_price", 0.0)),
            take_profit_price=float(raw.get("take_profit_price", 0.0)),
            opened_at=utcnow(),
        )
    except (TypeError, ValueError):
        return None
