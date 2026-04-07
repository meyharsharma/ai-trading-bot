"""Tests for the live Kraken execution adapter using a fake KrakenTools."""
import pytest

from agent.exec.kraken_live import KrakenLiveExecutionAdapter
from agent.state.models import Order


class FakeTools:
    def __init__(self):
        self.add_order_calls: list[dict] = []
        self.add_order_response: dict = {
            "txid": ["ABC-123"],
            "price": "65000.0",
            "fee": "0.17",
        }
        self.balances: dict = {"ZUSD": 1000.0}
        self.positions: list = []
        self.tickers: dict = {
            "BTC/USD": {"bid": 64999.0, "ask": 65001.0, "last": 65000.0, "mid": 65000.0},
            "ETH/USD": {"bid": 2999.0, "ask": 3001.0, "last": 3000.0, "mid": 3000.0},
        }

    async def add_order(self, **kwargs):
        self.add_order_calls.append(kwargs)
        return self.add_order_response

    async def get_ticker(self, symbol):
        return self.tickers[symbol]

    async def get_balance(self):
        return self.balances

    async def get_open_positions(self):
        return self.positions


async def test_live_refuses_when_disabled():
    adapter = KrakenLiveExecutionAdapter(FakeTools(), allow_live=False)
    with pytest.raises(RuntimeError, match="disabled"):
        await adapter.submit_order(
            Order(symbol="BTC/USD", side="BUY", quantity=0.001)
        )


async def test_live_submits_order_and_parses_fill():
    tools = FakeTools()
    adapter = KrakenLiveExecutionAdapter(tools, allow_live=True)

    fill = await adapter.submit_order(
        Order(symbol="BTC/USD", side="BUY", quantity=0.001)
    )

    assert fill.venue == "kraken"
    assert fill.fill_price == 65000.0
    assert fill.fee_usd == pytest.approx(0.17)
    assert fill.venue_order_id == "ABC-123"

    sent = tools.add_order_calls[0]
    assert sent["symbol"] == "BTC/USD"
    assert sent["side"] == "BUY"
    assert sent["order_type"] == "market"
    assert sent["quantity"] == 0.001


async def test_live_falls_back_to_ticker_mid_when_response_missing_price():
    tools = FakeTools()
    tools.add_order_response = {"txid": ["ZZZ-1"]}  # no price/fee fields
    adapter = KrakenLiveExecutionAdapter(tools, allow_live=True)

    fill = await adapter.submit_order(
        Order(symbol="BTC/USD", side="BUY", quantity=0.001)
    )
    assert fill.fill_price == 65000.0  # picked from ticker mid
    assert fill.fee_usd == 0.0
    assert fill.venue_order_id == "ZZZ-1"


async def test_live_get_portfolio_uses_balance_and_positions():
    tools = FakeTools()
    tools.positions = [
        {
            "symbol": "XBTUSD",
            "quantity": 0.002,
            "avg_entry_price": 64000.0,
        }
    ]
    adapter = KrakenLiveExecutionAdapter(tools, allow_live=True)
    snap = await adapter.get_portfolio()

    assert snap.cash_usd == 1000.0
    assert len(snap.positions) == 1
    pos = snap.positions[0]
    assert pos.symbol == "BTC/USD"
    assert pos.quantity == pytest.approx(0.002)
    expected_unreal = (65000.0 - 64000.0) * 0.002
    assert snap.unrealized_pnl_usd == pytest.approx(expected_unreal)
    assert snap.equity_usd == pytest.approx(1000.0 + 65000.0 * 0.002)


async def test_live_get_mark_price_uses_ticker():
    adapter = KrakenLiveExecutionAdapter(FakeTools(), allow_live=True)
    assert await adapter.get_mark_price("BTC/USD") == 65000.0
