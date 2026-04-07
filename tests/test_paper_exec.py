"""Tests for the paper execution adapter — fills, fees, PnL, position math."""
import pytest

from agent.exec.paper import PaperConfig, PaperExecutionAdapter
from agent.kraken_mcp.tools import Quote
from agent.state.models import Order


class FakeQuotes:
    """Mutable async quote source mimicking ``KrakenFeed.get_quote``."""

    def __init__(self, quotes: dict[str, Quote]):
        self.quotes = dict(quotes)

    def set_mid(self, symbol: str, mid: float, *, spread_bps: float = 0.0) -> None:
        half = mid * spread_bps / 20_000.0
        self.quotes[symbol] = Quote(bid=mid - half, ask=mid + half, last=mid)

    async def __call__(self, symbol: str) -> Quote:
        return self.quotes[symbol]


def _flat_quote(price: float) -> Quote:
    """Zero-spread quote — useful for fee math tests where spread would
    confound the assertions."""
    return Quote(bid=price, ask=price, last=price)


@pytest.fixture
def quotes() -> FakeQuotes:
    return FakeQuotes(
        {
            "BTC/USD": _flat_quote(65000.0),
            "ETH/USD": _flat_quote(3000.0),
        }
    )


@pytest.fixture
def adapter(quotes: FakeQuotes) -> PaperExecutionAdapter:
    return PaperExecutionAdapter(
        quotes,
        PaperConfig(starting_capital_usd=1000.0, fee_bps=26.0, slippage_bps=0.0),
    )


# ---------------- core flow ----------------

async def test_buy_records_position_and_debits_cash(adapter):
    fill = await adapter.submit_order(
        Order(symbol="BTC/USD", side="BUY", quantity=0.001)
    )
    assert fill.fill_price == 65000.0
    assert fill.venue == "paper"
    assert fill.venue_order_id and fill.venue_order_id.startswith("paper-")

    snap = await adapter.get_portfolio()
    assert len(snap.positions) == 1
    pos = snap.positions[0]
    assert pos.symbol == "BTC/USD"
    assert pos.quantity == pytest.approx(0.001)
    assert pos.avg_entry_price == pytest.approx(65000.0 * (1 + 26 / 10_000.0))
    expected_cash = 1000.0 - 65.0 * (1 + 26 / 10_000.0)
    assert snap.cash_usd == pytest.approx(expected_cash)


async def test_round_trip_realizes_correct_pnl(adapter, quotes):
    await adapter.submit_order(Order(symbol="BTC/USD", side="BUY", quantity=0.001))
    quotes.quotes["BTC/USD"] = _flat_quote(70000.0)
    sell = await adapter.submit_order(
        Order(symbol="BTC/USD", side="SELL", quantity=0.001)
    )
    assert sell.fill_price == 70000.0

    final = await adapter.get_portfolio()
    assert final.positions == ()

    fee_rate = 26 / 10_000.0
    eff_entry = 65000.0 * (1 + fee_rate)
    eff_exit = 70000.0 * (1 - fee_rate)
    expected_realized = (eff_exit - eff_entry) * 0.001
    assert final.realized_pnl_usd == pytest.approx(expected_realized)
    assert final.cash_usd == pytest.approx(1000.0 + expected_realized)


async def test_unrealized_pnl_uses_live_mid(adapter, quotes):
    await adapter.submit_order(Order(symbol="BTC/USD", side="BUY", quantity=0.001))
    quotes.quotes["BTC/USD"] = _flat_quote(70000.0)
    snap = await adapter.get_portfolio()
    pos = snap.positions[0]
    expected = (70000.0 - pos.avg_entry_price) * pos.quantity
    assert snap.unrealized_pnl_usd == pytest.approx(expected)


async def test_buy_rejected_when_insufficient_cash(adapter):
    with pytest.raises(RuntimeError, match="insufficient cash"):
        await adapter.submit_order(
            Order(symbol="BTC/USD", side="BUY", quantity=1.0)
        )


async def test_sell_without_position_rejected(adapter):
    with pytest.raises(RuntimeError, match="cannot SELL"):
        await adapter.submit_order(
            Order(symbol="ETH/USD", side="SELL", quantity=0.1)
        )


async def test_average_entry_price_compounds_across_buys(adapter, quotes):
    await adapter.submit_order(Order(symbol="BTC/USD", side="BUY", quantity=0.001))
    quotes.quotes["BTC/USD"] = _flat_quote(75000.0)
    await adapter.submit_order(Order(symbol="BTC/USD", side="BUY", quantity=0.001))

    snap = await adapter.get_portfolio()
    pos = snap.positions[0]
    assert pos.quantity == pytest.approx(0.002)

    fee_rate = 26 / 10_000.0
    eff1 = 65000.0 * (1 + fee_rate)
    eff2 = 75000.0 * (1 + fee_rate)
    assert pos.avg_entry_price == pytest.approx((eff1 + eff2) / 2)


async def test_partial_sell_keeps_remaining_position(adapter, quotes):
    await adapter.submit_order(Order(symbol="BTC/USD", side="BUY", quantity=0.002))
    quotes.quotes["BTC/USD"] = _flat_quote(66000.0)
    await adapter.submit_order(Order(symbol="BTC/USD", side="SELL", quantity=0.001))

    snap = await adapter.get_portfolio()
    assert len(snap.positions) == 1
    assert snap.positions[0].quantity == pytest.approx(0.001)


async def test_invalid_quote_raises(quotes):
    quotes.quotes["BTC/USD"] = Quote(bid=0.0, ask=0.0, last=0.0)
    adapter = PaperExecutionAdapter(
        quotes,
        PaperConfig(starting_capital_usd=1000.0, slippage_bps=0.0),
    )
    with pytest.raises(RuntimeError, match="invalid quote"):
        await adapter.submit_order(
            Order(symbol="BTC/USD", side="BUY", quantity=0.001)
        )


# ---------------- realistic fill model ----------------

async def test_buy_crosses_the_spread():
    quotes = FakeQuotes({"BTC/USD": Quote(bid=64990.0, ask=65010.0, last=65000.0)})
    adapter = PaperExecutionAdapter(
        quotes,
        PaperConfig(starting_capital_usd=10_000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    fill = await adapter.submit_order(
        Order(symbol="BTC/USD", side="BUY", quantity=0.001)
    )
    assert fill.fill_price == 65010.0  # paid the ask


async def test_sell_hits_the_bid():
    quotes = FakeQuotes({"BTC/USD": Quote(bid=64990.0, ask=65010.0, last=65000.0)})
    adapter = PaperExecutionAdapter(
        quotes,
        PaperConfig(starting_capital_usd=10_000.0, fee_bps=0.0, slippage_bps=0.0),
    )
    await adapter.submit_order(Order(symbol="BTC/USD", side="BUY", quantity=0.001))
    fill = await adapter.submit_order(
        Order(symbol="BTC/USD", side="SELL", quantity=0.001)
    )
    assert fill.fill_price == 64990.0  # received the bid


async def test_slippage_haircut_on_top_of_spread():
    quotes = FakeQuotes({"BTC/USD": Quote(bid=64990.0, ask=65010.0, last=65000.0)})
    adapter = PaperExecutionAdapter(
        quotes,
        PaperConfig(starting_capital_usd=10_000.0, fee_bps=0.0, slippage_bps=10.0),
    )
    buy = await adapter.submit_order(
        Order(symbol="BTC/USD", side="BUY", quantity=0.001)
    )
    sell = await adapter.submit_order(
        Order(symbol="BTC/USD", side="SELL", quantity=0.0005)
    )
    # 10 bps = 0.001
    assert buy.fill_price == pytest.approx(65010.0 * 1.001)
    assert sell.fill_price == pytest.approx(64990.0 * 0.999)


async def test_fill_falls_back_to_mid_when_no_book():
    """If only a last-trade price is available, fill at last with slippage."""
    quotes = FakeQuotes({"BTC/USD": Quote(bid=0.0, ask=0.0, last=65000.0)})
    adapter = PaperExecutionAdapter(
        quotes,
        PaperConfig(starting_capital_usd=10_000.0, fee_bps=0.0, slippage_bps=10.0),
    )
    buy = await adapter.submit_order(
        Order(symbol="BTC/USD", side="BUY", quantity=0.001)
    )
    assert buy.fill_price == pytest.approx(65000.0 * 1.001)


async def test_get_mark_price_returns_mid(adapter):
    assert await adapter.get_mark_price("BTC/USD") == 65000.0
