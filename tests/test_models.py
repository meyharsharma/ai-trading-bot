"""Smoke test for the frozen contracts. If this breaks, the worktrees diverge."""
from agent.state import (
    Decision,
    Fill,
    Order,
    PortfolioSnapshot,
    Position,
    RiskedDecision,
    ValidationArtifact,
    canonical_hash,
    utcnow,
)


def _decision() -> Decision:
    return Decision(
        timestamp=utcnow(),
        symbol="BTC/USD",
        action="BUY",
        size_pct=0.10,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        reasoning="MA fast crossed above MA slow with RSI<70.",
        signals={"ma_fast": 65000.0, "ma_slow": 64000.0, "rsi": 58.2},
        model="claude-opus-4-6",
    )


def test_decision_constructs():
    d = _decision()
    assert d.symbol == "BTC/USD"
    assert d.action == "BUY"


def test_risked_decision_passes():
    d = _decision()
    rd = RiskedDecision(
        decision=d,
        passed=True,
        clamped=False,
        reasons=[],
        risk_checks={"max_risk_per_trade": True, "max_open_positions": True},
        final_size_pct=0.10,
        final_stop_loss_pct=0.03,
    )
    assert rd.passed
    assert rd.final_size_pct == 0.10


def test_order_fill_position():
    order = Order(symbol="BTC/USD", side="BUY", quantity=0.001)
    fill = Fill(
        order=order,
        filled_at=utcnow(),
        fill_price=65000.0,
        fee_usd=0.17,
        venue="paper",
    )
    pos = Position(
        symbol="BTC/USD",
        quantity=0.001,
        avg_entry_price=65000.0,
        stop_loss_price=63050.0,
        take_profit_price=68900.0,
        opened_at=utcnow(),
    )
    snap = PortfolioSnapshot(
        timestamp=utcnow(),
        cash_usd=935.0,
        positions=(pos,),
        equity_usd=1000.0,
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
    )
    assert fill.fill_price == 65000.0
    assert snap.positions[0].symbol == "BTC/USD"


def test_validation_artifact():
    d = _decision()
    artifact = ValidationArtifact(
        decision_hash=canonical_hash(d),
        trade_hash=None,
        risk_checks={"ok": True},
        pre_state_hash="0x" + "0" * 64,
        post_state_hash="0x" + "1" * 64,
        reasoning_uri="onchain://reasoning/abc",
        timestamp=utcnow(),
        agent_id="agent-1",
    )
    assert artifact.decision_hash.startswith("0x")
    assert len(artifact.decision_hash) == 66


def test_canonical_hash_is_deterministic():
    d = _decision()
    h1 = canonical_hash(d)
    h2 = canonical_hash(d)
    assert h1 == h2
    assert h1.startswith("0x") and len(h1) == 66


def test_canonical_hash_changes_with_content():
    d1 = _decision()
    d2 = d1.model_copy(update={"size_pct": 0.20})
    assert canonical_hash(d1) != canonical_hash(d2)
