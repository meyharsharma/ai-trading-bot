"""SQLite store CRUD + atomicity contract."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.state import (
    Decision,
    Fill,
    Order,
    PortfolioSnapshot,
    Position,
    RiskedDecision,
    Store,
    ValidationArtifact,
    canonical_hash,
    snapshot_state_hash,
    utcnow,
)


def _ts() -> datetime:
    return datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)


def _decision(action="BUY", size=0.10) -> Decision:
    return Decision(
        timestamp=_ts(),
        symbol="BTC/USD",
        action=action,
        size_pct=size,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        reasoning="MA crossover with RSI<70",
        signals={"ma_fast": 65000.0, "ma_slow": 64000.0, "rsi": 58.2},
        model="claude-opus-4-6",
    )


def _risked(d: Decision, *, passed=True, size=None) -> RiskedDecision:
    return RiskedDecision(
        decision=d,
        passed=passed,
        clamped=False,
        reasons=[],
        risk_checks={"max_risk_per_trade": True, "max_open_positions": True},
        final_size_pct=size if size is not None else d.size_pct,
        final_stop_loss_pct=d.stop_loss_pct,
    )


def _fill() -> Fill:
    return Fill(
        order=Order(symbol="BTC/USD", side="BUY", quantity=0.001),
        filled_at=_ts(),
        fill_price=65010.0,
        fee_usd=0.17,
        venue="paper",
        venue_order_id="paper-abc123",
    )


def _portfolio(equity=1000.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=_ts(),
        cash_usd=equity,
        positions=(),
        equity_usd=equity,
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
    )


def _artifact(d: Decision, f: Fill, pre: PortfolioSnapshot, post: PortfolioSnapshot) -> ValidationArtifact:
    return ValidationArtifact(
        decision_hash=canonical_hash(d),
        trade_hash=canonical_hash(f),
        risk_checks={"ok": True},
        pre_state_hash=snapshot_state_hash(pre),
        post_state_hash=snapshot_state_hash(post),
        reasoning_uri="data:text/plain;base64,YWJj",
        timestamp=_ts(),
        agent_id="1",
    )


def test_store_creates_schema(tmp_path: Path):
    db = tmp_path / "s.sqlite"
    with Store(db) as s:
        assert s.count("decisions") == 0
        assert s.count("fills") == 0
        assert s.count("artifacts") == 0
        assert s.count("cycles") == 0
    assert db.exists()


def test_full_happy_path(tmp_path: Path):
    with Store(tmp_path / "s.sqlite") as s:
        cycle_id = s.start_cycle(meta={"mode": "paper"})
        d = _decision()
        rd = _risked(d)
        drow = s.record_decision(cycle_id, rd)
        f = _fill()
        frow = s.record_fill(drow.id, f)
        pre = _portfolio(1000.0)
        post = _portfolio(999.83)
        a = _artifact(d, f, pre, post)
        arow = s.record_artifact(
            decision_id=drow.id,
            fill_id=frow.id,
            artifact=a,
            artifact_hash=canonical_hash(a),
            tx_hash="0xabc",
            via="dry_run",
            block_number=None,
            status="ok",
        )
        s.finish_cycle(cycle_id)

        assert s.count("decisions") == 1
        assert s.count("fills") == 1
        assert s.count("artifacts") == 1
        assert arow.status == "ok"
        assert drow.decision_hash == canonical_hash(d)


def test_hold_decision_persists_without_fill(tmp_path: Path):
    with Store(tmp_path / "s.sqlite") as s:
        cycle_id = s.start_cycle()
        d = _decision(action="HOLD", size=0.0)
        rd = _risked(d, size=0.0)
        s.record_decision(cycle_id, rd)
        s.finish_cycle(cycle_id)
        assert s.count("decisions") == 1
        assert s.count("fills") == 0
        assert s.count("artifacts") == 0


def test_unverified_artifact_round_trip(tmp_path: Path):
    """A failed on-chain submit lands as 'unverified'; retry flips it to 'ok'."""
    with Store(tmp_path / "s.sqlite") as s:
        cycle_id = s.start_cycle()
        d = _decision()
        drow = s.record_decision(cycle_id, _risked(d))
        f = _fill()
        frow = s.record_fill(drow.id, f)
        a = _artifact(d, f, _portfolio(), _portfolio(999.83))
        arow = s.record_artifact(
            decision_id=drow.id,
            fill_id=frow.id,
            artifact=a,
            artifact_hash=canonical_hash(a),
            tx_hash=None,
            via=None,
            block_number=None,
            status="unverified",
            error="rpc timeout",
        )
        unverified = s.list_unverified_artifacts()
        assert len(unverified) == 1
        assert unverified[0]["id"] == arow.id

        s.mark_artifact_verified(
            arow.id, tx_hash="0xdeadbeef", via="dry_run", block_number=42
        )
        assert s.list_unverified_artifacts() == []


def test_invalid_artifact_status_rejected(tmp_path: Path):
    with Store(tmp_path / "s.sqlite") as s:
        cycle_id = s.start_cycle()
        d = _decision()
        drow = s.record_decision(cycle_id, _risked(d))
        f = _fill()
        frow = s.record_fill(drow.id, f)
        a = _artifact(d, f, _portfolio(), _portfolio())
        with pytest.raises(ValueError):
            s.record_artifact(
                decision_id=drow.id,
                fill_id=frow.id,
                artifact=a,
                artifact_hash=canonical_hash(a),
                tx_hash=None,
                via=None,
                block_number=None,
                status="bogus",
            )


def test_cycle_marks_error_on_finish(tmp_path: Path):
    with Store(tmp_path / "s.sqlite") as s:
        cycle_id = s.start_cycle()
        s.finish_cycle(cycle_id, error="boom")
        row = s._conn.execute(
            "SELECT status, error FROM cycles WHERE id=?", (cycle_id,)
        ).fetchone()
        assert row["status"] == "error"
        assert row["error"] == "boom"


def test_decision_hash_round_trips_via_canonical_hash(tmp_path: Path):
    """The hash stored in the row must match canonical_hash(decision) exactly,
    so the trade-audit verifier can re-derive it from off-chain JSON."""
    with Store(tmp_path / "s.sqlite") as s:
        cycle_id = s.start_cycle()
        d = _decision()
        rd = _risked(d)
        drow = s.record_decision(cycle_id, rd)
        assert drow.decision_hash == canonical_hash(d)
