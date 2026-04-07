"""Vault router tests — dry-run only, no RPC."""
from __future__ import annotations

import pytest

from agent.chain import ChainClient, ChainConfig, VaultRouter
from agent.chain.vault import VaultIntent
from agent.state import (
    Decision,
    Fill,
    Order,
    RiskedDecision,
    canonical_hash,
    utcnow,
)


def _dry_client() -> ChainClient:
    return ChainClient(ChainConfig(dry_run=True))


def _risked(passed: bool = True) -> RiskedDecision:
    d = Decision(
        timestamp=utcnow(),
        symbol="BTC/USD",
        action="BUY",
        size_pct=0.10,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        reasoning="ma cross",
        signals={"ma_fast": 1.0, "ma_slow": 0.5, "rsi": 55.0},
        model="claude-opus-4-6",
    )
    return RiskedDecision(
        decision=d,
        passed=passed,
        clamped=False,
        reasons=[] if passed else ["test rejection"],
        risk_checks={"ok": passed},
        final_size_pct=0.10 if passed else 0.0,
        final_stop_loss_pct=0.03,
    )


def _fill() -> Fill:
    return Fill(
        order=Order(symbol="BTC/USD", side="BUY", quantity=0.001),
        filled_at=utcnow(),
        fill_price=65000.0,
        fee_usd=0.17,
        venue="paper",
    )


def test_vault_router_dry_run_mode():
    router = VaultRouter(_dry_client())
    assert router.mode == "dry_run"


def test_route_intent_dry_run():
    router = VaultRouter(_dry_client())
    receipt = router.route_intent(agent_id=42, risked=_risked(), fill=_fill())

    assert receipt.via == "dry_run"
    assert receipt.tx_hash.startswith("0x")
    assert receipt.intent.agent_id == 42
    assert receipt.intent.symbol == "BTC/USD"
    assert receipt.intent.intent_hash.startswith("0x")


def test_route_intent_refuses_rejected_decision():
    """A rejected RiskedDecision must never reach the vault."""
    router = VaultRouter(_dry_client())
    with pytest.raises(ValueError):
        router.route_intent(agent_id=42, risked=_risked(passed=False), fill=_fill())


def test_intent_hash_changes_with_decision():
    risked_a = _risked()
    fill = _fill()
    intent_a = VaultIntent.from_decision_fill(42, risked_a, fill)

    risked_b_decision = risked_a.decision.model_copy(update={"size_pct": 0.20})
    risked_b = risked_a.model_copy(update={"decision": risked_b_decision})
    intent_b = VaultIntent.from_decision_fill(42, risked_b, fill)

    assert intent_a.intent_hash != intent_b.intent_hash
    assert intent_a.decision_hash != intent_b.decision_hash


def test_intent_hash_changes_with_fill():
    """Auditor-defense: swapping the fill must invalidate the intent hash."""
    risked = _risked()
    intent_a = VaultIntent.from_decision_fill(42, risked, _fill())
    other_fill = _fill().model_copy(update={"fill_price": 70000.0})
    intent_b = VaultIntent.from_decision_fill(42, risked, other_fill)
    assert intent_a.intent_hash != intent_b.intent_hash


def test_intent_payload_is_canonical_json():
    intent = VaultIntent.from_decision_fill(42, _risked(), _fill())
    payload = intent.to_payload_bytes()
    # Same input → same bytes → same on-chain calldata.
    assert payload == intent.to_payload_bytes()
    assert b"BTC/USD" in payload


def test_decision_hash_matches_canonical():
    risked = _risked()
    intent = VaultIntent.from_decision_fill(42, risked, _fill())
    assert intent.decision_hash == canonical_hash(risked.decision)
