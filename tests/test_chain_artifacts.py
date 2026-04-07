"""Artifact submission tests — dry-run only, no RPC."""
from __future__ import annotations

import pytest

from agent.chain import ArtifactsClient, ChainClient, ChainConfig
from agent.chain._client import hex_to_bytes32
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


def _dry_client() -> ChainClient:
    return ChainClient(ChainConfig(dry_run=True))


def _decision() -> Decision:
    return Decision(
        timestamp=utcnow(),
        symbol="BTC/USD",
        action="BUY",
        size_pct=0.10,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        reasoning="MA20 > MA50, RSI 58.",
        signals={"ma_fast": 65000.0, "ma_slow": 64000.0, "rsi": 58.2},
        model="claude-opus-4-6",
    )


def _artifact() -> ValidationArtifact:
    d = _decision()
    return ValidationArtifact(
        decision_hash=canonical_hash(d),
        trade_hash=canonical_hash({"fake": "fill"}),
        risk_checks={"max_risk_per_trade": True, "max_open_positions": True},
        pre_state_hash="0x" + "a" * 64,
        post_state_hash="0x" + "b" * 64,
        reasoning_uri="data:text/plain;base64,aGVsbG8=",
        timestamp=utcnow(),
        agent_id="42",
    )


def test_submit_dry_run_returns_receipt():
    client = ArtifactsClient(_dry_client(), agent_id=42)
    sub = client.submit(_artifact())

    assert sub.via == "dry_run"
    assert sub.tx_hash.startswith("0x")
    assert sub.artifact_hash.startswith("0x")
    assert len(sub.artifact_hash) == 66


def test_submit_dry_run_is_deterministic():
    client = ArtifactsClient(_dry_client(), agent_id=42)
    a = _artifact()
    sub1 = client.submit(a)
    sub2 = client.submit(a)
    assert sub1.tx_hash == sub2.tx_hash
    assert sub1.artifact_hash == sub2.artifact_hash


def test_submit_dry_run_changes_with_artifact():
    client = ArtifactsClient(_dry_client(), agent_id=42)
    sub_a = client.submit(_artifact())
    artifact_b = _artifact().model_copy(update={"reasoning_uri": "data:other"})
    sub_b = client.submit(artifact_b)
    assert sub_a.artifact_hash != sub_b.artifact_hash


def test_artifact_hash_matches_canonical_hash():
    """The on-chain `requestHash` MUST be re-derivable by any auditor."""
    client = ArtifactsClient(_dry_client(), agent_id=42)
    a = _artifact()
    sub = client.submit(a)
    assert sub.artifact_hash == canonical_hash(a)


def test_rejects_zero_agent_id():
    with pytest.raises(ValueError):
        ArtifactsClient(_dry_client(), agent_id=0)


def test_rejects_malformed_hash_inputs():
    """Bad hash shapes must fail loudly, not inside the ABI encoder."""
    client = ArtifactsClient(_dry_client(), agent_id=42)
    bad = _artifact().model_copy(update={"pre_state_hash": "not-hex"})
    with pytest.raises(ValueError):
        client.submit(bad)


def test_hex_to_bytes32_roundtrip():
    h = "0x" + "ab" * 32
    raw = hex_to_bytes32(h)
    assert len(raw) == 32
    assert raw.hex() == "ab" * 32


def test_hex_to_bytes32_rejects_short():
    with pytest.raises(ValueError):
        hex_to_bytes32("0x" + "ab" * 16)
