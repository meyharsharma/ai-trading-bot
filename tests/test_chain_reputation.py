"""Reputation aggregator tests — pure functions, no chain access."""
from __future__ import annotations

from datetime import timedelta

from agent.chain.reputation import compute
from agent.state import ValidationArtifact, canonical_hash, utcnow


def _artifact(
    decision_hash: str | None = None,
    risk_checks: dict | None = None,
    trade_hash: str | None = "0x" + "f" * 64,
    bump_seconds: int = 0,
) -> ValidationArtifact:
    base_ts = utcnow()
    return ValidationArtifact(
        decision_hash=decision_hash or ("0x" + "1" * 64),
        trade_hash=trade_hash,
        risk_checks=risk_checks if risk_checks is not None else {"a": True, "b": True},
        pre_state_hash="0x" + "2" * 64,
        post_state_hash="0x" + "3" * 64,
        reasoning_uri="data:,test",
        timestamp=base_ts + timedelta(seconds=bump_seconds),
        agent_id="42",
    )


def test_empty_history_yields_zero_score():
    score = compute(agent_id=42, artifacts=[])
    assert score.artifact_count == 0
    assert score.composite == 0.0
    assert score.first_seen is None and score.last_seen is None


def test_all_passing_artifacts_score_one():
    artifacts = [_artifact(decision_hash=f"0x{i:064x}") for i in range(5)]
    score = compute(agent_id=42, artifacts=artifacts)

    assert score.artifact_count == 5
    assert score.distinct_decision_count == 5
    assert score.risk_check_pass_rate == 1.0
    assert score.full_pass_rate == 1.0
    assert score.integrity_pass_rate == 1.0
    assert score.trade_anchored_count == 5
    assert score.composite == 1.0


def test_partial_risk_failures_lower_pass_rate():
    artifacts = [
        _artifact(decision_hash="0x" + "1" * 64, risk_checks={"a": True, "b": True}),
        _artifact(decision_hash="0x" + "2" * 64, risk_checks={"a": True, "b": False}),
        _artifact(decision_hash="0x" + "3" * 64, risk_checks={"a": False, "b": False}),
    ]
    score = compute(agent_id=42, artifacts=artifacts)

    # Mean of (1.0, 0.5, 0.0) = 0.5
    assert score.risk_check_pass_rate == 0.5
    # Only the first artifact had ALL checks pass.
    assert score.full_pass_rate == pytest_approx(1 / 3)


def test_empty_risk_checks_count_as_failure():
    """Skipping risk-check declaration is itself a discipline failure."""
    artifacts = [_artifact(risk_checks={})]
    score = compute(agent_id=42, artifacts=artifacts)
    assert score.risk_check_pass_rate == 0.0
    assert score.full_pass_rate == 0.0


def test_distinct_decision_count_dedupes():
    artifacts = [
        _artifact(decision_hash="0x" + "a" * 64),
        _artifact(decision_hash="0x" + "a" * 64),  # duplicate
        _artifact(decision_hash="0x" + "b" * 64),
    ]
    score = compute(agent_id=42, artifacts=artifacts)
    assert score.artifact_count == 3
    assert score.distinct_decision_count == 2


def test_trade_anchored_count_excludes_holds():
    artifacts = [
        _artifact(decision_hash="0x" + "1" * 64, trade_hash=None),       # HOLD
        _artifact(decision_hash="0x" + "2" * 64, trade_hash="0x" + "f" * 64),
    ]
    score = compute(agent_id=42, artifacts=artifacts)
    assert score.trade_anchored_count == 1


def test_first_and_last_seen_span_history():
    a = _artifact(decision_hash="0x" + "1" * 64, bump_seconds=0)
    b = _artifact(decision_hash="0x" + "2" * 64, bump_seconds=60)
    c = _artifact(decision_hash="0x" + "3" * 64, bump_seconds=30)
    score = compute(agent_id=42, artifacts=[b, c, a])  # order shouldn't matter
    assert score.first_seen == a.timestamp
    assert score.last_seen == b.timestamp


def test_to_dict_serializable():
    artifacts = [_artifact()]
    score = compute(agent_id=42, artifacts=artifacts)
    d = score.to_dict()
    # Round-tripped through dict, all values JSON-friendly.
    import json

    encoded = json.dumps(d)
    parsed = json.loads(encoded)
    assert parsed["agent_id"] == 42
    assert parsed["artifact_count"] == 1
    assert "composite" in parsed


def test_compute_is_idempotent():
    """Same artifact set in → same score out. Required for public auditability."""
    artifacts = [_artifact(decision_hash=f"0x{i:064x}") for i in range(3)]
    score_a = compute(agent_id=42, artifacts=artifacts)
    score_b = compute(agent_id=42, artifacts=artifacts)
    assert score_a == score_b


# Tiny helper to avoid the pytest.approx import dance.
def pytest_approx(value, rel=1e-9):
    import pytest

    return pytest.approx(value, rel=rel)
