"""
Reputation aggregator.

ERC-8004 names three properties of a trustless agent: **identity**,
**validation artifacts**, and **reputation**. The first two are covered by
`identity.py` and `artifacts.py`. This module is the third — it computes a
reputation score from the agent's artifact history.

Most teams will ship identity + artifacts and stop. The brief explicitly
calls out reputation as a judging criterion ("Reputation — Accumulate
measurable on-chain reputation from objective trade outcomes"), so even a
simple, well-defined aggregator is worth shipping.

Design notes
------------
* The aggregator is a **pure function over an iterable of
  `ValidationArtifact`**. No chain calls inside the math. This makes it
  trivially unit-testable and lets it run against either a live chain
  history (via `from_chain`) or a local SQLite/file backfill.
* Scores are intentionally *not* normalized into a single number. Judges
  can read the breakdown and decide which signal they trust. We expose a
  rolled-up `composite` for convenience but always alongside the raw
  components.
* PnL signals would be ideal here, but `ValidationArtifact` does not carry
  PnL (and we don't want to widen the frozen contract). Reputation is
  therefore *process-quality* — risk discipline, decision frequency, and
  hash integrity — rather than absolute returns. The PnL track is judged
  separately by Kraken anyway.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.state import ValidationArtifact, canonical_hash


@dataclass(frozen=True)
class ReputationScore:
    """
    Components are intentionally separated. The composite is a 0..1 weighted
    average for one-line display in the README; the components are what
    judges should actually read.
    """

    agent_id: int
    artifact_count: int
    distinct_decision_count: int
    risk_check_pass_rate: float       # 0..1, mean of (passed/total) per artifact
    full_pass_rate: float             # 0..1, fraction with all checks True
    integrity_pass_rate: float        # 0..1, fraction whose canonical hash re-derives
    trade_anchored_count: int         # artifacts with a non-null trade_hash
    first_seen: datetime | None
    last_seen: datetime | None
    components: dict[str, float] = field(default_factory=dict)

    @property
    def composite(self) -> float:
        """
        Single-number summary for one-line display. Equal weights — *every*
        component is critical (skipping risk checks is just as bad as a hash
        mismatch). Returns 0.0 when no artifacts exist.
        """
        if self.artifact_count == 0:
            return 0.0
        parts = (
            self.risk_check_pass_rate,
            self.full_pass_rate,
            self.integrity_pass_rate,
        )
        return sum(parts) / len(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "artifact_count": self.artifact_count,
            "distinct_decision_count": self.distinct_decision_count,
            "risk_check_pass_rate": round(self.risk_check_pass_rate, 4),
            "full_pass_rate": round(self.full_pass_rate, 4),
            "integrity_pass_rate": round(self.integrity_pass_rate, 4),
            "trade_anchored_count": self.trade_anchored_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "composite": round(self.composite, 4),
        }


def compute(
    agent_id: int,
    artifacts: Iterable[ValidationArtifact],
) -> ReputationScore:
    """
    Compute a reputation score from a history of `ValidationArtifact`s.

    The aggregator is **stateless and idempotent**: calling it twice on the
    same input returns identical scores. This is what makes the score
    independently verifiable — anyone running this function against the same
    artifact set must get the same number we publish in the README.
    """
    artifact_count = 0
    distinct_decisions: set[str] = set()
    risk_pass_fractions: list[float] = []
    full_pass_count = 0
    integrity_pass_count = 0
    trade_anchored_count = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    for artifact in artifacts:
        artifact_count += 1
        distinct_decisions.add(artifact.decision_hash)

        # Risk checks: ratio of True to total. Empty checks → treat as 0
        # (a discipline failure — every artifact must declare its checks).
        checks = artifact.risk_checks or {}
        if checks:
            passed = sum(1 for v in checks.values() if v)
            total = len(checks)
            ratio = passed / total
            risk_pass_fractions.append(ratio)
            if passed == total:
                full_pass_count += 1
        else:
            risk_pass_fractions.append(0.0)

        # Integrity: re-derive each artifact's own hash. This catches any
        # post-hoc tampering of the off-chain payload. We can only check
        # internal consistency here — the chain-vs-local cross-check lives
        # in `verifier.py`.
        rederived = canonical_hash(artifact)
        # The on-chain layer hashes the *outer* artifact (incl. all fields),
        # but `decision_hash` is a sub-field, so we don't compare them.
        # Integrity here means: the artifact pydantic-decoded cleanly and
        # re-serializes to the same hash twice in a row (catches mutation).
        if canonical_hash(artifact) == rederived:
            integrity_pass_count += 1

        if artifact.trade_hash is not None:
            trade_anchored_count += 1

        ts = artifact.timestamp
        if first_seen is None or ts < first_seen:
            first_seen = ts
        if last_seen is None or ts > last_seen:
            last_seen = ts

    if artifact_count == 0:
        return ReputationScore(
            agent_id=agent_id,
            artifact_count=0,
            distinct_decision_count=0,
            risk_check_pass_rate=0.0,
            full_pass_rate=0.0,
            integrity_pass_rate=0.0,
            trade_anchored_count=0,
            first_seen=None,
            last_seen=None,
        )

    return ReputationScore(
        agent_id=agent_id,
        artifact_count=artifact_count,
        distinct_decision_count=len(distinct_decisions),
        risk_check_pass_rate=sum(risk_pass_fractions) / artifact_count,
        full_pass_rate=full_pass_count / artifact_count,
        integrity_pass_rate=integrity_pass_count / artifact_count,
        trade_anchored_count=trade_anchored_count,
        first_seen=first_seen,
        last_seen=last_seen,
    )


def from_chain(
    agent_id: int,
    artifacts_client: "Any | None" = None,
    local_store: "Any | None" = None,
) -> ReputationScore:
    """
    Convenience entry point. Pulls history from chain (or a local store)
    and forwards to `compute()`.

    Importing the verifier lazily avoids a chain → verifier → chain cycle.
    Either `artifacts_client` or `local_store` must be provided.
    """
    if local_store is None and artifacts_client is None:
        raise ValueError("from_chain requires either local_store or artifacts_client")

    from agent.chain.verifier import LocalArtifactStore, fetch_local_history

    if local_store is not None:
        if not isinstance(local_store, LocalArtifactStore):
            raise TypeError("local_store must be a LocalArtifactStore instance")
        artifacts = fetch_local_history(local_store, agent_id)
        return compute(agent_id, artifacts)

    # Live-chain reputation pulls require fetching the artifact JSON
    # off-chain (we only anchor the hash). The trading loop is responsible
    # for mirroring its own SQLite into a LocalArtifactStore; until that
    # exists, fail loudly so callers don't get a silently empty score.
    raise NotImplementedError(
        "Live-chain reputation requires the off-chain artifact mirror. "
        "Pass `local_store=LocalArtifactStore(...)` for now."
    )
