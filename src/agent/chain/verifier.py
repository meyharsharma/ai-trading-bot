"""
Trustless verifier — the demo's most important script made library-callable.

The premise of ERC-8004 is that **anyone** can independently confirm an
agent's history without trusting the agent operator. This module makes that
premise executable: given an `agent_id` and the agent's published off-chain
artifact directory, it walks every artifact, recomputes the canonical hash,
and compares it to what was anchored on-chain.

Three orthogonal checks per artifact:

1. **filename ↔ payload** — does `0x<hash>.json` actually canonical-hash to
   `0x<hash>`? (catches local tampering)
2. **on-chain ↔ local** — does every chain-anchored hash exist locally?
   (catches missing or withheld artifacts)
3. **local ↔ on-chain** — does every local artifact appear on-chain?
   (catches the operator publishing artifacts they never anchored)

The CLI (`scripts/verify_agent.py`) is a thin wrapper around `Verifier`.

Off-chain artifact convention
-----------------------------
Artifacts are stored as `{root}/{agent_id}/0x<canonical_hash>.json`. The
filename **is** the integrity claim. The contents are exactly
`ValidationArtifact.model_dump(mode="json")` serialized with sorted keys
and no whitespace — same canonicalization as `agent.state.canonical_hash`.

Use `LocalArtifactStore.write(artifact)` to produce files in this format
(the trading loop uses this; the verifier never writes).
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from agent.state import ValidationArtifact, canonical_hash

from .artifacts import ArtifactsClient, OnChainArtifactRef


# --------------------------------------------------------------- store

class LocalArtifactStore:
    """
    On-disk mirror of the agent's published artifacts.

    Filename is the canonical hash. The contents are the canonical JSON
    (sorted keys, no whitespace) so re-hashing the file bytes is
    bit-for-bit reproducible across machines.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _dir(self, agent_id: int) -> Path:
        return self.root / str(agent_id)

    def _path(self, agent_id: int, artifact_hash: str) -> Path:
        return self._dir(agent_id) / f"{artifact_hash.lower()}.json"

    def write(self, agent_id: int, artifact: ValidationArtifact) -> Path:
        """Persist an artifact in canonical form. Returns the file path."""
        h = canonical_hash(artifact)
        d = self._dir(agent_id)
        d.mkdir(parents=True, exist_ok=True)
        path = self._path(agent_id, h)
        # Same encoding as canonical_hash so byte-level audit is exact.
        path.write_text(
            json.dumps(
                artifact.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
        return path

    def read_raw(self, agent_id: int, artifact_hash: str) -> dict | None:
        """Returns the raw dict, or None if the file is missing."""
        path = self._path(agent_id, artifact_hash)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def read(self, agent_id: int, artifact_hash: str) -> ValidationArtifact | None:
        raw = self.read_raw(agent_id, artifact_hash)
        if raw is None:
            return None
        return ValidationArtifact.model_validate(raw)

    def list_hashes(self, agent_id: int) -> list[str]:
        """All artifact hashes the agent has published locally, lowercased."""
        d = self._dir(agent_id)
        if not d.exists():
            return []
        return sorted(p.stem.lower() for p in d.glob("0x*.json"))


def fetch_local_history(
    store: LocalArtifactStore, agent_id: int
) -> Iterator[ValidationArtifact]:
    """Yields every locally-stored artifact for an agent. Skips bad files."""
    for h in store.list_hashes(agent_id):
        artifact = store.read(agent_id, h)
        if artifact is not None:
            yield artifact


# --------------------------------------------------------------- result types

@dataclass(frozen=True)
class ArtifactCheck:
    """
    One row of the verification report.

    `passed` honors `chain_required`: when chain access was unavailable,
    a check passes on local integrity alone. This keeps the per-row and
    overall verdicts consistent.
    """

    artifact_hash: str
    local_present: bool
    local_filename_matches_payload: bool
    on_chain_present: bool
    notes: tuple[str, ...] = ()
    chain_required: bool = True

    @property
    def passed(self) -> bool:
        local_ok = self.local_present and self.local_filename_matches_payload
        if not self.chain_required:
            return local_ok
        return local_ok and self.on_chain_present


@dataclass(frozen=True)
class VerificationReport:
    agent_id: int
    on_chain_count: int
    local_count: int
    checks: tuple[ArtifactCheck, ...]
    chain_source: str               # "agent_artifacts" | "validation_registry" | "none"
    chain_skipped: bool = False     # true when no RPC; only local checks ran

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        passed_n = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        verdict = "PASS" if self.passed else "FAIL"
        skipped = " (chain check skipped)" if self.chain_skipped else ""
        return (
            f"agent {self.agent_id}: {verdict}{skipped} — "
            f"{passed_n}/{total} artifacts verified, "
            f"{self.on_chain_count} on-chain refs, {self.local_count} local files"
        )


# --------------------------------------------------------------- core

def verify_local_only(
    store: LocalArtifactStore, agent_id: int
) -> VerificationReport:
    """
    Filename ↔ payload check, no chain access.

    This is the minimum useful verification: it catches anyone who has
    edited the off-chain JSON after publication, even if they have no RPC.
    """
    checks: list[ArtifactCheck] = []
    for h in store.list_hashes(agent_id):
        raw = store.read_raw(agent_id, h)
        if raw is None:
            checks.append(
                ArtifactCheck(
                    artifact_hash=h,
                    local_present=False,
                    local_filename_matches_payload=False,
                    on_chain_present=False,
                    notes=("file vanished mid-read",),
                    chain_required=False,
                )
            )
            continue
        # Re-derive the canonical hash from the dict and compare to the
        # filename. We do not require pydantic validation here so a partial
        # corruption shows up as a hash mismatch rather than a parse error.
        actual = canonical_hash(raw)
        checks.append(
            ArtifactCheck(
                artifact_hash=h,
                local_present=True,
                local_filename_matches_payload=(actual.lower() == h.lower()),
                on_chain_present=False,
                notes=() if actual.lower() == h.lower() else (f"recomputed={actual}",),
                chain_required=False,
            )
        )
    return VerificationReport(
        agent_id=agent_id,
        on_chain_count=0,
        local_count=len(checks),
        checks=tuple(checks),
        chain_source="none",
        chain_skipped=True,
    )


def verify(
    agent_id: int,
    store: LocalArtifactStore,
    artifacts_client: ArtifactsClient | None = None,
) -> VerificationReport:
    """
    Full verification: chain history × local store.

    If `artifacts_client` is omitted or in dry-run, falls back to local-only
    verification with `chain_skipped=True`.
    """
    if artifacts_client is None or artifacts_client._c.config.dry_run:
        return verify_local_only(store, agent_id)

    refs: list[OnChainArtifactRef] = artifacts_client.fetch_history()
    chain_hashes = {r.artifact_hash.lower() for r in refs}
    local_hashes = {h.lower() for h in store.list_hashes(agent_id)}

    chain_source = refs[0].source if refs else "agent_artifacts"
    all_hashes = sorted(chain_hashes | local_hashes)
    checks: list[ArtifactCheck] = []
    for h in all_hashes:
        local_present = h in local_hashes
        on_chain_present = h in chain_hashes
        notes: list[str] = []

        local_match = False
        if local_present:
            raw = store.read_raw(agent_id, h)
            if raw is not None:
                actual = canonical_hash(raw)
                local_match = actual.lower() == h
                if not local_match:
                    notes.append(f"recomputed={actual}")
        else:
            notes.append("missing locally — chain anchor without published payload")

        if not on_chain_present:
            notes.append("local file with no on-chain anchor")

        checks.append(
            ArtifactCheck(
                artifact_hash=h,
                local_present=local_present,
                local_filename_matches_payload=local_match,
                on_chain_present=on_chain_present,
                notes=tuple(notes),
            )
        )

    return VerificationReport(
        agent_id=agent_id,
        on_chain_count=len(chain_hashes),
        local_count=len(local_hashes),
        checks=tuple(checks),
        chain_source=chain_source,
        chain_skipped=False,
    )
