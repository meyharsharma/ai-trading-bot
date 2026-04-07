"""Verifier tests — local-only path (no chain) and tampering detection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.chain.verifier import (
    LocalArtifactStore,
    verify,
    verify_local_only,
)
from agent.state import ValidationArtifact, canonical_hash, utcnow


def _artifact(suffix: int = 1) -> ValidationArtifact:
    return ValidationArtifact(
        decision_hash="0x" + str(suffix) * 64,
        trade_hash="0x" + "f" * 64,
        risk_checks={"max_risk_per_trade": True},
        pre_state_hash="0x" + "a" * 64,
        post_state_hash="0x" + "b" * 64,
        reasoning_uri=f"data:,reason-{suffix}",
        timestamp=utcnow(),
        agent_id="42",
    )


# --------------------------------------------------------------- store

def test_store_write_uses_canonical_hash_filename(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    a = _artifact()
    path = store.write(agent_id=42, artifact=a)
    expected = canonical_hash(a)
    assert path.name == f"{expected}.json"
    # Filename in `list_hashes` is lowercased.
    assert store.list_hashes(42) == [expected.lower()]


def test_store_round_trip(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    a = _artifact()
    store.write(42, a)
    loaded = store.read(42, canonical_hash(a))
    assert loaded == a


def test_store_missing_returns_none(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    assert store.read(42, "0x" + "0" * 64) is None
    assert store.list_hashes(42) == []


# --------------------------------------------------------------- local-only

def test_verify_local_only_passes_for_clean_store(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    for i in range(3):
        store.write(42, _artifact(i + 1))

    report = verify_local_only(store, agent_id=42)
    assert report.chain_skipped is True
    assert report.passed is True
    assert report.local_count == 3
    assert all(c.local_filename_matches_payload for c in report.checks)


def test_verify_local_only_detects_tampered_payload(tmp_path: Path):
    """Edit the file after publication; the verifier must catch it."""
    store = LocalArtifactStore(tmp_path)
    a = _artifact(1)
    path = store.write(42, a)

    # Mutate the file *without* renaming it. This is the exact attack the
    # filename-vs-payload check exists to defeat.
    raw = json.loads(path.read_text())
    raw["reasoning_uri"] = "data:,sneaky-replacement"
    path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")))

    report = verify_local_only(store, agent_id=42)
    assert report.passed is False
    bad = report.checks[0]
    assert bad.local_present is True
    assert bad.local_filename_matches_payload is False
    assert any("recomputed=" in note for note in bad.notes)


def test_verify_handles_empty_directory(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    report = verify_local_only(store, agent_id=42)
    # No files = vacuously fine for local-only mode.
    assert report.local_count == 0
    assert report.passed is True


# --------------------------------------------------------------- full verify

def test_verify_falls_back_to_local_when_no_client(tmp_path: Path):
    store = LocalArtifactStore(tmp_path)
    store.write(42, _artifact(1))
    report = verify(agent_id=42, store=store, artifacts_client=None)
    assert report.chain_skipped is True
    assert report.passed is True


def test_verify_dry_run_client_falls_back_to_local(tmp_path: Path):
    """A dry-run ChainClient cannot fetch chain history; verifier degrades."""
    from agent.chain import ArtifactsClient, ChainClient, ChainConfig

    store = LocalArtifactStore(tmp_path)
    store.write(42, _artifact(1))

    client = ArtifactsClient(ChainClient(ChainConfig(dry_run=True)), agent_id=42)
    report = verify(agent_id=42, store=store, artifacts_client=client)
    assert report.chain_skipped is True
    assert report.passed is True
