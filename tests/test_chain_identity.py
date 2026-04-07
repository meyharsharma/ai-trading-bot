"""Identity layer tests — all run in dry-run mode (no RPC, no key)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.chain import (
    AgentManifest,
    ChainClient,
    ChainConfig,
    IdentityClient,
    load_persisted_id,
    persist_identity,
)
from agent.chain.identity import IdentityRecord


def _dry_client() -> ChainClient:
    return ChainClient(ChainConfig(dry_run=True))


def _manifest() -> AgentManifest:
    return AgentManifest(
        name="test-agent",
        owner_address="0x000000000000000000000000000000000000dEaD",
        model="claude-opus-4-6",
        repo_url="https://github.com/example/agent",
        strategy_summary="momentum + risk gate",
    )


def test_manifest_data_uri_round_trip():
    m = _manifest()
    uri = m.to_data_uri()
    assert uri.startswith("data:application/json;base64,")

    import base64

    payload = base64.b64decode(uri.split(",", 1)[1])
    parsed = json.loads(payload)
    # Canonical encoding: keys sorted, no whitespace.
    assert parsed["name"] == "test-agent"
    assert parsed["model"] == "claude-opus-4-6"


def test_dry_run_register_returns_deterministic_id():
    client = IdentityClient(_dry_client())
    rec_a = client.register(_manifest())
    rec_b = client.register(_manifest())
    # Same manifest in → same mock receipt + agent_id out. This is what makes
    # the dry-run mode usable in CI.
    assert rec_a.agent_id == rec_b.agent_id
    assert rec_a.tx_hash == rec_b.tx_hash
    assert rec_a.tx_hash.startswith("0x")
    assert rec_a.agent_id > 0


def test_dry_run_register_changes_with_manifest():
    client = IdentityClient(_dry_client())
    rec_a = client.register(_manifest())
    other = AgentManifest(
        name="other-agent",
        owner_address="0x000000000000000000000000000000000000dEaD",
        model="claude-opus-4-6",
        repo_url="https://example.com/x",
        strategy_summary="different",
    )
    rec_b = client.register(other)
    assert rec_a.agent_id != rec_b.agent_id


def test_persist_and_load(tmp_path: Path):
    rec = IdentityRecord(
        agent_id=42,
        owner_address="0xabc",
        tx_hash="0x" + "1" * 64,
        chain_id=84532,
        registry_address="0x8004A818BFB912233c491871b3d84c89A494BD9e",
        agent_uri="data:application/json;base64,e30=",
    )
    target = tmp_path / "agent_id.json"
    persist_identity(rec, path=target)
    loaded = load_persisted_id(path=target)
    assert loaded == rec


def test_load_returns_none_when_missing(tmp_path: Path):
    assert load_persisted_id(path=tmp_path / "nope.json") is None


def test_chain_config_dry_run_when_env_empty():
    cfg = ChainConfig.from_env(env={})
    assert cfg.dry_run is True


def test_chain_config_live_when_both_set():
    cfg = ChainConfig.from_env(
        env={"RPC_URL": "https://sepolia.base.org", "PRIVATE_KEY": "0x" + "1" * 64}
    )
    assert cfg.dry_run is False
    assert cfg.chain_id == 84532


def test_chain_config_dry_run_when_only_rpc_set():
    """Half-configured env should fail safe to dry-run, not crash."""
    cfg = ChainConfig.from_env(env={"RPC_URL": "https://sepolia.base.org"})
    assert cfg.dry_run is True
