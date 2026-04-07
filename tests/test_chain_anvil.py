"""
Integration tests against a local anvil fork of Base Sepolia.

These tests are skipped unless `./scripts/dev_anvil.sh` is running. They
exist so we can exercise the **real** ERC-8004 IdentityRegistry bytecode
without burning testnet gas. The fork shadows the canonical
0x8004A8…BD9e deployment so `register()` actually mints an ERC-721 token
locally.

Run with:
    ./scripts/dev_anvil.sh &
    PYTHONPATH=src uv run pytest tests/test_chain_anvil.py -m integration -v
"""
from __future__ import annotations

import pytest

from agent.chain import AgentManifest, IdentityClient
from agent.chain._client import DEFAULT_IDENTITY_REGISTRY


pytestmark = pytest.mark.integration


def test_chain_client_connects_to_anvil(anvil_chain_client):
    """
    The most basic round-trip: web3 talks to anvil, the prefunded account
    is loaded, and the chain id matches the fork.
    """
    cc = anvil_chain_client
    assert cc.w3.is_connected()
    # anvil mirrors the forked chain id (84532 by default).
    assert cc.w3.eth.chain_id == cc.config.chain_id
    # Prefunded account #0 is non-empty and has > 0 balance on a fresh fork.
    assert cc.w3.eth.get_balance(cc.address) > 0


def test_register_identity_against_real_contract(anvil_chain_client):
    """
    Calls `register(string)` on the real ERC-8004 IdentityRegistry shadowed
    by the fork. Confirms:
        * the tx broadcasts and confirms,
        * the receipt is *not* a dry-run mock,
        * an agent_id is decoded from the ERC-721 mint log.
    """
    identity = IdentityClient(anvil_chain_client)
    manifest = AgentManifest(
        name="anvil-integration",
        owner_address=anvil_chain_client.address,
        model="claude-opus-4-6",
        repo_url="https://example.com/anvil",
        strategy_summary="integration test",
    )
    record = identity.register(manifest)

    assert record.tx_hash.startswith("0x") and len(record.tx_hash) == 66
    assert record.agent_id > 0
    assert record.registry_address == DEFAULT_IDENTITY_REGISTRY


def test_two_registrations_yield_distinct_ids(anvil_chain_client):
    """
    Re-registering the same manifest twice mints two distinct ERC-721 ids.
    Catches any accidental nonce/cache reuse in `ChainClient.send()`.
    """
    identity = IdentityClient(anvil_chain_client)
    manifest = AgentManifest(
        name="anvil-twice",
        owner_address=anvil_chain_client.address,
        model="claude-opus-4-6",
        repo_url="https://example.com/twice",
        strategy_summary="nonce check",
    )
    a = identity.register(manifest)
    b = identity.register(manifest)
    assert a.agent_id != b.agent_id
    assert a.tx_hash != b.tx_hash
