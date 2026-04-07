"""Block-explorer URL helper tests."""
from __future__ import annotations

import pytest

from agent.chain.explorer import (
    DEFAULT_CHAIN_ID,
    address_url,
    agent_token_url,
    get_explorer,
    tx_url,
)


def test_default_is_base_sepolia():
    assert DEFAULT_CHAIN_ID == 84532
    explorer = get_explorer()
    assert "sepolia.basescan.org" in explorer.base_url


def test_tx_url_base_sepolia():
    h = "0x" + "ab" * 32
    url = tx_url(h)
    assert url == f"https://sepolia.basescan.org/tx/{h}"


def test_tx_url_normalizes_case_and_prefix():
    """0x prefix added if missing; hex lowercased."""
    raw = "AB" * 32
    url = tx_url(raw)
    assert url.endswith("/" + "0x" + "ab" * 32)


def test_tx_url_rejects_non_hex():
    with pytest.raises(ValueError):
        tx_url("0xnothex!!")


def test_tx_url_rejects_empty():
    with pytest.raises(ValueError):
        tx_url("")


def test_address_url_mainnet():
    addr = "0x8004A818BFB912233c491871b3d84c89A494BD9e"
    url = address_url(addr, chain_id=1)
    assert url == "https://etherscan.io/address/" + addr.lower()


def test_unknown_chain_raises():
    with pytest.raises(KeyError):
        get_explorer(chain_id=999_999)


def test_agent_token_url_includes_token_id():
    addr = "0x8004A818BFB912233c491871b3d84c89A494BD9e"
    url = agent_token_url(addr, agent_id=42)
    assert "/token/" in url
    assert url.endswith("?a=42")


def test_explorer_block_url():
    e = get_explorer(84532)
    assert e.block(123).endswith("/block/123")
    with pytest.raises(ValueError):
        e.block(-1)


def test_all_supported_chains_have_consistent_shape():
    for chain_id in (1, 11155111, 8453, 84532):
        e = get_explorer(chain_id)
        assert e.base_url.startswith("https://")
        assert not e.base_url.endswith("/")
