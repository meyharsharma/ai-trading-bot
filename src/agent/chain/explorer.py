"""
Block-explorer URL helpers.

Every artifact and tx we surface in the README, demo video, or verifier
output should be a clickable link. This module is the single source of truth
for which explorer maps to which chain id, so the rest of the code never
constructs URLs by hand.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Explorer:
    name: str
    base_url: str  # no trailing slash

    def tx(self, tx_hash: str) -> str:
        return f"{self.base_url}/tx/{_normalize_hex(tx_hash)}"

    def address(self, addr: str) -> str:
        return f"{self.base_url}/address/{_normalize_hex(addr)}"

    def block(self, number: int) -> str:
        if number < 0:
            raise ValueError(f"block number must be >= 0, got {number}")
        return f"{self.base_url}/block/{number}"

    def token(self, contract: str, token_id: int | None = None) -> str:
        url = f"{self.base_url}/token/{_normalize_hex(contract)}"
        if token_id is not None:
            # BaseScan / Etherscan use ?a= for the per-token-id deep link.
            url += f"?a={token_id}"
        return url


# Chain id → explorer. Centralized so a new chain only needs one entry.
EXPLORERS: dict[int, Explorer] = {
    1: Explorer("Etherscan", "https://etherscan.io"),
    11155111: Explorer("Sepolia Etherscan", "https://sepolia.etherscan.io"),
    8453: Explorer("BaseScan", "https://basescan.org"),
    84532: Explorer("Base Sepolia BaseScan", "https://sepolia.basescan.org"),
    59141: Explorer("Linea Sepolia", "https://sepolia.lineascan.build"),
    296: Explorer("HashScan (Hedera Testnet)", "https://hashscan.io/testnet"),
}

DEFAULT_CHAIN_ID = 84532  # Base Sepolia — see docs/ERC8004_NOTES.md


def get_explorer(chain_id: int = DEFAULT_CHAIN_ID) -> Explorer:
    """Returns the explorer for the chain. Raises if unknown — fail loud."""
    explorer = EXPLORERS.get(chain_id)
    if explorer is None:
        raise KeyError(
            f"no explorer registered for chain_id={chain_id}; "
            f"add one to agent.chain.explorer.EXPLORERS"
        )
    return explorer


def tx_url(tx_hash: str, chain_id: int = DEFAULT_CHAIN_ID) -> str:
    """Convenience: BaseScan/Etherscan URL for a tx hash on the given chain."""
    return get_explorer(chain_id).tx(tx_hash)


def address_url(address: str, chain_id: int = DEFAULT_CHAIN_ID) -> str:
    """Convenience: explorer URL for an address on the given chain."""
    return get_explorer(chain_id).address(address)


def agent_token_url(
    registry_address: str,
    agent_id: int,
    chain_id: int = DEFAULT_CHAIN_ID,
) -> str:
    """
    Deep link for an ERC-8004 agent NFT — points at the IdentityRegistry
    contract with the agent's tokenId. Useful in the README to give judges
    a one-click view of the agent's identity.
    """
    return get_explorer(chain_id).token(registry_address, token_id=agent_id)


def _normalize_hex(value: str) -> str:
    """Lowercases and ensures the `0x` prefix. Rejects obviously bad input."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty string, got {value!r}")
    v = value.strip().lower()
    if not v.startswith("0x"):
        v = "0x" + v
    # Light shape validation — explorers themselves will give clearer errors
    # for malformed hashes, but catching obvious garbage here helps the demo
    # script fail at the source rather than in a browser.
    body = v[2:]
    if not body or any(c not in "0123456789abcdef" for c in body):
        raise ValueError(f"not a hex string: {value!r}")
    return v
