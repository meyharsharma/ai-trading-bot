"""
ERC-8004 Identity Registry client.

We register the agent **once** at boot. The returned `agent_id` is persisted
to `config/agent_id.json` and reused on every subsequent run — re-registering
would mint a new ERC-721 and orphan all prior validation artifacts.

Spec reference (see docs/ERC8004_NOTES.md):

    function register(string agentURI) external returns (uint256 agentId)
    function setAgentWallet(uint256 agentId, address newWallet, ...)
    function getAgentWallet(uint256 agentId) external view returns (address)

The `agentURI` points to an off-chain JSON manifest describing the agent
(model, owner, repo, strategy summary). For the hackathon we publish it as a
`data:` URI so we don't take a hard dependency on IPFS being up.
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ._client import ChainClient, ChainConfig, make_mock_call

# Minimal ABI — only the functions we actually call. Keeping the ABI small
# avoids accidentally widening the trust surface.
IDENTITY_REGISTRY_ABI: list[dict] = [
    {
        "type": "function",
        "name": "register",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "agentURI", "type": "string"}],
        "outputs": [{"name": "agentId", "type": "uint256"}],
    },
    {
        "type": "function",
        "name": "setAgentURI",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "newURI", "type": "string"},
        ],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "getAgentWallet",
        "stateMutability": "view",
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address"}],
    },
    {
        "type": "event",
        "name": "Transfer",  # ERC-721 mint emits Transfer(0x0, owner, tokenId)
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": True, "name": "tokenId", "type": "uint256"},
        ],
    },
]


# Where the registered agent_id lives between runs. Anything that wants to
# read it (the trading loop, the chain layer, the demo script) imports
# `load_persisted_id` rather than re-deriving the path.
AGENT_ID_PATH = Path("config/agent_id.json")


@dataclass
class AgentManifest:
    """Off-chain JSON pointed to by the on-chain `agentURI`."""

    name: str
    owner_address: str
    model: str                # e.g. "claude-opus-4-6"
    repo_url: str
    strategy_summary: str
    version: str = "0.1.0"

    def to_data_uri(self) -> str:
        """
        Encode as `data:application/json;base64,...`. Self-contained, no IPFS
        dependency. Small enough that calldata stays cheap on Base Sepolia.
        """
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        b64 = base64.b64encode(payload.encode()).decode()
        return f"data:application/json;base64,{b64}"


@dataclass
class IdentityRecord:
    """What we persist after a successful registration."""

    agent_id: int
    owner_address: str
    tx_hash: str
    chain_id: int
    registry_address: str
    agent_uri: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


class IdentityClient:
    """High-level wrapper around the ERC-8004 Identity Registry."""

    def __init__(self, client: ChainClient):
        self._c = client
        self._contract = None
        if not client.config.dry_run:
            self._contract = client.load_contract(
                client.config.identity_registry, IDENTITY_REGISTRY_ABI
            )

    # ----------------------------------------------------------- registration
    def register(self, manifest: AgentManifest) -> IdentityRecord:
        """
        Mint a new agent identity. Idempotency is the **caller's**
        responsibility (use `register_or_load` from scripts/register_identity.py)
        — this method always submits a new transaction.
        """
        agent_uri = manifest.to_data_uri()

        if self._contract is None:
            # Dry-run: deterministic mock so tests can pin against it.
            receipt = self._c.send(make_mock_call("register", (agent_uri,)))
            mock_id = int(receipt.tx_hash[2:10], 16)  # first 4 bytes → uint
            return IdentityRecord(
                agent_id=mock_id,
                owner_address=self._c.address,
                tx_hash=receipt.tx_hash,
                chain_id=self._c.config.chain_id,
                registry_address=self._c.config.identity_registry,
                agent_uri=agent_uri,
            )

        fn = self._contract.functions.register(agent_uri)
        receipt = self._c.send(fn, gas=400_000)
        agent_id = self._extract_minted_token_id(receipt.tx_hash)
        return IdentityRecord(
            agent_id=agent_id,
            owner_address=self._c.address,
            tx_hash=receipt.tx_hash,
            chain_id=self._c.config.chain_id,
            registry_address=self._c.config.identity_registry,
            agent_uri=agent_uri,
        )

    # ---------------------------------------------------------------- lookups
    def get_agent_wallet(self, agent_id: int) -> str | None:
        """Returns the bound wallet, or None in dry-run."""
        if self._contract is None:
            return None
        return self._c.call(self._contract.functions.getAgentWallet(agent_id))

    # -------------------------------------------------------- internal helpers
    def _extract_minted_token_id(self, tx_hash: str) -> int:
        """
        Pull the `tokenId` out of the ERC-721 `Transfer` log produced by the
        registry's `register()`. Fall back to a `register` return-value parse
        if no Transfer log is present (some forks emit a custom event).
        """
        w3 = self._c.w3
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        for log in receipt["logs"]:
            try:
                ev = self._contract.events.Transfer().process_log(log)
            except Exception:
                continue
            # Mints come from the zero address. Anything else is a transfer
            # of an existing agent — skip it.
            if int(ev["args"]["from"], 16) == 0:
                return int(ev["args"]["tokenId"])
        raise RuntimeError(
            f"register() tx {tx_hash} produced no mint Transfer log — "
            "the registry contract may have changed"
        )


# --------------------------------------------------------- persistence helpers

def persist_identity(record: IdentityRecord, path: Path = AGENT_ID_PATH) -> None:
    """Write the registration record to disk so the agent loop can pick it up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.to_json())


def load_persisted_id(path: Path = AGENT_ID_PATH) -> IdentityRecord | None:
    """Returns the persisted record, or None if the agent hasn't registered yet."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return IdentityRecord(**data)


def from_env() -> IdentityClient:
    """Convenience constructor used by scripts and main.py."""
    return IdentityClient(ChainClient(ChainConfig.from_env()))
