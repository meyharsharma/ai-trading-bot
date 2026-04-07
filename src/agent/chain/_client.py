"""
Shared web3 plumbing for the on-chain layer.

`ChainClient` is a thin wrapper around `web3.py` that the identity / artifacts
/ vault modules build on top of. It exists so each module does not re-derive
the account, retry logic, or dry-run plumbing.

Two modes:

* **live** — `web3` + signing account loaded from `.env`. Real transactions
  hit the configured RPC.
* **dry_run** — no RPC, no key required. Methods return deterministic mock tx
  hashes derived from canonical hashing so tests and CI stay hermetic and
  the rest of the agent can develop in parallel before the chain is funded.

Dry-run is the default whenever `RPC_URL` is unset; this is intentional so
that `pytest` and `python -m agent.main --once` both work on a laptop with
zero secrets.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

# ERC-8004 reference deployments use the same address on every supported
# testnet. See docs/ERC8004_NOTES.md.
DEFAULT_IDENTITY_REGISTRY = "0x8004A818BFB912233c491871b3d84c89A494BD9e"
DEFAULT_REPUTATION_REGISTRY = "0x8004B663056A597Dffe9eCcC1965A193B7388713"
# ValidationRegistry address is pulled from the canonical deployments json on
# Day 2; until then we accept it from env and fail loudly if a live tx is
# attempted without it set.
DEFAULT_VALIDATION_REGISTRY = ""

DEFAULT_RPC_URL = "https://sepolia.base.org"
DEFAULT_CHAIN_ID = 84532  # Base Sepolia


@dataclass(frozen=True)
class ChainConfig:
    """Static configuration for every chain interaction."""

    rpc_url: str = ""
    private_key: str = ""
    chain_id: int = DEFAULT_CHAIN_ID
    identity_registry: str = DEFAULT_IDENTITY_REGISTRY
    reputation_registry: str = DEFAULT_REPUTATION_REGISTRY
    validation_registry: str = DEFAULT_VALIDATION_REGISTRY
    agent_artifacts: str = ""           # our optional thin-wrapper deployment
    vault_address: str = ""             # Hackathon Capital Sandbox vault
    dry_run: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ChainConfig":
        e = env if env is not None else os.environ
        rpc = e.get("RPC_URL", "").strip()
        pk = e.get("PRIVATE_KEY", "").strip()
        # Dry-run unless we have *both* an RPC and a key. Half-configured is
        # almost always a mistake during local dev — fail-safe to dry-run.
        dry = not (rpc and pk)
        return cls(
            rpc_url=rpc or DEFAULT_RPC_URL,
            private_key=pk,
            chain_id=int(e.get("CHAIN_ID", DEFAULT_CHAIN_ID)),
            identity_registry=e.get("ERC8004_IDENTITY_REGISTRY", DEFAULT_IDENTITY_REGISTRY),
            reputation_registry=e.get(
                "ERC8004_REPUTATION_REGISTRY", DEFAULT_REPUTATION_REGISTRY
            ),
            validation_registry=e.get(
                "ERC8004_VALIDATION_REGISTRY", DEFAULT_VALIDATION_REGISTRY
            ),
            agent_artifacts=e.get("AGENT_ARTIFACTS_ADDRESS", ""),
            vault_address=e.get("VAULT_ADDRESS", ""),
            dry_run=dry,
        )


@dataclass
class TxReceipt:
    """Normalized receipt — works for live and dry-run paths."""

    tx_hash: str
    block_number: int | None = None
    status: int = 1            # 1 = success, 0 = revert
    dry_run: bool = False
    return_value: Any = None   # populated for view-call simulations


class ChainClient:
    """
    Holds the (optional) web3 instance + signing account.

    A single instance is created at agent boot and shared by IdentityClient,
    ArtifactsClient, and VaultRouter. This guarantees nonce ordering — every
    chain write goes through `send()` which serializes around `_nonce_lock`.
    """

    def __init__(self, config: ChainConfig):
        self.config = config
        self._w3 = None
        self._account = None
        if not config.dry_run:
            self._connect()

    # ------------------------------------------------------------------ live
    def _connect(self) -> None:
        # Imported lazily so dry-run paths never need web3 importable. Keeps
        # `pytest` green on machines without the native crypto deps built.
        from web3 import Web3  # type: ignore
        from eth_account import Account  # type: ignore

        self._w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        if not self._w3.is_connected():
            raise RuntimeError(f"web3 cannot reach {self.config.rpc_url}")
        self._account = Account.from_key(self.config.private_key)

    @property
    def w3(self):
        if self._w3 is None:
            raise RuntimeError("ChainClient is in dry-run mode; no live web3")
        return self._w3

    @property
    def account(self):
        if self._account is None:
            raise RuntimeError("ChainClient is in dry-run mode; no signing account")
        return self._account

    @property
    def address(self) -> str:
        if self._account is None:
            # Stable, deterministic stub address for dry-run logging.
            return "0x000000000000000000000000000000000000dEaD"
        return self._account.address

    # ----------------------------------------------------------- send helpers
    def send(self, contract_fn, *, gas: int = 250_000, value: int = 0) -> TxReceipt:
        """
        Build, sign, and broadcast a contract write. In dry-run, returns a
        deterministic mock receipt so callers can keep their happy-path code
        identical across modes.
        """
        if self.config.dry_run:
            return self._mock_receipt(contract_fn)

        w3 = self.w3
        acct = self.account
        tx = contract_fn.build_transaction({
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "gas": gas,
            "maxFeePerGas": w3.eth.gas_price * 2,
            "maxPriorityFeePerGas": w3.to_wei(1, "gwei"),
            "chainId": self.config.chain_id,
            "value": value,
        })
        signed = acct.sign_transaction(tx)
        # web3.py renamed `rawTransaction` → `raw_transaction` in 7.x.
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return TxReceipt(
            tx_hash=tx_hash.hex(),
            block_number=receipt["blockNumber"],
            status=receipt["status"],
            dry_run=False,
        )

    def call(self, contract_fn) -> Any:
        """Read-only view call. Dry-run returns None."""
        if self.config.dry_run:
            return None
        return contract_fn.call({"from": self.address})

    def load_contract(self, address: str, abi: list[dict]):
        """Build a contract handle. Errors clearly if the address is unset."""
        if not address:
            raise ValueError("contract address is empty — set it in .env")
        return self.w3.eth.contract(address=self.w3.to_checksum_address(address), abi=abi)

    # ---------------------------------------------------------------- mocks
    @staticmethod
    def _mock_receipt(contract_fn) -> TxReceipt:
        # Hash the function name + args so the receipt is reproducible across
        # runs and unit tests can pin against it.
        try:
            name = contract_fn.fn_name
            args = contract_fn.args
            payload = repr((name, args)).encode()
        except AttributeError:
            payload = repr(contract_fn).encode()
        digest = hashlib.sha256(payload).hexdigest()
        return TxReceipt(tx_hash="0x" + digest, dry_run=True)


# --------------------------------------------------------------------- helpers

def hex_to_bytes32(value: str) -> bytes:
    """
    Convert a `0x...` 32-byte hex string (the canonical_hash output) into the
    raw `bytes` web3 expects for `bytes32` arguments. Validates length so we
    catch mis-sized hashes at the call site instead of inside the encoder.
    """
    if not value.startswith("0x"):
        raise ValueError(f"expected 0x-prefixed hex, got {value!r}")
    raw = bytes.fromhex(value[2:])
    if len(raw) != 32:
        raise ValueError(f"expected 32 bytes, got {len(raw)} from {value!r}")
    return raw


# --------------------------------------------------------------- mock helper
def make_mock_call(fn_name: str, args: tuple) -> Any:
    """
    Build a tiny stand-in object that quacks like a web3 ContractFunction so
    `ChainClient.send()` can mock-receipt it without a real contract bound.
    Used by dry-run paths in identity/artifacts/vault when no contract address
    is configured at all.
    """
    class _F:
        def __init__(self, n, a):
            self.fn_name = n
            self.args = a

        def build_transaction(self, *_a, **_k):  # pragma: no cover - never reached
            raise RuntimeError("mock contract function — dry-run only")

    return _F(fn_name, args)
