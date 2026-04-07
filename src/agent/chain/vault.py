"""
Hackathon Capital Sandbox vault router.

The brief mandates that the agent "operate through the Hackathon Capital
Sandbox vault and risk router". As of Day 1 the organizers have **not yet
published a vault address or ABI** — see `docs/ERC8004_NOTES.md` for the
status of that ask.

This module is the seam where that integration will land. We expose:

* `VaultRouter` — a small client whose `route_intent()` method takes a
  `(RiskedDecision, Fill)` pair and submits an "intent" to the vault.
* A pluggable ABI: when the real ABI lands we replace `_PLACEHOLDER_ABI` and
  the rest of the agent does not change.

Until the address is known we run in **fallback mode**: the router emits the
same intent as a structured event via `AgentArtifacts.sol` (or, in dry-run, a
deterministic mock receipt). This keeps the trading loop's contract stable so
no other worktree blocks on us.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.state import Fill, RiskedDecision, canonical_hash

from ._client import ChainClient, ChainConfig, hex_to_bytes32, make_mock_call

# Placeholder until the organizers publish the real Hackathon Capital Sandbox
# ABI. The real surface is expected to be something like:
#     submitIntent(uint256 agentId, bytes32 intentHash, bytes payload)
# Keeping the call shape narrow now means swapping in the real ABI is a
# one-file change.
_PLACEHOLDER_VAULT_ABI: list[dict] = [
    {
        "type": "function",
        "name": "submitIntent",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "intentHash", "type": "bytes32"},
            {"name": "payload", "type": "bytes"},
        ],
        "outputs": [],
    },
]


@dataclass(frozen=True)
class VaultIntent:
    """
    Wire-level intent we submit to the vault.

    `intent_hash` is the canonical hash of the (decision, fill) pair so the
    vault submission is one-to-one with the on-chain validation artifact for
    the same trade. Auditors can join the two by `intent_hash == artifact's
    pre/post pair source`.
    """

    agent_id: int
    symbol: str
    side: str
    quantity: float
    fill_price: float
    intent_hash: str
    decision_hash: str

    @classmethod
    def from_decision_fill(
        cls,
        agent_id: int,
        risked: RiskedDecision,
        fill: Fill,
    ) -> "VaultIntent":
        decision_hash = canonical_hash(risked.decision)
        # Hashing the (decision, fill) pair as one canonical blob means the
        # intent_hash changes if either side changes — so the auditor cannot
        # be tricked by swapping the fill out from under the decision.
        pair_hash = canonical_hash(
            {"decision": risked.decision.model_dump(mode="json"),
             "fill": fill.model_dump(mode="json"),
             "agent_id": agent_id}
        )
        return cls(
            agent_id=agent_id,
            symbol=fill.order.symbol,
            side=fill.order.side,
            quantity=fill.order.quantity,
            fill_price=fill.fill_price,
            intent_hash=pair_hash,
            decision_hash=decision_hash,
        )

    def to_payload_bytes(self) -> bytes:
        """ABI-friendly opaque payload — placeholder until real schema lands."""
        # Compact JSON because gas. The vault is expected to treat this as a
        # blob and only verify the hash on-chain.
        import json
        return json.dumps(
            {
                "agent_id": self.agent_id,
                "symbol": self.symbol,
                "side": self.side,
                "quantity": self.quantity,
                "fill_price": self.fill_price,
                "decision_hash": self.decision_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


@dataclass
class VaultReceipt:
    intent: VaultIntent
    tx_hash: str
    via: str            # "vault" | "fallback_event" | "dry_run"
    block_number: int | None = None


class VaultRouter:
    """
    Routes intents through the Hackathon Capital Sandbox vault.

    Three modes, picked at boot:
        * `vault`         — `VAULT_ADDRESS` is set; submit via the real ABI.
        * `fallback_event`— vault address unknown but we have a wrapper deployed;
                           emit the intent as an `AgentArtifacts` event so the
                           trading loop never has to handle "no vault yet".
        * `dry_run`       — no RPC at all; deterministic mock receipt for tests.
    """

    def __init__(self, client: ChainClient):
        self._c = client
        self._vault = None
        self._fallback = None

        if client.config.dry_run:
            self._mode = "dry_run"
            return

        if client.config.vault_address:
            self._vault = client.load_contract(
                client.config.vault_address, _PLACEHOLDER_VAULT_ABI
            )
            self._mode = "vault"
        elif client.config.agent_artifacts:
            # Lazy import to avoid a hard cycle (artifacts → vault → artifacts).
            from .artifacts import AGENT_ARTIFACTS_ABI

            self._fallback = client.load_contract(
                client.config.agent_artifacts, AGENT_ARTIFACTS_ABI
            )
            self._mode = "fallback_event"
        else:
            raise RuntimeError(
                "VaultRouter: no vault address and no fallback wrapper. "
                "Set VAULT_ADDRESS or AGENT_ARTIFACTS_ADDRESS, or use dry-run."
            )

    @property
    def mode(self) -> str:
        return self._mode

    # ----------------------------------------------------------------- routing
    def route_intent(
        self,
        agent_id: int,
        risked: RiskedDecision,
        fill: Fill,
    ) -> VaultReceipt:
        """
        Submit a single intent and return a normalized receipt. Raises on RPC
        failure — the trading loop catches this and marks the trade
        `unverified`, identical to artifact submission failures.
        """
        if not risked.passed:
            # Defensive: a rejected RiskedDecision should never reach the
            # vault. Loud failure beats a silently-anchored bad trade.
            raise ValueError("VaultRouter refuses to route a rejected RiskedDecision")

        intent = VaultIntent.from_decision_fill(agent_id, risked, fill)

        if self._mode == "vault":
            fn = self._vault.functions.submitIntent(
                intent.agent_id,
                hex_to_bytes32(intent.intent_hash),
                intent.to_payload_bytes(),
            )
            receipt = self._c.send(fn, gas=300_000)
            return VaultReceipt(
                intent=intent,
                tx_hash=receipt.tx_hash,
                via="vault",
                block_number=receipt.block_number,
            )

        if self._mode == "fallback_event":
            # Re-use the wrapper's `submit` event as a placeholder vault
            # intent. The decision_hash field doubles as the intent_hash and
            # the trade_hash field carries the pair hash.
            fn = self._fallback.functions.submit(
                intent.agent_id,
                hex_to_bytes32(intent.decision_hash),
                hex_to_bytes32(intent.intent_hash),
                hex_to_bytes32("0x" + "0" * 64),
                hex_to_bytes32("0x" + "0" * 64),
                f"vault-fallback://intent/{intent.intent_hash}",
            )
            receipt = self._c.send(fn, gas=300_000)
            return VaultReceipt(
                intent=intent,
                tx_hash=receipt.tx_hash,
                via="fallback_event",
                block_number=receipt.block_number,
            )

        # dry_run
        receipt = self._c.send(
            make_mock_call("submitIntent", (agent_id, intent.intent_hash))
        )
        return VaultReceipt(intent=intent, tx_hash=receipt.tx_hash, via="dry_run")


# --------------------------------------------------------------- helper API

def from_env() -> VaultRouter:
    """Convenience constructor mirroring identity/artifacts."""
    return VaultRouter(ChainClient(ChainConfig.from_env()))
