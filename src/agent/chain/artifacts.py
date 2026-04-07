"""
ValidationArtifact submission — the heart of the ERC-8004 narrative.

For every executed trade we anchor a `ValidationArtifact` (decision hash,
risk checks, pre/post portfolio hashes, reasoning pointer) on-chain. Two
write paths, picked at boot:

1. **Canonical** — `ValidationRegistry.validationRequest(...)`. This is what
   the brief expects: the standard ERC-8004 surface.
2. **Wrapped** — our optional `AgentArtifacts.sol` that emits a single
   indexed event with the full artifact in one shot. Used when its address
   is configured. Cheaper to query in the demo (no subgraph needed).

Both paths are wrapped behind `submit()` so the trading loop is agnostic.

Atomicity contract:
    The trading loop calls `submit(artifact)` *after* a successful exec
    fill but inside the same try-block. If submission raises, the caller
    marks the trade `unverified` and retries on the next loop tick. We
    deliberately do not retry inside `submit()` itself — keeping retry
    policy in one place (the loop) is easier to reason about than nested
    backoff.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.state import ValidationArtifact, canonical_hash

from ._client import ChainClient, ChainConfig, hex_to_bytes32, make_mock_call

# Standard ERC-8004 ValidationRegistry surface — we only need the write path.
VALIDATION_REGISTRY_ABI: list[dict] = [
    {
        "type": "function",
        "name": "validationRequest",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "validatorAddress", "type": "address"},
            {"name": "agentId", "type": "uint256"},
            {"name": "requestURI", "type": "string"},
            {"name": "requestHash", "type": "bytes32"},
        ],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "getValidationStatus",
        "stateMutability": "view",
        "inputs": [{"name": "requestHash", "type": "bytes32"}],
        "outputs": [
            {"name": "validatorAddress", "type": "address"},
            {"name": "agentId", "type": "uint256"},
            {"name": "response", "type": "uint8"},
            {"name": "responseHash", "type": "bytes32"},
            {"name": "tag", "type": "string"},
            {"name": "lastUpdate", "type": "uint256"},
        ],
    },
]

# Our thin wrapper. The Solidity is in contracts/src/AgentArtifacts.sol.
AGENT_ARTIFACTS_ABI: list[dict] = [
    {
        "type": "function",
        "name": "submit",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "decisionHash", "type": "bytes32"},
            {"name": "tradeHash", "type": "bytes32"},
            {"name": "preStateHash", "type": "bytes32"},
            {"name": "postStateHash", "type": "bytes32"},
            {"name": "reasoningURI", "type": "string"},
        ],
        "outputs": [],
    },
    {
        "type": "event",
        "name": "ArtifactSubmitted",
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "agentId", "type": "uint256"},
            {"indexed": True, "name": "decisionHash", "type": "bytes32"},
            {"indexed": False, "name": "tradeHash", "type": "bytes32"},
            {"indexed": False, "name": "preStateHash", "type": "bytes32"},
            {"indexed": False, "name": "postStateHash", "type": "bytes32"},
            {"indexed": False, "name": "reasoningURI", "type": "string"},
        ],
    },
]

# Sentinel used when an artifact has no associated trade (e.g. HOLD decisions
# the agent still wants to anchor for completeness).
ZERO_HASH = "0x" + "0" * 64


@dataclass(frozen=True)
class OnChainArtifactRef:
    """
    A reference to one artifact as recorded on-chain. The verifier joins
    these against off-chain JSON to prove the agent's history is intact.
    """

    artifact_hash: str         # bytes32, 0x-prefixed lowercase hex
    tx_hash: str | None        # available when sourced from event logs
    block_number: int | None
    source: str                # "agent_artifacts" | "validation_registry"


@dataclass
class ArtifactSubmission:
    """Result returned by `submit()`."""

    artifact: ValidationArtifact
    artifact_hash: str       # canonical_hash(artifact) — the on-chain `requestHash`
    tx_hash: str
    via: str                 # "validation_registry" | "agent_artifacts" | "dry_run"
    block_number: int | None = None


class ArtifactsClient:
    """Submits ValidationArtifacts via the ERC-8004 standard or our wrapper."""

    def __init__(self, client: ChainClient, agent_id: int):
        if agent_id <= 0:
            raise ValueError(f"agent_id must be positive, got {agent_id}")
        self._c = client
        self._agent_id = agent_id
        self._validation_contract = None
        self._wrapper_contract = None

        if client.config.dry_run:
            return

        # Prefer the wrapper if its address is configured — single tx, single
        # event, easy to filter for the demo. Fall back to the standard
        # ValidationRegistry otherwise.
        if client.config.agent_artifacts:
            self._wrapper_contract = client.load_contract(
                client.config.agent_artifacts, AGENT_ARTIFACTS_ABI
            )
        if client.config.validation_registry:
            self._validation_contract = client.load_contract(
                client.config.validation_registry, VALIDATION_REGISTRY_ABI
            )
        if self._wrapper_contract is None and self._validation_contract is None:
            raise RuntimeError(
                "ArtifactsClient: no on-chain target configured. "
                "Set ERC8004_VALIDATION_REGISTRY or AGENT_ARTIFACTS_ADDRESS."
            )

    @property
    def agent_id(self) -> int:
        return self._agent_id

    # ----------------------------------------------------------------- submit
    def submit(self, artifact: ValidationArtifact) -> ArtifactSubmission:
        """
        Anchor a single ValidationArtifact on-chain. Raises on RPC errors so
        the trading loop can mark the trade unverified and retry.
        """
        # The on-chain `requestHash` is the canonical hash of the artifact
        # itself. This is what makes the artifact independently verifiable —
        # anyone can re-derive it from the off-chain JSON we publish.
        artifact_hash = canonical_hash(artifact)

        # Sanity check the hashes the artifact carries — bad inputs will only
        # surface inside the ABI encoder otherwise, with terrible messages.
        for label, value in (
            ("decision_hash", artifact.decision_hash),
            ("pre_state_hash", artifact.pre_state_hash),
            ("post_state_hash", artifact.post_state_hash),
        ):
            hex_to_bytes32(value)  # raises ValueError on bad shape
        if artifact.trade_hash is not None:
            hex_to_bytes32(artifact.trade_hash)

        if self._wrapper_contract is not None:
            return self._submit_via_wrapper(artifact, artifact_hash)
        if self._validation_contract is not None:
            return self._submit_via_registry(artifact, artifact_hash)
        return self._submit_dry_run(artifact, artifact_hash)

    # ----------------------------------------------------------------- paths
    def _submit_via_wrapper(
        self, artifact: ValidationArtifact, artifact_hash: str
    ) -> ArtifactSubmission:
        fn = self._wrapper_contract.functions.submit(
            self._agent_id,
            hex_to_bytes32(artifact.decision_hash),
            hex_to_bytes32(artifact.trade_hash or ZERO_HASH),
            hex_to_bytes32(artifact.pre_state_hash),
            hex_to_bytes32(artifact.post_state_hash),
            artifact.reasoning_uri,
        )
        receipt = self._c.send(fn, gas=300_000)
        return ArtifactSubmission(
            artifact=artifact,
            artifact_hash=artifact_hash,
            tx_hash=receipt.tx_hash,
            via="agent_artifacts",
            block_number=receipt.block_number,
        )

    def _submit_via_registry(
        self, artifact: ValidationArtifact, artifact_hash: str
    ) -> ArtifactSubmission:
        # Self-validation: we are our own validator for the hackathon. The
        # reputation track scores us on objective trade outcomes (PnL,
        # drawdown), not on third-party signatures, so this is in spec.
        validator = self._c.address
        fn = self._validation_contract.functions.validationRequest(
            validator,
            self._agent_id,
            artifact.reasoning_uri,
            hex_to_bytes32(artifact_hash),
        )
        receipt = self._c.send(fn, gas=250_000)
        return ArtifactSubmission(
            artifact=artifact,
            artifact_hash=artifact_hash,
            tx_hash=receipt.tx_hash,
            via="validation_registry",
            block_number=receipt.block_number,
        )

    # ---------------------------------------------------------------- fetch
    def fetch_history(self, from_block: int = 0) -> list[OnChainArtifactRef]:
        """
        Pull every artifact this agent has ever anchored.

        Two paths:
        * `agent_artifacts` wrapper deployed → cheap `eth_getLogs` filter on
          the indexed `(agentId, decisionHash)` event topics. Returns
          tx_hash + block_number for each.
        * Otherwise → `ValidationRegistry.getAgentValidations(agentId)` view
          call. This returns only the request hashes; tx_hash is unknown
          unless the caller pairs them with their own write log.

        Dry-run returns an empty list — there is nothing to fetch and we
        deliberately do not invent fixture data here (that lives in tests).
        """
        if self._c.config.dry_run:
            return []

        if self._wrapper_contract is not None:
            return self._fetch_from_wrapper(from_block)
        if self._validation_contract is not None:
            return self._fetch_from_registry()
        return []

    def _fetch_from_wrapper(self, from_block: int) -> list[OnChainArtifactRef]:
        # `eth_getLogs` filtered on the indexed agentId topic. We compute
        # the topic hash via web3's contract event helper instead of by hand.
        event = self._wrapper_contract.events.ArtifactSubmitted
        logs = event.get_logs(
            from_block=from_block,
            argument_filters={"agentId": self._agent_id},
        )
        out: list[OnChainArtifactRef] = []
        for log in logs:
            # `decisionHash` is the indexed component carried as a bytes32
            # in the log topic. Web3 hands it back as `bytes`; convert.
            raw = log["args"]["decisionHash"]
            artifact_hash = "0x" + (raw.hex() if isinstance(raw, (bytes, bytearray)) else raw)
            out.append(
                OnChainArtifactRef(
                    artifact_hash=artifact_hash.lower(),
                    tx_hash=log["transactionHash"].hex(),
                    block_number=log["blockNumber"],
                    source="agent_artifacts",
                )
            )
        return out

    def _fetch_from_registry(self) -> list[OnChainArtifactRef]:
        fn = self._validation_contract.functions.getAgentValidations(self._agent_id)
        request_hashes = self._c.call(fn) or []
        out: list[OnChainArtifactRef] = []
        for raw in request_hashes:
            artifact_hash = "0x" + (
                raw.hex() if isinstance(raw, (bytes, bytearray)) else str(raw).lstrip("0x")
            )
            out.append(
                OnChainArtifactRef(
                    artifact_hash=artifact_hash.lower(),
                    tx_hash=None,
                    block_number=None,
                    source="validation_registry",
                )
            )
        return out

    def _submit_dry_run(
        self, artifact: ValidationArtifact, artifact_hash: str
    ) -> ArtifactSubmission:
        receipt = self._c.send(
            make_mock_call("submitArtifact", (self._agent_id, artifact_hash))
        )
        return ArtifactSubmission(
            artifact=artifact,
            artifact_hash=artifact_hash,
            tx_hash=receipt.tx_hash,
            via="dry_run",
        )


# --------------------------------------------------------------- helper API

def from_env(agent_id: int) -> ArtifactsClient:
    """Convenience constructor mirroring `identity.from_env()`."""
    return ArtifactsClient(ChainClient(ChainConfig.from_env()), agent_id)
