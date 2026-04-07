"""On-chain layer: ERC-8004 identity, validation artifacts, vault, reputation, verification."""
from agent.chain._client import ChainClient, ChainConfig, TxReceipt
from agent.chain.artifacts import (
    ArtifactSubmission,
    ArtifactsClient,
    OnChainArtifactRef,
)
from agent.chain.explorer import (
    EXPLORERS,
    Explorer,
    address_url,
    agent_token_url,
    get_explorer,
    tx_url,
)
from agent.chain.identity import (
    AGENT_ID_PATH,
    AgentManifest,
    IdentityClient,
    IdentityRecord,
    load_persisted_id,
    persist_identity,
)
from agent.chain.reputation import ReputationScore, compute as compute_reputation
from agent.chain.vault import VaultIntent, VaultReceipt, VaultRouter
from agent.chain.verifier import (
    ArtifactCheck,
    LocalArtifactStore,
    VerificationReport,
    verify,
    verify_local_only,
)

__all__ = [
    "AGENT_ID_PATH",
    "AgentManifest",
    "ArtifactCheck",
    "ArtifactSubmission",
    "ArtifactsClient",
    "ChainClient",
    "ChainConfig",
    "EXPLORERS",
    "Explorer",
    "IdentityClient",
    "IdentityRecord",
    "LocalArtifactStore",
    "OnChainArtifactRef",
    "ReputationScore",
    "TxReceipt",
    "VaultIntent",
    "VaultReceipt",
    "VaultRouter",
    "VerificationReport",
    "address_url",
    "agent_token_url",
    "compute_reputation",
    "get_explorer",
    "load_persisted_id",
    "persist_identity",
    "tx_url",
    "verify",
    "verify_local_only",
]
