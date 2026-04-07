// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title  AgentArtifacts
/// @notice Optional thin wrapper around the ERC-8004 ValidationRegistry that
///         emits a single indexed event per validation artifact. Lets the
///         demo UI / explorer query an agent's artifact history with a plain
///         `eth_getLogs` filter — no subgraph required.
///
/// @dev    This is *additive*: the agent still writes to the canonical
///         ValidationRegistry. This contract exists purely so judges can scan
///         on-chain history in <30 seconds during the demo.
contract AgentArtifacts {
    /// @notice Emitted for every validation artifact the agent anchors.
    /// @param  agentId        ERC-8004 identity registry token id.
    /// @param  decisionHash   sha256 of the canonical Decision JSON.
    /// @param  tradeHash      sha256 of the Fill JSON, or 0x0 for HOLD/no-trade.
    /// @param  preStateHash   portfolio hash before the trade.
    /// @param  postStateHash  portfolio hash after the trade.
    /// @param  reasoningURI   pointer to the off-chain natural-language rationale.
    event ArtifactSubmitted(
        uint256 indexed agentId,
        bytes32 indexed decisionHash,
        bytes32 tradeHash,
        bytes32 preStateHash,
        bytes32 postStateHash,
        string reasoningURI
    );

    /// @notice The wallet allowed to submit artifacts. Set once at deploy.
    /// @dev    We restrict to a single submitter so the on-chain history is
    ///         provably one agent's. Transferable via `setSubmitter` if the
    ///         agent's signing key needs to rotate.
    address public submitter;

    /// @notice Monotonic per-agent counter — useful for ordering artifacts
    ///         without relying on block timestamps.
    mapping(uint256 => uint256) public artifactCount;

    error NotSubmitter();
    error ZeroAddress();

    constructor(address initialSubmitter) {
        if (initialSubmitter == address(0)) revert ZeroAddress();
        submitter = initialSubmitter;
    }

    modifier onlySubmitter() {
        if (msg.sender != submitter) revert NotSubmitter();
        _;
    }

    /// @notice Anchor one validation artifact. The contract stores nothing
    ///         beyond the counter — all data lives in the indexed event log.
    function submit(
        uint256 agentId,
        bytes32 decisionHash,
        bytes32 tradeHash,
        bytes32 preStateHash,
        bytes32 postStateHash,
        string calldata reasoningURI
    ) external onlySubmitter {
        unchecked {
            // Counter is bounded by tx count; overflow is unreachable.
            artifactCount[agentId] += 1;
        }
        emit ArtifactSubmitted(
            agentId,
            decisionHash,
            tradeHash,
            preStateHash,
            postStateHash,
            reasoningURI
        );
    }

    /// @notice Rotate the submitter (e.g. when the agent's signing key changes).
    function setSubmitter(address newSubmitter) external onlySubmitter {
        if (newSubmitter == address(0)) revert ZeroAddress();
        submitter = newSubmitter;
    }
}
