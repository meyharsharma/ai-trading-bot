#!/usr/bin/env bash
# Spin up a local anvil node forked off Base Sepolia.
#
# Why fork instead of plain anvil? Because the ERC-8004 reference contracts
# are *already deployed* on Base Sepolia at the canonical addresses (see
# docs/ERC8004_NOTES.md). Forking lets our integration tests interact with
# the real contracts without burning testnet gas — anvil shadows state
# locally and accepts our writes against forked contract bytecode.
#
# Usage:
#   ./scripts/dev_anvil.sh                    # default port 8545
#   ./scripts/dev_anvil.sh --port 8546        # custom port
#
# Then in another shell:
#   export RPC_URL=http://127.0.0.1:8545
#   export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
#   uv run pytest tests/test_chain_anvil.py -m integration -v
#
# That key is anvil's deterministic prefunded account #0 — fine for local
# tests, NEVER use it on a real network.

set -euo pipefail

if ! command -v anvil >/dev/null 2>&1; then
    echo "error: anvil not found. Install Foundry: https://getfoundry.sh" >&2
    exit 1
fi

FORK_URL="${BASE_SEPOLIA_RPC:-https://sepolia.base.org}"
CHAIN_ID="${CHAIN_ID:-84532}"

echo "anvil: forking $FORK_URL (chain_id=$CHAIN_ID)"
exec anvil \
    --fork-url "$FORK_URL" \
    --chain-id "$CHAIN_ID" \
    --block-time 2 \
    "$@"
