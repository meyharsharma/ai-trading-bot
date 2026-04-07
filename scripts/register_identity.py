"""
One-shot ERC-8004 identity registration.

Usage:
    python scripts/register_identity.py
    python scripts/register_identity.py --force         # mint a new identity even if one exists
    python scripts/register_identity.py --dry-run       # don't touch the chain (CI)

The agent_id is persisted to `config/agent_id.json` and the trading loop
loads it from there. Re-running this script is a no-op unless `--force` is
passed — re-registering would orphan all prior validation artifacts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python scripts/register_identity.py` to work without `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.chain import (  # noqa: E402  (sys.path setup must precede import)
    AGENT_ID_PATH,
    AgentManifest,
    ChainClient,
    ChainConfig,
    IdentityClient,
    load_persisted_id,
    persist_identity,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register the trading agent on ERC-8004.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Mint a new identity even if config/agent_id.json already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip RPC calls and produce a deterministic mock identity (for CI).",
    )
    p.add_argument(
        "--name",
        default=os.environ.get("AGENT_NAME", "ai-trading-agent"),
        help="Human-readable agent name (goes into the off-chain manifest).",
    )
    p.add_argument(
        "--repo",
        default=os.environ.get(
            "AGENT_REPO_URL", "https://github.com/REPLACE_ME/ai-trading-agent"
        ),
        help="Public repo URL for the manifest.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("AGENT_MODEL", "claude-opus-4-6"),
        help="LLM model name embedded in the manifest.",
    )
    p.add_argument(
        "--strategy",
        default=(
            "LLM-driven momentum + mean-reversion on BTC/USD and ETH/USD with "
            "deterministic 2% per-trade risk gate. Every decision is anchored "
            "on-chain as an ERC-8004 validation artifact."
        ),
        help="One-paragraph strategy summary for the manifest.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    existing = load_persisted_id()
    if existing and not args.force:
        print(f"Agent already registered: id={existing.agent_id}")
        print(f"  tx:        {existing.tx_hash}")
        print(f"  registry:  {existing.registry_address}")
        print(f"  chain_id:  {existing.chain_id}")
        print("  (pass --force to re-register; this orphans prior artifacts)")
        return 0

    config = ChainConfig.from_env()
    if args.dry_run:
        # Force dry-run independent of env. We rebuild the dataclass because
        # it's frozen.
        config = ChainConfig(**{**config.__dict__, "dry_run": True})

    if config.dry_run:
        print("[dry-run] no RPC will be called; mock receipt will be persisted")
    else:
        print(f"network:  {config.rpc_url} (chain_id={config.chain_id})")
        print(f"registry: {config.identity_registry}")

    client = ChainClient(config)
    identity = IdentityClient(client)

    manifest = AgentManifest(
        name=args.name,
        owner_address=client.address,
        model=args.model,
        repo_url=args.repo,
        strategy_summary=args.strategy,
    )

    print(f"submitting register() as {client.address} ...")
    record = identity.register(manifest)

    persist_identity(record)
    print("OK")
    print(f"  agent_id:  {record.agent_id}")
    print(f"  tx_hash:   {record.tx_hash}")
    print(f"  saved to:  {AGENT_ID_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
