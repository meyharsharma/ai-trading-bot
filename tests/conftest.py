"""
Shared fixtures.

The most interesting thing here is `anvil_chain_client` — a fixture that
yields a real `ChainClient` connected to a locally running anvil fork. Any
test marked `@pytest.mark.integration` and asking for this fixture is
auto-skipped if anvil isn't running, so the default `pytest` invocation
stays hermetic and fast.

To run integration tests:
    ./scripts/dev_anvil.sh &
    PYTHONPATH=src uv run pytest -m integration -v
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

# Make `agent.*` importable without depending on an editable install. Mirrors
# the `PYTHONPATH=src` invocation the existing tests already require.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# anvil's deterministic prefunded account #0. Public, well-known, never
# valuable on a real network. Hard-coded so the integration tests are
# zero-config: spin up anvil, run pytest.
ANVIL_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
ANVIL_DEFAULT_URL = "http://127.0.0.1:8545"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that require a running anvil fork (skipped by default)",
    )


def _anvil_reachable(url: str, timeout: float = 0.5) -> bool:
    """Cheap TCP probe so we don't pay a JSON-RPC round trip just to skip."""
    try:
        host_port = url.split("://", 1)[1].split("/", 1)[0]
        host, port = host_port.split(":")
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


@pytest.fixture(scope="session")
def anvil_url() -> str:
    """The URL the anvil fork is expected on. Override via $ANVIL_URL."""
    return os.environ.get("ANVIL_URL", ANVIL_DEFAULT_URL)


@pytest.fixture(scope="session")
def anvil_chain_client(anvil_url: str):
    """
    Live `ChainClient` against the local anvil fork. Skips the test if
    anvil isn't running so CI without Foundry installed stays green.
    """
    if not _anvil_reachable(anvil_url):
        pytest.skip(f"anvil not reachable at {anvil_url} — start ./scripts/dev_anvil.sh")

    try:
        import web3  # noqa: F401  (imported to check the dep is available)
    except ImportError:
        pytest.skip("web3 not installed")

    from agent.chain import ChainClient, ChainConfig

    config = ChainConfig(
        rpc_url=anvil_url,
        private_key=ANVIL_PRIVATE_KEY,
        chain_id=int(os.environ.get("CHAIN_ID", "84532")),
        # Force the wrapper path off — integration test deploys its own
        # mock target if it needs one. The verifier-flavored tests use the
        # canonical Base Sepolia addresses that the fork already shadows.
        agent_artifacts="",
        dry_run=False,
    )
    client = ChainClient(config)
    yield client
