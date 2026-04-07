"""
Kraken MCP healthcheck CLI.

    python -m agent.kraken_mcp.healthcheck [--symbol BTC/USD] [--timeout 10]

Pings the Kraken CLI MCP server, lists advertised tools, validates required
env vars, and samples a real ticker call. Exits non-zero on any failure so
it's drop-in for cron / pm2 watchdogs during the live trading phase.

Exit codes
----------
  0  healthy
  2  missing required environment variable
  3  could not connect / initialize the MCP server
  4  connected but a sampled tool call failed
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Awaitable, Callable

from agent.kraken_mcp.client import KrakenMCPClient, KrakenMCPConfig, KrakenMCPError
from agent.kraken_mcp.tools import KrakenTools


# The Kraken CLI itself is responsible for reading these — we just verify
# they're present so a misconfigured environment fails loudly here rather
# than mid-trade.
REQUIRED_ENV: tuple[str, ...] = ("KRAKEN_API_KEY",)


def check_env(required: tuple[str, ...] = REQUIRED_ENV) -> list[str]:
    """Return the list of required env vars that are missing/empty."""
    return [k for k in required if not os.getenv(k)]


async def run_health_check(
    *,
    symbol: str = "BTC/USD",
    timeout: float = 10.0,
    client_factory: Callable[[], KrakenMCPClient] | None = None,
    out=print,
) -> int:
    """The healthcheck pipeline. Returns the process exit code."""
    out("[1/4] checking environment ...")
    missing = check_env()
    if missing:
        out(f"  FAIL: missing env vars: {missing}")
        return 2
    out("  OK")

    factory = client_factory or (lambda: KrakenMCPClient())

    out("[2/4] connecting to Kraken CLI MCP server ...")
    try:
        return await asyncio.wait_for(_probe(symbol, factory, out), timeout=timeout)
    except asyncio.TimeoutError:
        out(f"  FAIL: timed out after {timeout}s")
        return 3
    except KrakenMCPError as exc:
        out(f"  FAIL: {exc}")
        return 3
    except Exception as exc:  # noqa: BLE001 — last-resort guard for cron
        out(f"  FAIL: unexpected {type(exc).__name__}: {exc}")
        return 3


async def _probe(
    symbol: str,
    factory: Callable[[], KrakenMCPClient],
    out: Callable[[str], None],
) -> int:
    async with factory() as client:
        out(f"  OK (binary={KrakenMCPConfig().binary})")

        out("[3/4] listing advertised tools ...")
        names = sorted(client.tool_names)
        if not names:
            out("  WARN: server did not advertise any tools (list_tools empty)")
        else:
            out(f"  OK ({len(names)} tools)")
            for n in names:
                out(f"    - {n}")

        out(f"[4/4] sampling get_ticker {symbol} ...")
        tools = KrakenTools(client)
        try:
            ticker = await tools.get_ticker(symbol)
        except KrakenMCPError as exc:
            out(f"  FAIL: {exc}")
            return 4
        if not ticker.get("last"):
            out(f"  FAIL: ticker payload missing prices: {ticker}")
            return 4
        out(
            f"  OK last={ticker['last']} bid={ticker.get('bid')} "
            f"ask={ticker.get('ask')}"
        )

    out("\nhealthcheck PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent.kraken_mcp.healthcheck")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    sys.exit(asyncio.run(run_health_check(symbol=args.symbol, timeout=args.timeout)))


if __name__ == "__main__":
    main()
