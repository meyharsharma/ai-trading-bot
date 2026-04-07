"""Tests for the Kraken MCP healthcheck CLI.

The CLI's MCP-server interaction is mocked via a fake client factory; we
verify the exit-code matrix the watchdog will key off.
"""
from __future__ import annotations

import os

import pytest

from agent.kraken_mcp import healthcheck


# ---------------- env validation ----------------

def test_check_env_reports_missing(monkeypatch):
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    assert healthcheck.check_env() == ["KRAKEN_API_KEY"]


def test_check_env_passes_when_set(monkeypatch):
    monkeypatch.setenv("KRAKEN_API_KEY", "test-key")
    assert healthcheck.check_env() == []


# ---------------- end-to-end probe via fake client ----------------

class FakeSession:
    def __init__(self, ticker_payload):
        self._ticker = ticker_payload
        self.calls: list[tuple] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._ticker


class FakeClient:
    """Stand-in for KrakenMCPClient. Implements the surface healthcheck uses."""

    def __init__(self, *, tool_names, ticker_payload, fail_connect=False):
        self._tool_names = set(tool_names)
        self._ticker_payload = ticker_payload
        self._fail_connect = fail_connect
        self._session = None

    @property
    def tool_names(self):
        return self._tool_names

    async def call_tool(self, name, arguments=None):
        return self._ticker_payload

    async def __aenter__(self):
        if self._fail_connect:
            from agent.kraken_mcp.client import KrakenMCPError
            raise KrakenMCPError("simulated connect failure")
        return self

    async def __aexit__(self, *exc):
        return None


@pytest.fixture
def env_ok(monkeypatch):
    monkeypatch.setenv("KRAKEN_API_KEY", "test-key")


async def test_healthcheck_returns_2_when_env_missing(monkeypatch):
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    rc = await healthcheck.run_health_check(out=lambda *_: None)
    assert rc == 2


async def test_healthcheck_returns_3_on_connect_failure(env_ok):
    factory = lambda: FakeClient(
        tool_names=[], ticker_payload={}, fail_connect=True
    )
    rc = await healthcheck.run_health_check(
        client_factory=factory, out=lambda *_: None
    )
    assert rc == 3


async def test_healthcheck_returns_4_on_blank_ticker(env_ok):
    factory = lambda: FakeClient(
        tool_names={"get_ticker"},
        ticker_payload={"result": {"XXBTZUSD": {"a": ["0"], "b": ["0"], "c": ["0"]}}},
    )
    rc = await healthcheck.run_health_check(
        client_factory=factory, out=lambda *_: None
    )
    assert rc == 4


async def test_healthcheck_returns_0_on_success(env_ok):
    payload = {
        "result": {
            "XXBTZUSD": {
                "a": ["65010", "1", "1.0"],
                "b": ["64990", "1", "1.0"],
                "c": ["65000", "0.001"],
            }
        }
    }
    factory = lambda: FakeClient(
        tool_names={"get_ticker", "get_ohlcv"}, ticker_payload=payload
    )
    logs: list[str] = []
    rc = await healthcheck.run_health_check(
        client_factory=factory, out=logs.append
    )
    assert rc == 0
    joined = "\n".join(logs)
    assert "PASS" in joined
    assert "get_ticker" in joined  # advertised tool was listed


async def test_healthcheck_times_out(env_ok):
    import asyncio

    class HangingClient:
        @property
        def tool_names(self):
            return set()

        async def __aenter__(self):
            await asyncio.sleep(10)
            return self

        async def __aexit__(self, *exc):
            return None

    rc = await healthcheck.run_health_check(
        timeout=0.05, client_factory=lambda: HangingClient(), out=lambda *_: None
    )
    assert rc == 3
