"""
MCP client for the Kraken CLI Rust binary's built-in MCP server.

The Kraken CLI ships with a stdio-based MCP server. We treat it as a black box
and route every Kraken interaction (data + execution) through it. This file is
the only place that imports the `mcp` SDK; everything else in the agent calls
into `KrakenTools` (see ``tools.py``).
"""
from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class KrakenMCPError(RuntimeError):
    """Raised when the Kraken CLI MCP server returns an error or bad payload."""


@dataclass
class KrakenMCPConfig:
    """How to launch the Kraken CLI MCP server.

    Defaults assume the CLI binary is on $PATH and accepts an ``mcp`` subcommand
    that starts the stdio server. Override via env or by passing a custom
    config in tests / alternative deployments.
    """
    binary: str = field(default_factory=lambda: os.getenv("KRAKEN_CLI_BIN", "kraken"))
    args: list[str] = field(
        default_factory=lambda: os.getenv("KRAKEN_CLI_MCP_ARGS", "mcp").split()
    )
    env: dict[str, str] | None = None


class KrakenMCPClient:
    """Async context manager around an `mcp.ClientSession` over stdio.

    Usage::

        async with KrakenMCPClient() as client:
            payload = await client.call_tool("get_ticker", {"pair": "XBTUSD"})
    """

    def __init__(self, config: KrakenMCPConfig | None = None):
        self._config = config or KrakenMCPConfig()
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._tool_names: set[str] = set()

    async def __aenter__(self) -> "KrakenMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def connect(self) -> None:
        if self._session is not None:
            return
        params = StdioServerParameters(
            command=self._config.binary,
            args=list(self._config.args),
            env={**os.environ, **(self._config.env or {})},
        )
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except Exception as exc:
            await self._stack.aclose()
            self._stack = None
            self._session = None
            raise KrakenMCPError(f"failed to start Kraken CLI MCP server: {exc}") from exc

        try:
            listed = await self._session.list_tools()
            self._tool_names = {t.name for t in listed.tools}
        except Exception:
            # Some MCP servers don't implement list_tools; tool resolution
            # falls back to first-candidate-wins in that case.
            self._tool_names = set()

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    @property
    def tool_names(self) -> set[str]:
        return set(self._tool_names)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._session is None:
            raise KrakenMCPError(
                "KrakenMCPClient is not connected; use `async with` or call connect()"
            )
        result = await self._session.call_tool(name, arguments or {})
        if getattr(result, "isError", False):
            raise KrakenMCPError(f"tool {name!r} failed: {result}")
        return _extract_payload(result)


def _extract_payload(result: Any) -> Any:
    """MCP CallToolResult.content is a list of content blocks. Try JSON first.

    The Kraken CLI MCP server returns text blocks containing JSON; older MCP
    servers occasionally return plain text. We accept both.
    """
    contents = getattr(result, "content", None) or []
    texts: list[str] = []
    for block in contents:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    if not texts:
        return None
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except (ValueError, TypeError):
        return joined
