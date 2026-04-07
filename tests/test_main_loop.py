"""
End-to-end smoke test for the integration loop.

Wires the real risk gate, real paper exec, real store, and real chain
ArtifactsClient (in dry-run mode) — only the *outside-world* edges
(Kraken MCP, Anthropic SDK, web3 RPC) are faked. If this test passes, the
four layers fit together and `python -m agent.main --once` works in
dry-run mode without secrets.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import agent.main as main_mod
from agent.chain import ArtifactsClient, ChainClient, ChainConfig, VaultRouter
from agent.exec import PaperConfig, PaperExecutionAdapter
from agent.kraken_mcp.tools import OHLCVBar
from agent.risk.gate import RiskConfig, RiskGate
from agent.state import Store


# ---------------- fakes ----------------

class FakeKrakenTools:
    """In-memory replacement for KrakenTools. Drives a deterministic uptrend
    so the brain has a clear BUY signal."""

    def __init__(self, base_price: float = 60_000.0):
        self.base = base_price

    async def get_ohlcv(self, symbol: str, interval: str = "5m", limit: int = 200) -> list[OHLCVBar]:
        bars: list[OHLCVBar] = []
        now = datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(limit):
            ts = now - timedelta(minutes=(limit - i) * 5)
            # Strong, monotonic uptrend → fast MA > slow MA, RSI mid-range.
            close = self.base + i * 50.0
            bars.append(OHLCVBar(
                timestamp=ts,
                open=close - 5,
                high=close + 10,
                low=close - 10,
                close=close,
                volume=1.0,
            ))
        return bars

    async def get_ticker(self, symbol: str) -> dict[str, float]:
        last = self.base + 200 * 50.0
        return {"bid": last - 5, "ask": last + 5, "last": last, "mid": last}


class FakeAnthropicResponse:
    """Mimics the anthropic SDK response shape with one tool_use block."""

    def __init__(self, tool_input: dict[str, Any], tool_name: str):
        self.content = [_FakeBlock(tool_input, tool_name)]


class _FakeBlock:
    def __init__(self, tool_input: dict[str, Any], tool_name: str):
        self.type = "tool_use"
        self.name = tool_name
        self.input = tool_input


class FakeAnthropic:
    """Forces a BUY decision then ACCEPT critique."""

    def __init__(self):
        self._calls = 0
        self.messages = self  # SDK calls Anthropic().messages.create(...)

    def create(self, **kwargs: Any) -> FakeAnthropicResponse:
        self._calls += 1
        tool_name = kwargs["tools"][0]["name"]
        if tool_name == "submit_decision":
            return FakeAnthropicResponse(
                tool_input={
                    "symbol": "BTC/USD",
                    "action": "BUY",
                    "size_pct": 0.05,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06,
                    "reasoning": "Strong uptrend with MA fast > MA slow.",
                    "signals": {"ma_fast": 65000.0, "ma_slow": 64000.0, "rsi": 58.0},
                },
                tool_name=tool_name,
            )
        # critique tool
        return FakeAnthropicResponse(
            tool_input={"verdict": "ACCEPT", "weakness": "thesis is sound"},
            tool_name=tool_name,
        )


# ---------------- runtime builder ----------------

def _build_runtime(tmp_path: Path) -> main_mod.AgentRuntime:
    """Hand-roll an AgentRuntime that uses real layers + fake externals.

    We deliberately *don't* call `build_runtime()` because that needs a real
    Kraken CLI binary. The point of this test is to exercise the loop body
    against the real risk/exec/chain/store wiring.
    """
    from agent.brain.llm_strategist import LLMStrategist, SelfCritiquingStrategist
    from agent.data import KrakenFeed

    cfg = main_mod.LoopConfig(
        symbols=["BTC/USD"],
        loop_interval_seconds=0.1,
        candle_interval="5m",
        candle_lookback=200,
        intervals_multiframe=("5m",),
        risk=RiskConfig(
            max_risk_per_trade_pct=0.02,
            max_open_positions=3,
            max_position_size_pct=0.25,
            default_stop_loss_pct=0.03,
            max_stop_loss_pct=0.08,
            max_take_profit_pct=0.20,
            allow_leverage=False,
            allow_shorts=False,
        ),
        llm_model="claude-opus-4-6",
        llm_max_tokens=1024,
        llm_temperature=0.2,
        starting_capital_usd=1000.0,
        fee_bps=26.0,
        mode="paper",
        chain_enabled=True,
    )
    tools = FakeKrakenTools()
    feed = KrakenFeed(tools)
    strategist = SelfCritiquingStrategist(LLMStrategist(FakeAnthropic()))
    risk_gate = RiskGate(cfg.risk)
    exec_adapter = PaperExecutionAdapter(
        quote_fn=feed.get_quote,
        config=PaperConfig(starting_capital_usd=1000.0, fee_bps=26.0),
    )
    chain_client = ChainClient(ChainConfig())  # dry-run by default
    artifacts = ArtifactsClient(chain_client, agent_id=1)
    vault = None  # vault requires a configured fallback wrapper; skip for the smoke test
    store = Store(tmp_path / "state.sqlite")

    class _StubMCPClient:
        async def aclose(self) -> None:
            pass

    return main_mod.AgentRuntime(
        cfg=cfg,
        feed=feed,
        strategist=strategist,
        risk_gate=risk_gate,
        exec_adapter=exec_adapter,
        artifacts=artifacts,
        vault=vault,
        store=store,
        agent_id=1,
        mcp_client=_StubMCPClient(),  # type: ignore[arg-type]
    )


# ---------------- tests ----------------

@pytest.mark.asyncio
async def test_one_cycle_buy_path(tmp_path: Path):
    rt = _build_runtime(tmp_path)
    try:
        await main_mod.run_one_cycle(rt)
        # Brain emitted BUY → exec filled → artifact anchored.
        assert rt.store.count("cycles") == 1
        assert rt.store.count("decisions") == 1
        assert rt.store.count("fills") == 1
        assert rt.store.count("artifacts") == 1

        # The artifact should be 'ok' (dry-run chain returns a deterministic hash).
        rows = rt.store._conn.execute(
            "SELECT status, tx_hash, via FROM artifacts"
        ).fetchall()
        assert rows[0]["status"] == "ok"
        assert rows[0]["tx_hash"].startswith("0x")
        assert rows[0]["via"] == "dry_run"
    finally:
        await rt.aclose()


@pytest.mark.asyncio
async def test_decision_hash_in_db_matches_canonical_hash(tmp_path: Path):
    """The verifier story relies on this round-trip."""
    from agent.state import Decision, canonical_hash

    rt = _build_runtime(tmp_path)
    try:
        await main_mod.run_one_cycle(rt)
        row = rt.store._conn.execute(
            "SELECT * FROM decisions LIMIT 1"
        ).fetchone()
        # Reconstruct the decision and re-hash it.
        d = Decision(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            symbol=row["symbol"],
            action=row["action"],
            size_pct=row["size_pct"],
            stop_loss_pct=row["stop_loss_pct"],
            take_profit_pct=row["take_profit_pct"],
            reasoning=row["reasoning"],
            signals={k: float(v) for k, v in __import__("json").loads(row["signals_json"]).items()},
            model=row["model"],
        )
        assert canonical_hash(d) == row["decision_hash"]
    finally:
        await rt.aclose()


@pytest.mark.asyncio
async def test_loop_continues_when_exec_fails(tmp_path: Path):
    """If the exec layer raises, the cycle records the decision and moves on."""
    rt = _build_runtime(tmp_path)
    try:
        async def boom(_order):
            raise RuntimeError("simulated venue outage")

        with patch.object(rt.exec_adapter, "submit_order", side_effect=boom):
            await main_mod.run_one_cycle(rt)

        assert rt.store.count("decisions") == 1
        assert rt.store.count("fills") == 0
        assert rt.store.count("artifacts") == 0
        # The cycle row finishes with status='ok' — per-symbol failures are
        # contained, only top-level loop crashes flip the cycle to 'error'.
        cycle = rt.store._conn.execute("SELECT status FROM cycles").fetchone()
        assert cycle["status"] == "ok"
    finally:
        await rt.aclose()
