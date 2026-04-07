"""
The integration loop. Wires the four layers into a single agent.

    fetch  → brain → critique → risk gate → exec → on-chain artifact + vault
                                                  → SQLite persistence
                                                  → sleep

Run modes
---------
    python -m agent.main --once             one cycle, exit (smoke test / demo)
    python -m agent.main                    run forever (default loop interval)
    python -m agent.main --mode live        live Kraken (refuses without --i-know)

Design
------
* Every step lives in its own try/except so a single failure (LLM timeout,
  Kraken hiccup, on-chain revert) only kills the affected symbol's cycle. The
  loop continues with the next symbol on the next tick.
* Atomicity is recorded, not enforced: a fill that lands but fails to anchor
  on-chain is persisted with `status='unverified'`. The next tick retries
  pending submissions before doing new work.
* This file is the *only* place that imports a concrete adapter. Everything
  it touches above is the abstract `ExecutionAdapter`.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.brain.claude_code_strategist import (
    ClaudeCodeConfig,
    ClaudeCodeStrategist,
    is_available as claude_cli_available,
)
from agent.brain.llm_strategist import LLMStrategist, SelfCritiquingStrategist
from agent.chain import (
    ArtifactsClient,
    ChainClient,
    ChainConfig,
    VaultRouter,
    load_persisted_id,
)
from agent.data import KrakenFeed, MultiTimeframeSnapshot
from agent.exec import (
    ExecutionAdapter,
    KrakenLiveExecutionAdapter,
    PaperConfig,
    PaperExecutionAdapter,
)
from agent.kraken_mcp import KrakenMCPClient, KrakenTools
from agent.risk.gate import RiskConfig, RiskGate
from agent.state import (
    Decision,
    DecisionRow,
    Fill,
    FillRow,
    Order,
    PortfolioSnapshot,
    RiskedDecision,
    Store,
    ValidationArtifact,
    canonical_hash,
    snapshot_state_hash,
    utcnow,
)

log = logging.getLogger("agent.main")

DEFAULT_CONFIG_PATH = Path("config/strategy.yaml")
DEFAULT_DB_PATH = Path("state.sqlite")


# ============================================================ config dataclass

@dataclass(frozen=True)
class LoopConfig:
    symbols: list[str]
    loop_interval_seconds: float
    candle_interval: str
    candle_lookback: int
    intervals_multiframe: tuple[str, ...]
    risk: RiskConfig
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    starting_capital_usd: float
    fee_bps: float
    mode: str                            # 'paper' | 'live'
    chain_enabled: bool

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "LoopConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            symbols=list(raw["symbols"]),
            loop_interval_seconds=float(raw.get("loop_interval_seconds", 300)),
            candle_interval=str(raw.get("candle_interval", "5m")),
            candle_lookback=int(raw.get("candle_lookback", 200)),
            intervals_multiframe=tuple(
                raw.get("intervals_multiframe", ["1m", "5m", "1h"])
            ),
            risk=RiskConfig.from_dict(raw["risk"]),
            llm_model=str(raw["llm"]["model"]),
            llm_max_tokens=int(raw["llm"].get("max_tokens", 1024)),
            llm_temperature=float(raw["llm"].get("temperature", 0.2)),
            starting_capital_usd=float(raw["execution"].get("starting_capital_usd", 1000)),
            fee_bps=float(raw["execution"].get("fee_bps", 26)),
            mode=str(raw["execution"].get("mode", "paper")),
            chain_enabled=bool(raw.get("chain", {}).get("enabled", True)),
        )


# ============================================================ runtime container

@dataclass
class AgentRuntime:
    cfg: LoopConfig
    feed: KrakenFeed
    strategist: SelfCritiquingStrategist
    risk_gate: RiskGate
    exec_adapter: ExecutionAdapter
    artifacts: ArtifactsClient | None
    vault: VaultRouter | None
    store: Store
    agent_id: int
    mcp_client: KrakenMCPClient

    async def aclose(self) -> None:
        await self.exec_adapter.aclose()
        await self.mcp_client.aclose()
        self.store.close()


# ============================================================ wiring

async def build_runtime(
    *,
    cfg: LoopConfig,
    db_path: Path = DEFAULT_DB_PATH,
    anthropic_client: Any | None = None,
) -> AgentRuntime:
    """Wire every layer up. Stays in dry-run by default if env vars are absent."""

    # Kraken CLI MCP — the data + execution backend.
    mcp_client = KrakenMCPClient()
    await mcp_client.connect()
    tools = KrakenTools(mcp_client)
    feed = KrakenFeed(tools)

    # Brain — two paths:
    #   1. ANTHROPIC_API_KEY set → use the SDK directly (fastest, no subprocess
    #      overhead, useful for backtests and CI). The caller can also inject
    #      a fake client for tests.
    #   2. Otherwise, if the local `claude` CLI is on PATH → reuse the user's
    #      Claude Code OAuth credentials by shelling out per cycle. This is
    #      the default path during the hackathon: no API key, no billing setup.
    base: LLMStrategist | ClaudeCodeStrategist
    if anthropic_client is not None or os.getenv("ANTHROPIC_API_KEY"):
        if anthropic_client is None:
            from anthropic import Anthropic  # type: ignore

            anthropic_client = Anthropic()
        base = LLMStrategist(
            anthropic_client,
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            temperature=cfg.llm_temperature,
        )
    elif claude_cli_available():
        log.info("using local `claude` CLI as the strategist (no API key)")
        base = ClaudeCodeStrategist(
            ClaudeCodeConfig(model=cfg.llm_model)
        )
    else:
        raise RuntimeError(
            "no LLM brain available — set ANTHROPIC_API_KEY or install the "
            "`claude` CLI (https://docs.claude.com/claude-code)"
        )
    strategist = SelfCritiquingStrategist(base)

    # Risk gate — pure Python, no I/O.
    risk_gate = RiskGate(cfg.risk)

    # Execution — paper by default. Live requires explicit opt-in upstream.
    if cfg.mode == "live":
        exec_adapter: ExecutionAdapter = KrakenLiveExecutionAdapter(tools)
    else:
        exec_adapter = PaperExecutionAdapter(
            quote_fn=feed.get_quote,
            config=PaperConfig(
                starting_capital_usd=cfg.starting_capital_usd,
                fee_bps=cfg.fee_bps,
            ),
        )

    # Chain — only built if enabled. Falls back to dry-run when env unset.
    artifacts: ArtifactsClient | None = None
    vault: VaultRouter | None = None
    agent_id = 0
    if cfg.chain_enabled:
        chain_cfg = ChainConfig.from_env()
        chain_client = ChainClient(chain_cfg)
        agent_id = _resolve_agent_id(chain_cfg.dry_run)
        try:
            artifacts = ArtifactsClient(chain_client, agent_id=agent_id)
        except RuntimeError as exc:
            # No on-chain target configured at all — degrade gracefully.
            log.warning("ArtifactsClient unavailable: %s", exc)
            artifacts = None
        try:
            vault = VaultRouter(chain_client)
        except RuntimeError as exc:
            log.warning("VaultRouter unavailable: %s", exc)
            vault = None

    store = Store(db_path)

    return AgentRuntime(
        cfg=cfg,
        feed=feed,
        strategist=strategist,
        risk_gate=risk_gate,
        exec_adapter=exec_adapter,
        artifacts=artifacts,
        vault=vault,
        store=store,
        agent_id=agent_id,
        mcp_client=mcp_client,
    )


def _resolve_agent_id(dry_run: bool) -> int:
    """Load the persisted ERC-8004 agent_id, or fall back to 1 in dry-run."""
    record = load_persisted_id()
    if record is not None:
        return record.agent_id
    if dry_run:
        return 1
    raise RuntimeError(
        "no persisted agent_id found — run scripts/register_identity.py first"
    )


# ============================================================ one cycle

async def run_one_cycle(rt: AgentRuntime) -> None:
    """Run a single end-to-end cycle across every configured symbol."""
    cycle_id = rt.store.start_cycle(meta={"mode": rt.cfg.mode, "agent_id": rt.agent_id})
    cycle_error: str | None = None
    try:
        # Retry any artifact submissions left over from previous ticks.
        await _retry_unverified(rt)

        for symbol in rt.cfg.symbols:
            try:
                await _process_symbol(rt, cycle_id, symbol)
            except Exception as exc:  # noqa: BLE001
                log.exception("symbol %s failed: %s", symbol, exc)
    except Exception as exc:  # noqa: BLE001
        cycle_error = repr(exc)
        log.exception("cycle failed")
    finally:
        rt.store.finish_cycle(cycle_id, error=cycle_error)


async def _process_symbol(rt: AgentRuntime, cycle_id: int, symbol: str) -> None:
    # 1. Data
    snapshot = await rt.feed.get_multi_snapshot(
        symbol, intervals=rt.cfg.intervals_multiframe, lookback=rt.cfg.candle_lookback
    )
    portfolio_pre = await rt.exec_adapter.get_portfolio()

    # 2. Brain
    decision = rt.strategist.decide(
        symbol=symbol,                                  # type: ignore[arg-type]
        price=snapshot.quote.mid,
        signals=_signals_for_prompt(snapshot),
        portfolio=_portfolio_for_prompt(portfolio_pre),
    )

    # 3. Risk
    risked = rt.risk_gate.evaluate(decision, portfolio=portfolio_pre)

    # 4. Persist decision (always — HOLDs included).
    decision_row = rt.store.record_decision(cycle_id, risked)
    log.info(
        "decision id=%d %s %s size=%.4f passed=%s",
        decision_row.id,
        decision.symbol,
        decision.action,
        risked.final_size_pct,
        risked.passed,
    )

    if not risked.passed or decision.action == "HOLD" or risked.final_size_pct <= 0:
        return

    # 5. Exec
    quantity = _compute_quantity(
        size_pct=risked.final_size_pct,
        equity_usd=portfolio_pre.equity_usd,
        mid_price=snapshot.quote.mid,
    )
    if quantity <= 0:
        log.info("decision id=%d zero quantity after sizing", decision_row.id)
        return

    order = Order(
        symbol=symbol,                                  # type: ignore[arg-type]
        side="BUY" if decision.action == "BUY" else "SELL",
        quantity=quantity,
    )
    try:
        fill = await rt.exec_adapter.submit_order(order)
    except Exception as exc:  # noqa: BLE001
        log.warning("exec failed for decision %d: %s", decision_row.id, exc)
        return
    fill_row = rt.store.record_fill(decision_row.id, fill)
    log.info(
        "fill id=%d %s %s qty=%.6f @ %.2f fee=%.4f",
        fill_row.id, fill.order.symbol, fill.order.side,
        fill.order.quantity, fill.fill_price, fill.fee_usd,
    )

    # 6. Chain — atomic-ish: persist the artifact row first as 'unverified',
    #    then attempt submission and flip to 'ok' on success.
    portfolio_post = await rt.exec_adapter.get_portfolio()
    artifact = _build_artifact(
        rt=rt,
        decision=decision,
        fill=fill,
        risked=risked,
        portfolio_pre=portfolio_pre,
        portfolio_post=portfolio_post,
    )

    artifact_row = rt.store.record_artifact(
        decision_id=decision_row.id,
        fill_id=fill_row.id,
        artifact=artifact,
        artifact_hash=canonical_hash(artifact),
        tx_hash=None,
        via=None,
        block_number=None,
        status="unverified",
    )

    if rt.artifacts is None:
        log.info("chain disabled — skipping artifact submission")
        return

    try:
        submission = await asyncio.to_thread(rt.artifacts.submit, artifact)
        rt.store.mark_artifact_verified(
            artifact_row.id,
            tx_hash=submission.tx_hash,
            via=submission.via,
            block_number=submission.block_number,
        )
        log.info(
            "artifact id=%d anchored via=%s tx=%s",
            artifact_row.id, submission.via, submission.tx_hash[:14],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("artifact submission failed (will retry): %s", exc)
        # Stays as 'unverified' in the DB. _retry_unverified picks it up next tick.

    if rt.vault is not None:
        try:
            await asyncio.to_thread(rt.vault.route_intent, rt.agent_id, risked, fill)
        except Exception as exc:  # noqa: BLE001
            log.warning("vault routing failed: %s", exc)


async def _retry_unverified(rt: AgentRuntime) -> None:
    """Re-attempt any artifact submissions left in 'unverified' state."""
    if rt.artifacts is None:
        return
    rows = rt.store.list_unverified_artifacts()
    if not rows:
        return
    log.info("retrying %d unverified artifact(s)", len(rows))
    for row in rows:
        artifact = ValidationArtifact(
            decision_hash=row["decision_hash"],
            trade_hash=row["trade_hash"],
            risk_checks={},  # not retried — informational only on resubmit
            pre_state_hash=row["pre_state_hash"],
            post_state_hash=row["post_state_hash"],
            reasoning_uri=row["reasoning_uri"],
            timestamp=utcnow(),
            agent_id=str(rt.agent_id),
        )
        try:
            submission = await asyncio.to_thread(rt.artifacts.submit, artifact)
            rt.store.mark_artifact_verified(
                int(row["id"]),
                tx_hash=submission.tx_hash,
                via=submission.via,
                block_number=submission.block_number,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("retry failed for artifact %s: %s", row["id"], exc)


# ============================================================ helpers

def _compute_quantity(*, size_pct: float, equity_usd: float, mid_price: float) -> float:
    if mid_price <= 0 or equity_usd <= 0 or size_pct <= 0:
        return 0.0
    notional = size_pct * equity_usd
    qty = notional / mid_price
    # Round to 6 decimals — Kraken's BTC tick. Cheap defensive normalization.
    return float(f"{qty:.6f}")


def _signals_for_prompt(snapshot: MultiTimeframeSnapshot) -> dict[str, float]:
    """Flatten the multi-timeframe snapshot into a single signals dict."""
    out: dict[str, float] = {}
    for interval, snap in snapshot.timeframes.items():
        ind = snap.indicators
        for name, value in {
            f"{interval}_ma_fast": ind.ma_fast,
            f"{interval}_ma_slow": ind.ma_slow,
            f"{interval}_rsi": ind.rsi,
            f"{interval}_close": snap.last_price,
        }.items():
            if value is None:
                continue
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                continue
    out["bid"] = snapshot.quote.bid
    out["ask"] = snapshot.quote.ask
    out["mid"] = snapshot.quote.mid
    return out


def _portfolio_for_prompt(p: PortfolioSnapshot) -> dict[str, Any]:
    return {
        "cash_usd": p.cash_usd,
        "equity_usd": p.equity_usd,
        "realized_pnl_usd": p.realized_pnl_usd,
        "unrealized_pnl_usd": p.unrealized_pnl_usd,
        "open_positions": [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
            }
            for pos in p.positions
        ],
    }


def _build_artifact(
    *,
    rt: AgentRuntime,
    decision: Decision,
    fill: Fill,
    risked: RiskedDecision,
    portfolio_pre: PortfolioSnapshot,
    portfolio_post: PortfolioSnapshot,
) -> ValidationArtifact:
    reasoning_uri = _reasoning_to_data_uri(decision.reasoning)
    return ValidationArtifact(
        decision_hash=canonical_hash(decision),
        trade_hash=canonical_hash(fill),
        risk_checks=dict(risked.risk_checks),
        pre_state_hash=snapshot_state_hash(portfolio_pre),
        post_state_hash=snapshot_state_hash(portfolio_post),
        reasoning_uri=reasoning_uri,
        timestamp=utcnow(),
        agent_id=str(rt.agent_id),
    )


def _reasoning_to_data_uri(reasoning: str) -> str:
    """Pack the natural-language reasoning into a self-contained data: URI.

    Same trick `AgentManifest` uses — keeps us off IPFS for the demo while
    making the reasoning publicly fetchable from the on-chain pointer.
    """
    encoded = base64.b64encode(reasoning.encode()).decode()
    return f"data:text/plain;base64,{encoded}"


# ============================================================ driver

async def run_loop(rt: AgentRuntime, *, once: bool = False) -> None:
    stop = asyncio.Event()

    def _signal_handler() -> None:
        log.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows / restricted env: fall back to default handling.
            pass

    while not stop.is_set():
        await run_one_cycle(rt)
        if once:
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=rt.cfg.loop_interval_seconds)
        except asyncio.TimeoutError:
            pass


# ============================================================ entrypoint

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agent.main", description="Trading agent loop")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--mode", choices=("paper", "live"), default=None,
                   help="Override execution.mode from the config file")
    p.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    p.add_argument("--i-know", action="store_true",
                   help="Required to run --mode live. Real money goes here.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def amain() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = LoopConfig.from_yaml(args.config)
    if args.mode is not None:
        cfg = _override_mode(cfg, args.mode)

    if cfg.mode == "live" and not args.i_know:
        log.error(
            "refusing to run --mode live without --i-know. "
            "Confirm with the organizers that the leaderboard accepts your "
            "execution mode and that you intend to risk real capital."
        )
        return 2

    rt = await build_runtime(cfg=cfg, db_path=args.db)
    try:
        await run_loop(rt, once=args.once)
    finally:
        await rt.aclose()
    return 0


def _override_mode(cfg: LoopConfig, mode: str) -> LoopConfig:
    return LoopConfig(
        symbols=cfg.symbols,
        loop_interval_seconds=cfg.loop_interval_seconds,
        candle_interval=cfg.candle_interval,
        candle_lookback=cfg.candle_lookback,
        intervals_multiframe=cfg.intervals_multiframe,
        risk=cfg.risk,
        llm_model=cfg.llm_model,
        llm_max_tokens=cfg.llm_max_tokens,
        llm_temperature=cfg.llm_temperature,
        starting_capital_usd=cfg.starting_capital_usd,
        fee_bps=cfg.fee_bps,
        mode=mode,
        chain_enabled=cfg.chain_enabled,
    )


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
