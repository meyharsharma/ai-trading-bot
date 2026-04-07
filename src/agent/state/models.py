"""
Frozen data contracts shared across all four layers.

These models are the merge boundary between worktrees. Do NOT change them in a
feature branch — any change goes to `main` first and the other worktrees rebase.

Layer flow:
    Brain      → produces Decision
    Risk Gate  → consumes Decision, produces RiskedDecision
    Exec       → consumes RiskedDecision, produces Fill (and updates Position)
    Chain      → consumes (RiskedDecision, Fill, portfolio snapshots) → ValidationArtifact
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Symbol = Literal["BTC/USD", "ETH/USD"]
Action = Literal["BUY", "SELL", "HOLD"]


# ---------- Brain output ----------

class Decision(BaseModel):
    """Raw decision emitted by the LLM strategist. Not yet risk-checked."""
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    symbol: Symbol
    action: Action
    size_pct: float = Field(ge=0.0, le=1.0, description="Fraction of capital to deploy")
    stop_loss_pct: float = Field(ge=0.0, le=1.0)
    take_profit_pct: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, description="Natural-language rationale (ERC-8004 gold)")
    signals: dict[str, float] = Field(default_factory=dict)
    model: str


# ---------- Risk gate output ----------

class RiskedDecision(BaseModel):
    """Output of the risk gate. Either passes (possibly clamped) or is rejected."""
    model_config = ConfigDict(frozen=True)

    decision: Decision
    passed: bool
    clamped: bool = False
    reasons: list[str] = Field(default_factory=list)
    risk_checks: dict[str, bool] = Field(default_factory=dict)
    final_size_pct: float = Field(ge=0.0, le=1.0)
    final_stop_loss_pct: float = Field(ge=0.0, le=1.0)


# ---------- Execution layer ----------

class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: Symbol
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True)

    order: Order
    filled_at: datetime
    fill_price: float
    fee_usd: float
    venue: Literal["paper", "kraken"]
    venue_order_id: str | None = None


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: Symbol
    quantity: float
    avg_entry_price: float
    stop_loss_price: float
    take_profit_price: float
    opened_at: datetime


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    cash_usd: float
    positions: tuple[Position, ...] = ()
    equity_usd: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float


# ---------- On-chain layer ----------

class ValidationArtifact(BaseModel):
    """Payload anchored on-chain via the ERC-8004 registry / vault."""
    model_config = ConfigDict(frozen=True)

    decision_hash: str
    trade_hash: str | None
    risk_checks: dict[str, bool]
    pre_state_hash: str
    post_state_hash: str
    reasoning_uri: str
    timestamp: datetime
    agent_id: str


# ---------- Canonical hashing ----------

def canonical_hash(obj: BaseModel | dict[str, Any]) -> str:
    """
    Deterministic sha256 over a model's JSON representation.

    Used everywhere a hash anchors something on-chain. Must be stable across
    Python versions and across worktrees — keep this function frozen.
    """
    if isinstance(obj, BaseModel):
        payload = obj.model_dump(mode="json")
    else:
        payload = obj
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "0x" + hashlib.sha256(encoded.encode()).hexdigest()


def utcnow() -> datetime:
    """Single source of truth for timestamps. Always UTC, always tz-aware."""
    return datetime.now(timezone.utc)
