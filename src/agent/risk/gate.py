"""
Deterministic risk gate.

Sits between the LLM strategist and the execution layer. Its only contract:
take a `Decision` (and a snapshot of the current portfolio) and return a
`RiskedDecision` that either passes (possibly clamped) or is rejected with
a list of human-readable reasons.

This file is the *only* place hard risk rules live. The LLM is free to
hallucinate; the gate is not. Every rule below maps to a key under `risk:` in
`config/strategy.yaml` so we can tune from config without touching code.

Rules implemented (all from strategy.yaml):
    risk.max_risk_per_trade_pct  → size_pct * stop_loss_pct must be <= this
    risk.max_open_positions      → reject BUY if at limit
    risk.max_position_size_pct   → clamp size_pct down
    risk.default_stop_loss_pct   → minimum stop, clamp narrower stops up
    risk.max_stop_loss_pct       → clamp wider stops down
    risk.max_take_profit_pct     → reject if exceeded (we don't store TP in
                                   RiskedDecision; flag the violation instead)
    risk.allow_leverage          → false ⇒ size_pct capped at 1.0 (sanity)
    risk.allow_shorts            → false ⇒ SELL only allowed if we hold the symbol

HOLDs always pass (with size clamped to 0) — they cost nothing and let the
loop continue.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.state import Decision, PortfolioSnapshot, RiskedDecision


# ---------- config ----------

@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade_pct: float
    max_open_positions: int
    max_position_size_pct: float
    default_stop_loss_pct: float
    max_stop_loss_pct: float
    max_take_profit_pct: float
    allow_leverage: bool
    allow_shorts: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RiskConfig":
        return cls(
            max_risk_per_trade_pct=float(raw["max_risk_per_trade_pct"]),
            max_open_positions=int(raw["max_open_positions"]),
            max_position_size_pct=float(raw["max_position_size_pct"]),
            default_stop_loss_pct=float(raw["default_stop_loss_pct"]),
            max_stop_loss_pct=float(raw["max_stop_loss_pct"]),
            max_take_profit_pct=float(raw["max_take_profit_pct"]),
            allow_leverage=bool(raw["allow_leverage"]),
            allow_shorts=bool(raw["allow_shorts"]),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RiskConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data["risk"])


# ---------- gate ----------

class RiskGate:
    """Stateless evaluator. Inject portfolio state per call."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def evaluate(
        self,
        decision: Decision,
        portfolio: PortfolioSnapshot | None = None,
    ) -> RiskedDecision:
        c = self.config
        reasons: list[str] = []
        checks: dict[str, bool] = {}
        clamped = False

        # HOLD short-circuit ------------------------------------------------
        if decision.action == "HOLD":
            checks["hold_passthrough"] = True
            return RiskedDecision(
                decision=decision,
                passed=True,
                clamped=False,
                reasons=[],
                risk_checks=checks,
                final_size_pct=0.0,
                final_stop_loss_pct=max(decision.stop_loss_pct, c.default_stop_loss_pct),
            )

        # SELL constraints --------------------------------------------------
        held_symbols = {p.symbol for p in (portfolio.positions if portfolio else ())}
        if decision.action == "SELL":
            if not c.allow_shorts and decision.symbol not in held_symbols:
                checks["allow_shorts"] = False
                reasons.append(
                    f"SELL on {decision.symbol} would open a short — shorts disabled"
                )
                return self._reject(decision, reasons, checks)
            checks["allow_shorts"] = True

        # BUY constraints ---------------------------------------------------
        if decision.action == "BUY" and portfolio is not None:
            open_count = len(portfolio.positions)
            if decision.symbol not in held_symbols and open_count >= c.max_open_positions:
                checks["max_open_positions"] = False
                reasons.append(
                    f"Already holding {open_count} positions (cap {c.max_open_positions})"
                )
                return self._reject(decision, reasons, checks)
            checks["max_open_positions"] = True

        # Stop-loss clamping ------------------------------------------------
        sl = decision.stop_loss_pct
        if sl < c.default_stop_loss_pct:
            sl = c.default_stop_loss_pct
            clamped = True
            reasons.append(
                f"stop_loss_pct raised to default {c.default_stop_loss_pct}"
            )
        if sl > c.max_stop_loss_pct:
            sl = c.max_stop_loss_pct
            clamped = True
            reasons.append(
                f"stop_loss_pct capped at max {c.max_stop_loss_pct}"
            )
        checks["stop_loss_in_band"] = True

        # Take-profit check (advisory; not stored in RiskedDecision) --------
        if decision.take_profit_pct > c.max_take_profit_pct:
            checks["max_take_profit"] = False
            reasons.append(
                f"take_profit_pct {decision.take_profit_pct} > max {c.max_take_profit_pct}"
            )
            # Not fatal — clamp implicitly downstream. Flag and continue.
            clamped = True
        else:
            checks["max_take_profit"] = True

        # Position size clamping --------------------------------------------
        size = decision.size_pct
        if size > c.max_position_size_pct:
            size = c.max_position_size_pct
            clamped = True
            reasons.append(
                f"size_pct capped at max_position_size_pct {c.max_position_size_pct}"
            )
        checks["max_position_size"] = True

        # No-leverage sanity (cannot exceed 100% of capital) ----------------
        if not c.allow_leverage and size > 1.0:
            size = 1.0
            clamped = True
            reasons.append("size_pct capped at 1.0 (no leverage)")
        checks["no_leverage"] = True

        # Risk-per-trade rule -----------------------------------------------
        # Effective $-risk = size_pct * stop_loss_pct (fraction of equity).
        risk = size * sl
        if risk > c.max_risk_per_trade_pct:
            # Clamp size so risk == max_risk_per_trade_pct (sl already in band).
            new_size = c.max_risk_per_trade_pct / sl if sl > 0 else 0.0
            if new_size < size:
                size = max(0.0, new_size)
                clamped = True
                reasons.append(
                    f"size_pct clamped to honor max_risk_per_trade_pct "
                    f"{c.max_risk_per_trade_pct}"
                )
        checks["max_risk_per_trade"] = True

        # If clamping zeroed the size, this is effectively a HOLD — pass it
        # through but mark passed=False so the caller knows nothing executes.
        if size <= 0.0:
            reasons.append("size clamped to zero — nothing to execute")
            return RiskedDecision(
                decision=decision,
                passed=False,
                clamped=True,
                reasons=reasons,
                risk_checks=checks,
                final_size_pct=0.0,
                final_stop_loss_pct=sl,
            )

        return RiskedDecision(
            decision=decision,
            passed=True,
            clamped=clamped,
            reasons=reasons,
            risk_checks=checks,
            final_size_pct=size,
            final_stop_loss_pct=sl,
        )

    # ---------- helpers ----------

    @staticmethod
    def _reject(
        decision: Decision,
        reasons: list[str],
        checks: dict[str, bool],
    ) -> RiskedDecision:
        return RiskedDecision(
            decision=decision,
            passed=False,
            clamped=False,
            reasons=reasons,
            risk_checks=checks,
            final_size_pct=0.0,
            final_stop_loss_pct=max(decision.stop_loss_pct, 0.0),
        )
