"""Risk gate is the only thing standing between a hallucinating LLM and our
capital. Test it like our money depends on it — because it does."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.risk.gate import RiskConfig, RiskGate
from agent.state import Decision, PortfolioSnapshot, Position, utcnow


@pytest.fixture
def config() -> RiskConfig:
    return RiskConfig(
        max_risk_per_trade_pct=0.02,
        max_open_positions=3,
        max_position_size_pct=0.25,
        default_stop_loss_pct=0.03,
        max_stop_loss_pct=0.08,
        max_take_profit_pct=0.20,
        allow_leverage=False,
        allow_shorts=False,
    )


@pytest.fixture
def gate(config: RiskConfig) -> RiskGate:
    return RiskGate(config)


def _decision(**overrides) -> Decision:
    base = dict(
        timestamp=utcnow(),
        symbol="BTC/USD",
        action="BUY",
        size_pct=0.10,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        reasoning="test",
        signals={},
        model="test",
    )
    base.update(overrides)
    return Decision(**base)


def _empty_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=utcnow(),
        cash_usd=1000.0,
        positions=(),
        equity_usd=1000.0,
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
    )


def _portfolio_with(symbols: list[str]) -> PortfolioSnapshot:
    positions = tuple(
        Position(
            symbol=s,
            quantity=0.001,
            avg_entry_price=60000.0,
            stop_loss_price=58000.0,
            take_profit_price=63000.0,
            opened_at=utcnow(),
        )
        for s in symbols
    )
    return PortfolioSnapshot(
        timestamp=utcnow(),
        cash_usd=500.0,
        positions=positions,
        equity_usd=1000.0,
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
    )


# ---------- HOLD ----------

def test_hold_passes_with_zero_size(gate: RiskGate) -> None:
    rd = gate.evaluate(_decision(action="HOLD", size_pct=0.5), _empty_portfolio())
    assert rd.passed
    assert rd.final_size_pct == 0.0
    assert not rd.clamped


# ---------- size clamping ----------

def test_size_clamped_to_max_position(gate: RiskGate) -> None:
    rd = gate.evaluate(
        _decision(size_pct=0.80, stop_loss_pct=0.03, take_profit_pct=0.05),
        _empty_portfolio(),
    )
    # 0.80 → clamp to 0.25 (max_position_size), then risk=0.25*0.03=0.0075 ≤ 2%
    assert rd.passed
    assert rd.clamped
    assert rd.final_size_pct == pytest.approx(0.25)


def test_risk_per_trade_clamps_size(gate: RiskGate) -> None:
    # max_position 25%, stop=8% → risk=0.02 exactly OK.
    # Use stop=0.06 → max allowed size = 0.02/0.06 ≈ 0.333; capped first by 0.25.
    rd = gate.evaluate(
        _decision(size_pct=0.25, stop_loss_pct=0.06, take_profit_pct=0.10),
        _empty_portfolio(),
    )
    assert rd.passed
    # 0.25 * 0.06 = 0.015 < 0.02 → no further clamp.
    assert rd.final_size_pct == pytest.approx(0.25)

    # Now force a violation: small max_position bypass via direct config tweak.
    tight_cfg = RiskConfig(
        max_risk_per_trade_pct=0.02,
        max_open_positions=3,
        max_position_size_pct=1.0,  # disable position-size clamp
        default_stop_loss_pct=0.03,
        max_stop_loss_pct=0.08,
        max_take_profit_pct=0.20,
        allow_leverage=False,
        allow_shorts=False,
    )
    tight_gate = RiskGate(tight_cfg)
    rd2 = tight_gate.evaluate(
        _decision(size_pct=0.50, stop_loss_pct=0.08), _empty_portfolio()
    )
    # 0.50*0.08 = 0.04 > 0.02 → clamp to 0.02/0.08 = 0.25
    assert rd2.passed
    assert rd2.clamped
    assert rd2.final_size_pct == pytest.approx(0.25)


# ---------- stop-loss clamping ----------

def test_stop_loss_raised_to_default(gate: RiskGate) -> None:
    rd = gate.evaluate(_decision(stop_loss_pct=0.01), _empty_portfolio())
    assert rd.final_stop_loss_pct == pytest.approx(0.03)
    assert rd.clamped


def test_stop_loss_capped_at_max(gate: RiskGate) -> None:
    rd = gate.evaluate(_decision(stop_loss_pct=0.15, size_pct=0.05), _empty_portfolio())
    assert rd.final_stop_loss_pct == pytest.approx(0.08)
    assert rd.clamped


# ---------- max open positions ----------

def test_buy_rejected_when_at_max_positions(gate: RiskGate) -> None:
    portfolio = _portfolio_with(["ETH/USD", "ETH/USD", "ETH/USD"])  # 3 open
    rd = gate.evaluate(_decision(symbol="BTC/USD", action="BUY"), portfolio)
    assert not rd.passed
    assert any("max_open_positions" in r or "positions" in r for r in rd.reasons)


def test_buy_allowed_when_already_holding_symbol(gate: RiskGate) -> None:
    # If we already hold BTC, "buying more" of an already-open symbol shouldn't
    # be blocked by the open-position counter (no new slot consumed).
    portfolio = _portfolio_with(["BTC/USD", "ETH/USD", "ETH/USD"])
    rd = gate.evaluate(_decision(symbol="BTC/USD", action="BUY"), portfolio)
    assert rd.passed


# ---------- shorts ----------

def test_sell_without_position_rejected(gate: RiskGate) -> None:
    rd = gate.evaluate(_decision(action="SELL"), _empty_portfolio())
    assert not rd.passed
    assert rd.risk_checks.get("allow_shorts") is False


def test_sell_with_position_allowed(gate: RiskGate) -> None:
    rd = gate.evaluate(_decision(action="SELL"), _portfolio_with(["BTC/USD"]))
    assert rd.passed


# ---------- take-profit ----------

def test_take_profit_overflow_flagged(gate: RiskGate) -> None:
    rd = gate.evaluate(_decision(take_profit_pct=0.50), _empty_portfolio())
    # Not fatal but must be flagged.
    assert rd.risk_checks.get("max_take_profit") is False
    assert rd.clamped


# ---------- config loading ----------

def test_config_round_trips_from_yaml(tmp_path) -> None:
    import yaml

    cfg_path = tmp_path / "strategy.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "risk": {
                    "max_risk_per_trade_pct": 0.02,
                    "max_open_positions": 3,
                    "max_position_size_pct": 0.25,
                    "default_stop_loss_pct": 0.03,
                    "max_stop_loss_pct": 0.08,
                    "max_take_profit_pct": 0.20,
                    "allow_leverage": False,
                    "allow_shorts": False,
                }
            }
        )
    )
    cfg = RiskConfig.from_yaml(cfg_path)
    assert cfg.max_risk_per_trade_pct == 0.02
    assert cfg.allow_shorts is False


def test_loads_real_strategy_yaml() -> None:
    """Make sure config/strategy.yaml stays compatible with the gate."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    cfg = RiskConfig.from_yaml(repo_root / "config" / "strategy.yaml")
    assert cfg.max_open_positions >= 1
    assert 0.0 < cfg.max_risk_per_trade_pct < 0.1
