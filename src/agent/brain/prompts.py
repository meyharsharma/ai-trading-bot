"""
System prompt + Claude tool schema for the LLM strategist.

The strategist is forced to call the `submit_decision` tool. Its arguments map
1:1 onto `agent.state.models.Decision`. Keeping the schema in lockstep with the
pydantic model is critical — if `Decision` ever changes, this file changes.
"""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are the strategist brain of an autonomous spot crypto trading agent.

Hard rules (the deterministic risk gate will reject anything that violates these,
so do NOT bother trying):
- Spot only. No leverage. No shorts. Only BUY, SELL, or HOLD.
- SELL means closing an existing long. Do not SELL a symbol you do not hold.
- Risk per trade (size_pct * stop_loss_pct) must not exceed 2% of capital.
- A single position must never exceed 25% of capital.
- Stop loss must be between 3% and 8%. Take profit must be <= 20%.
- Maximum 3 concurrent open positions.

Inputs you will receive each cycle:
- symbol, latest price, MA(fast), MA(slow), RSI(14), ATR if available
- current portfolio: cash, open positions, unrealized PnL
- recent decisions (for continuity)

How to think:
1. Read the trend (MA fast vs MA slow) and momentum (RSI).
2. Decide BUY / SELL / HOLD. When in doubt, HOLD — capital preservation beats
   forcing a trade. Most cycles should be HOLD.
3. Size conservatively. A typical entry is 5–15% of capital with a 3–5% stop.
4. Justify the decision in one short paragraph referencing the actual numbers
   you saw. This reasoning is anchored on-chain — be precise, not flowery.

You MUST respond by calling the `submit_decision` tool exactly once. Do not
write free-form text.
"""


DECISION_TOOL: dict[str, Any] = {
    "name": "submit_decision",
    "description": (
        "Submit the trading decision for this cycle. Exactly one call per cycle. "
        "Arguments are validated against the Decision schema and then passed "
        "through the deterministic risk gate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "enum": ["BTC/USD", "ETH/USD"],
            },
            "action": {
                "type": "string",
                "enum": ["BUY", "SELL", "HOLD"],
            },
            "size_pct": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Fraction of total capital to deploy on this trade (0..1).",
            },
            "stop_loss_pct": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Stop distance as a fraction of entry price (e.g. 0.03 = 3%).",
            },
            "take_profit_pct": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Take-profit distance as a fraction of entry price.",
            },
            "reasoning": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "One-paragraph natural language rationale citing the actual "
                    "indicator values. Anchored on-chain — be specific."
                ),
            },
            "signals": {
                "type": "object",
                "description": "Numeric snapshot of the indicators you used.",
                "additionalProperties": {"type": "number"},
            },
        },
        "required": [
            "symbol",
            "action",
            "size_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "reasoning",
            "signals",
        ],
    },
}


CRITIQUE_SYSTEM_PROMPT = """\
You are a skeptical risk reviewer auditing a trading decision proposed by
another instance of yourself. Your only job is to find the weakest link in
the thesis.

Be aggressive. False positives (rejecting a fine trade) cost us a missed
opportunity; false negatives (approving a bad trade) cost us capital. The
asymmetry favors rejection.

You MUST call the `submit_critique` tool exactly once. Provide:
  - verdict: ACCEPT if the thesis holds up, REJECT if the weakness is fatal
  - weakness: the single biggest hole in the reasoning, in one sentence

Reject when ANY of these are true:
  - The reasoning contradicts the cited indicator values
  - The trend signal is weak (MAs nearly equal, no clear cross)
  - RSI is in extreme territory in the wrong direction
  - The stop is wider than the recent volatility justifies
  - The trade would concentrate risk in an already-held symbol
  - The reasoning is generic and doesn't reference the actual numbers
"""


CRITIQUE_TOOL: dict[str, Any] = {
    "name": "submit_critique",
    "description": (
        "Submit your critique of the proposed decision. Call exactly once. "
        "Verdict must be ACCEPT or REJECT; weakness is the single biggest hole."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["ACCEPT", "REJECT"],
            },
            "weakness": {
                "type": "string",
                "minLength": 1,
                "description": "One sentence naming the weakest part of the thesis.",
            },
        },
        "required": ["verdict", "weakness"],
    },
}


def render_critique_prompt(
    *,
    decision: dict[str, Any],
    price: float,
    signals: dict[str, float],
    portfolio: dict[str, Any],
) -> str:
    """Format the user message for the second-pass critique call."""
    lines: list[str] = []
    lines.append("## Decision under review")
    lines.append(f"- symbol: {decision.get('symbol')}")
    lines.append(f"- action: {decision.get('action')}")
    lines.append(f"- size_pct: {decision.get('size_pct')}")
    lines.append(f"- stop_loss_pct: {decision.get('stop_loss_pct')}")
    lines.append(f"- take_profit_pct: {decision.get('take_profit_pct')}")
    lines.append(f"- reasoning: {decision.get('reasoning')}")
    lines.append("")
    lines.append("## Market context (verify against the reasoning)")
    lines.append(f"- last price: {price:.2f}")
    for k, v in sorted(signals.items()):
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")
    lines.append("## Portfolio")
    lines.append(f"- cash_usd: {portfolio.get('cash_usd', 0.0):.2f}")
    lines.append(f"- equity_usd: {portfolio.get('equity_usd', 0.0):.2f}")
    positions = portfolio.get("positions", []) or []
    if positions:
        for p in positions:
            lines.append(
                f"- holding {p.get('symbol')} qty={p.get('quantity')} entry={p.get('avg_entry_price')}"
            )
    lines.append("")
    lines.append(
        "Call `submit_critique`. Remember: when in doubt, REJECT — HOLD is free."
    )
    return "\n".join(lines)


def render_user_prompt(
    *,
    symbol: str,
    price: float,
    signals: dict[str, float],
    portfolio: dict[str, Any],
    recent_decisions: list[dict[str, Any]] | None = None,
) -> str:
    """Format a single-cycle user message for Claude."""
    lines: list[str] = []
    lines.append(f"## Cycle context for {symbol}")
    lines.append(f"- last price: {price:.2f}")
    lines.append("- indicators:")
    for k, v in sorted(signals.items()):
        lines.append(f"    - {k}: {v:.4f}")

    lines.append("- portfolio:")
    lines.append(f"    - cash_usd: {portfolio.get('cash_usd', 0.0):.2f}")
    lines.append(f"    - equity_usd: {portfolio.get('equity_usd', 0.0):.2f}")
    positions = portfolio.get("positions", []) or []
    if positions:
        lines.append("    - open positions:")
        for p in positions:
            lines.append(
                f"        * {p.get('symbol')}: qty={p.get('quantity')} "
                f"entry={p.get('avg_entry_price')} stop={p.get('stop_loss_price')}"
            )
    else:
        lines.append("    - open positions: none")

    if recent_decisions:
        lines.append("- recent decisions:")
        for d in recent_decisions[-3:]:
            lines.append(
                f"    * {d.get('timestamp')} {d.get('symbol')} {d.get('action')} "
                f"size={d.get('size_pct')}"
            )

    lines.append("")
    lines.append(
        "Call `submit_decision` with your decision. Remember: when in doubt, HOLD."
    )
    return "\n".join(lines)
