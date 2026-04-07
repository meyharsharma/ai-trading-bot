"""
LLM strategist: turns market context into a `Decision` via Claude tool-use.

The strategist's only job is to call Anthropic with the system prompt + a
single-cycle user message and force a `submit_decision` tool call. We then
validate the tool input against the frozen `Decision` pydantic model.

Design notes
------------
- The Anthropic client is injected, so tests can pass a fake client without
  monkey-patching anything.
- On any failure (network, malformed tool call, schema mismatch) we return a
  conservative HOLD decision with the failure reason captured in `reasoning`.
  HOLD is always safe and lets the loop continue. The risk gate will pass it
  through trivially.
- We never accept free-form text output. If the model declines to call the
  tool, we treat it as a failure and HOLD.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from agent.brain.prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_TOOL,
    DECISION_TOOL,
    SYSTEM_PROMPT,
    render_critique_prompt,
    render_user_prompt,
)
from agent.state import Decision, Symbol, utcnow

log = logging.getLogger(__name__)


class AnthropicLike(Protocol):
    """Minimal surface area we use from the anthropic SDK. Lets us fake it in tests."""

    class messages:  # noqa: N801 — mirroring SDK shape
        @staticmethod
        def create(**kwargs: Any) -> Any: ...


class LLMStrategist:
    """Wraps Claude tool-use into a typed `Decision`."""

    def __init__(
        self,
        client: Any,
        *,
        model: str = "claude-opus-4-6",
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ---------- public ----------

    def decide(
        self,
        *,
        symbol: Symbol,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
        recent_decisions: list[dict[str, Any]] | None = None,
    ) -> Decision:
        """Run one strategist cycle. Always returns a Decision (HOLD on failure)."""
        user_prompt = render_user_prompt(
            symbol=symbol,
            price=price,
            signals=signals,
            portfolio=portfolio,
            recent_decisions=recent_decisions,
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                tools=[DECISION_TOOL],
                tool_choice={"type": "tool", "name": DECISION_TOOL["name"]},
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # noqa: BLE001 — network/SDK errors all funnel here
            log.warning("LLM call failed: %s", exc)
            return self._hold(symbol, signals, f"LLM call failed: {exc}")

        tool_input = self._extract_tool_input(response)
        if tool_input is None:
            return self._hold(symbol, signals, "Model returned no tool call")

        # The model may pick a different symbol than the one we asked about
        # (e.g. answering "no signal here, hold ETH instead"). We trust its
        # symbol selection but still validate against the schema.
        try:
            return Decision(
                timestamp=utcnow(),
                symbol=tool_input["symbol"],
                action=tool_input["action"],
                size_pct=float(tool_input["size_pct"]),
                stop_loss_pct=float(tool_input["stop_loss_pct"]),
                take_profit_pct=float(tool_input["take_profit_pct"]),
                reasoning=str(tool_input["reasoning"]),
                signals={k: float(v) for k, v in (tool_input.get("signals") or {}).items()},
                model=self.model,
            )
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            log.warning("LLM tool input failed schema validation: %s", exc)
            return self._hold(symbol, signals, f"Schema validation failed: {exc}")

    # ---------- internals ----------

    @staticmethod
    def _extract_tool_input(
        response: Any,
        *,
        tool_name: str = DECISION_TOOL["name"],
    ) -> dict[str, Any] | None:
        """
        Pull the first matching tool_use block out of the response.

        Tolerates both the official anthropic SDK shape (objects with .type /
        .input) and a plain-dict shape that's convenient for tests.
        """
        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content")
        if not content:
            return None

        for block in content:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type != "tool_use":
                continue

            name = getattr(block, "name", None)
            if name is None and isinstance(block, dict):
                name = block.get("name")
            if name != tool_name:
                continue

            tool_input = getattr(block, "input", None)
            if tool_input is None and isinstance(block, dict):
                tool_input = block.get("input")
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    return None
            if isinstance(tool_input, dict):
                return tool_input
        return None

    # ---------- self-critique (second pass) ----------

    def critique(
        self,
        decision: Decision,
        *,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
    ) -> "Critique":
        """
        Second-pass call: ask the model to find the weakest part of its own
        thesis and reject if fatal. Two-shot reasoning is the single biggest
        quality jump for LLM trading decisions and gives us a juicy
        validation artifact for ERC-8004.

        Defaults to ACCEPT on any failure (network, parse, schema). The first
        pass already cleared the schema; if the critic itself crashes we
        don't want to silently kill every trade.
        """
        prompt = render_critique_prompt(
            decision=decision.model_dump(mode="json"),
            price=price,
            signals=signals,
            portfolio=portfolio,
        )
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=CRITIQUE_SYSTEM_PROMPT,
                tools=[CRITIQUE_TOOL],
                tool_choice={"type": "tool", "name": CRITIQUE_TOOL["name"]},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Critique call failed, defaulting to ACCEPT: %s", exc)
            return Critique(verdict="ACCEPT", weakness=f"critique unavailable: {exc}")

        tool_input = self._extract_tool_input(response, tool_name=CRITIQUE_TOOL["name"])
        if tool_input is None:
            return Critique(verdict="ACCEPT", weakness="critic returned no tool call")

        verdict = tool_input.get("verdict")
        weakness = tool_input.get("weakness", "")
        if verdict not in ("ACCEPT", "REJECT") or not isinstance(weakness, str) or not weakness:
            return Critique(verdict="ACCEPT", weakness="critic returned malformed payload")
        return Critique(verdict=verdict, weakness=weakness)  # type: ignore[arg-type]

    def _hold(self, symbol: Symbol, signals: dict[str, float], reason: str) -> Decision:
        return Decision(
            timestamp=utcnow(),
            symbol=symbol,
            action="HOLD",
            size_pct=0.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            reasoning=f"Defaulted to HOLD: {reason}",
            signals={k: float(v) for k, v in signals.items()},
            model=self.model,
        )


# ---------- self-critique wrapper ----------

@dataclass(frozen=True)
class Critique:
    verdict: Literal["ACCEPT", "REJECT"]
    weakness: str


class StrategistLike(Protocol):
    """Duck-typed shape that `SelfCritiquingStrategist` wraps. Both
    `LLMStrategist` (anthropic SDK) and `ClaudeCodeStrategist` (local CLI)
    satisfy this without inheritance."""

    model: str

    def decide(
        self,
        *,
        symbol: Symbol,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
        recent_decisions: list[dict[str, Any]] | None = None,
    ) -> Decision: ...

    def critique(
        self,
        decision: Decision,
        *,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
    ) -> "Critique": ...


class SelfCritiquingStrategist:
    """
    Two-pass strategist: run the base strategist, then ask the same brain to
    audit its own decision. If the critic returns REJECT, downgrade to HOLD
    and embed the critique in the reasoning so it lands in the on-chain
    artifact.

    HOLDs from the base pass are returned untouched (nothing to critique).
    """

    def __init__(self, base: StrategistLike) -> None:
        self.base = base
        self.model = base.model

    def decide(
        self,
        *,
        symbol: Symbol,
        price: float,
        signals: dict[str, float],
        portfolio: dict[str, Any],
        recent_decisions: list[dict[str, Any]] | None = None,
    ) -> Decision:
        decision = self.base.decide(
            symbol=symbol,
            price=price,
            signals=signals,
            portfolio=portfolio,
            recent_decisions=recent_decisions,
        )
        if decision.action == "HOLD":
            return decision

        critique = self.base.critique(
            decision,
            price=price,
            signals=signals,
            portfolio=portfolio,
        )
        if critique.verdict == "ACCEPT":
            # Append the critic's note to the reasoning so the on-chain
            # artifact records that a second pass actually happened.
            return decision.model_copy(
                update={
                    "reasoning": (
                        f"{decision.reasoning}\n\n[critic ACCEPT] {critique.weakness}"
                    )
                }
            )

        return Decision(
            timestamp=utcnow(),
            symbol=decision.symbol,
            action="HOLD",
            size_pct=0.0,
            stop_loss_pct=decision.stop_loss_pct,
            take_profit_pct=decision.take_profit_pct,
            reasoning=(
                f"Critic REJECTED original {decision.action} thesis: "
                f"{critique.weakness}\n\nOriginal reasoning: {decision.reasoning}"
            ),
            signals=dict(decision.signals),
            model=decision.model,
        )
