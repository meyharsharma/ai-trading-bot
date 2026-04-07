"""
Prompt evaluation harness.

Runs a strategist N times against the same market snapshot and measures
how much the resulting decisions vary. Low variance at temperature > 0
is the cleanest single piece of evidence that the agent is *reasoning*
about the data rather than rolling dice — exactly the trustless-narrative
hook the ERC-8004 judges are looking for.

Metrics reported per snapshot:
    - sample size
    - dominant action (the modal BUY/SELL/HOLD) and its share
    - action_consistency: dominant_share (1.0 = perfectly stable)
    - size_pct mean / std
    - stop_loss_pct mean / std

The harness is strategist-agnostic: pass any object with a `.decide(...)`
method. For LLM strategists, the same snapshot can be re-evaluated by the
real Anthropic client to measure true sampling variance.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agent.state import Decision, Symbol


@dataclass
class SnapshotEval:
    label: str
    n: int
    dominant_action: str
    action_consistency: float
    action_counts: dict[str, int]
    size_pct_mean: float
    size_pct_std: float
    stop_loss_pct_mean: float
    stop_loss_pct_std: float
    decisions: list[Decision] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n": self.n,
            "dominant_action": self.dominant_action,
            "action_consistency": self.action_consistency,
            "action_counts": self.action_counts,
            "size_pct_mean": self.size_pct_mean,
            "size_pct_std": self.size_pct_std,
            "stop_loss_pct_mean": self.stop_loss_pct_mean,
            "stop_loss_pct_std": self.stop_loss_pct_std,
        }


@dataclass
class Snapshot:
    label: str
    symbol: Symbol
    price: float
    signals: dict[str, float]
    portfolio: dict[str, Any]


def evaluate_snapshot(
    strategist,
    snapshot: Snapshot,
    *,
    n_runs: int = 10,
) -> SnapshotEval:
    """Run `strategist.decide(...)` n_runs times against `snapshot`."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    decisions: list[Decision] = []
    for _ in range(n_runs):
        d = strategist.decide(
            symbol=snapshot.symbol,
            price=snapshot.price,
            signals=dict(snapshot.signals),
            portfolio=dict(snapshot.portfolio),
        )
        decisions.append(d)

    actions = [d.action for d in decisions]
    counts = Counter(actions)
    dominant_action, dominant_count = counts.most_common(1)[0]
    consistency = dominant_count / n_runs

    sizes = [d.size_pct for d in decisions]
    stops = [d.stop_loss_pct for d in decisions]

    size_std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
    stop_std = statistics.stdev(stops) if len(stops) > 1 else 0.0

    return SnapshotEval(
        label=snapshot.label,
        n=n_runs,
        dominant_action=dominant_action,
        action_consistency=consistency,
        action_counts=dict(counts),
        size_pct_mean=statistics.fmean(sizes),
        size_pct_std=size_std,
        stop_loss_pct_mean=statistics.fmean(stops),
        stop_loss_pct_std=stop_std,
        decisions=decisions,
    )


def evaluate_many(
    strategist,
    snapshots: list[Snapshot],
    *,
    n_runs: int = 10,
) -> list[SnapshotEval]:
    return [evaluate_snapshot(strategist, s, n_runs=n_runs) for s in snapshots]


def format_eval(reports: list[SnapshotEval]) -> str:
    header = (
        f"{'snapshot':<18} {'n':>4} {'mode':>6} {'consist':>9} "
        f"{'sizeμ':>8} {'sizeσ':>8} {'stopμ':>8} {'stopσ':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r.label:<18} {r.n:>4d} {r.dominant_action:>6} "
            f"{r.action_consistency * 100:>8.1f}% "
            f"{r.size_pct_mean:>8.4f} {r.size_pct_std:>8.4f} "
            f"{r.stop_loss_pct_mean:>8.4f} {r.stop_loss_pct_std:>8.4f}"
        )
    return "\n".join(lines)


def overall_consistency(reports: list[SnapshotEval]) -> float:
    """Mean action_consistency across all snapshots — a single headline number."""
    if not reports:
        return 0.0
    return math.fsum(r.action_consistency for r in reports) / len(reports)
