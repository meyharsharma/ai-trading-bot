"""
ExecutionAdapter ABC.

The brain and the loop should NEVER import a concrete adapter directly. They
inject one selected at startup based on ``config.execution.mode`` (paper vs
kraken). This is what makes the live-vs-paper swap a one-line change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent.state.models import Fill, Order, PortfolioSnapshot


class ExecutionAdapter(ABC):
    """Abstract execution backend.

    Implementations:
      - ``PaperExecutionAdapter``  (default; in-memory portfolio, fills at
        live Kraken mid pulled via MCP)
      - ``KrakenLiveExecutionAdapter``  (places real orders via Kraken CLI MCP)
    """

    venue: str  # "paper" | "kraken"

    @abstractmethod
    async def submit_order(self, order: Order) -> Fill:
        """Submit ``order`` and return the resulting fill. Must raise on failure."""

    @abstractmethod
    async def get_portfolio(self) -> PortfolioSnapshot:
        """Return the current portfolio snapshot (cash + positions + PnL)."""

    @abstractmethod
    async def get_mark_price(self, symbol: str) -> float:
        """Return the current mark price the adapter would use to value ``symbol``."""

    async def aclose(self) -> None:  # optional cleanup hook
        return None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} venue={self.venue!r}>"

    # Subclasses may override to expose adapter-specific diagnostics.
    def info(self) -> dict[str, Any]:  # pragma: no cover - cosmetic
        return {"venue": self.venue}
