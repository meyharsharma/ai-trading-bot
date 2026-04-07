# ai-trading-bot

LLM-driven autonomous trading agent for the Kraken + ERC-8004 hackathon.

A Claude-powered strategist trades BTC/USD and ETH/USD via the **Kraken CLI MCP server**, with every decision deterministically risk-checked and atomically anchored on-chain as an **ERC-8004 validation artifact**. Auditable, trustless, backtested.

## Architecture

```
Data (Kraken CLI MCP) → Brain (Claude) → Risk Gate → Exec (Kraken CLI MCP) → On-Chain (ERC-8004)
```

Four loosely-coupled layers connected by frozen pydantic contracts (`Decision`, `ValidationArtifact`). Every executed trade emits an on-chain artifact in the same atomic step — if the chain write fails, the trade is flagged.

See `hackathon_strategy_brief.docx` for the full strategy and `docs/ARCHITECTURE.md` (coming soon) for the build details.

## Status

Day 1 — scaffolding.
