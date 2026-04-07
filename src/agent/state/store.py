"""
SQLite persistence for the trading loop.

Four tables, intentionally narrow:

    cycles      — one row per loop iteration. Operational, used by ops.
    decisions   — every Decision the brain emits + the RiskedDecision verdict.
                  HOLDs are persisted too because the audit story is "every
                  thought is recorded", not "every trade".
    fills       — every successful Fill from the execution adapter.
    artifacts   — every on-chain submission (or failed attempt). Joins fills
                  to chain receipts; status='unverified' rows are retry queue.

Schema is created on open. No migrations — if we need to change shape we
delete the file and re-run. This is a hackathon demo, not a production DB.

Why stdlib sqlite3 and not an ORM:
    The whole module is ~250 lines. SQLAlchemy would dwarf the actual logic
    and add a heavy import. Plain SQL is also easier for judges to query.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from agent.state.models import (
    Decision,
    Fill,
    PortfolioSnapshot,
    RiskedDecision,
    ValidationArtifact,
    canonical_hash,
    utcnow,
)

DEFAULT_DB_PATH = Path("state.sqlite")


SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,            -- 'running' | 'ok' | 'error'
    error         TEXT,
    meta_json     TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id            INTEGER NOT NULL REFERENCES cycles(id),
    timestamp           TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    action              TEXT NOT NULL,
    size_pct            REAL NOT NULL,
    stop_loss_pct       REAL NOT NULL,
    take_profit_pct     REAL NOT NULL,
    reasoning           TEXT NOT NULL,
    signals_json        TEXT NOT NULL,
    model               TEXT NOT NULL,
    decision_hash       TEXT NOT NULL,
    passed              INTEGER NOT NULL,
    clamped             INTEGER NOT NULL,
    final_size_pct      REAL NOT NULL,
    final_stop_loss_pct REAL NOT NULL,
    risk_reasons_json   TEXT NOT NULL,
    risk_checks_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_cycle ON decisions(cycle_id);
CREATE INDEX IF NOT EXISTS ix_decisions_hash  ON decisions(decision_hash);

CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER NOT NULL REFERENCES decisions(id),
    filled_at       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        REAL NOT NULL,
    fill_price      REAL NOT NULL,
    fee_usd         REAL NOT NULL,
    venue           TEXT NOT NULL,
    venue_order_id  TEXT,
    fill_hash       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fills_decision ON fills(decision_id);
CREATE INDEX IF NOT EXISTS ix_fills_hash     ON fills(fill_hash);

CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER NOT NULL REFERENCES decisions(id),
    fill_id         INTEGER REFERENCES fills(id),
    artifact_hash   TEXT NOT NULL,
    decision_hash   TEXT NOT NULL,
    trade_hash      TEXT,
    pre_state_hash  TEXT NOT NULL,
    post_state_hash TEXT NOT NULL,
    reasoning_uri   TEXT NOT NULL,
    tx_hash         TEXT,
    via             TEXT,                   -- 'agent_artifacts' | 'validation_registry' | 'dry_run'
    block_number    INTEGER,
    status          TEXT NOT NULL,          -- 'ok' | 'unverified'
    error           TEXT,
    submitted_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_artifacts_decision ON artifacts(decision_id);
CREATE INDEX IF NOT EXISTS ix_artifacts_status   ON artifacts(status);
CREATE INDEX IF NOT EXISTS ix_artifacts_hash     ON artifacts(artifact_hash);
"""


@dataclass(frozen=True)
class DecisionRow:
    id: int
    decision: Decision
    risked: RiskedDecision
    decision_hash: str


@dataclass(frozen=True)
class FillRow:
    id: int
    decision_id: int
    fill: Fill
    fill_hash: str


@dataclass(frozen=True)
class ArtifactRow:
    id: int
    decision_id: int
    fill_id: int | None
    artifact_hash: str
    tx_hash: str | None
    status: str
    via: str | None


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now_iso() -> str:
    return _iso(utcnow())


class Store:
    """Thin SQLite wrapper. Synchronous — the loop awaits everything else,
    but DB writes are <1ms and don't justify an async driver."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            self._conn.execute("BEGIN")
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ---------------- cycles ----------------

    def start_cycle(self, meta: dict[str, Any] | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO cycles (started_at, status, meta_json) VALUES (?, 'running', ?)",
            (_now_iso(), json.dumps(meta or {}, sort_keys=True)),
        )
        return int(cur.lastrowid)

    def finish_cycle(self, cycle_id: int, *, error: str | None = None) -> None:
        self._conn.execute(
            "UPDATE cycles SET finished_at = ?, status = ?, error = ? WHERE id = ?",
            (_now_iso(), "error" if error else "ok", error, cycle_id),
        )

    # ---------------- decisions ----------------

    def record_decision(self, cycle_id: int, risked: RiskedDecision) -> DecisionRow:
        d = risked.decision
        decision_hash = canonical_hash(d)
        cur = self._conn.execute(
            """
            INSERT INTO decisions (
                cycle_id, timestamp, symbol, action, size_pct, stop_loss_pct,
                take_profit_pct, reasoning, signals_json, model, decision_hash,
                passed, clamped, final_size_pct, final_stop_loss_pct,
                risk_reasons_json, risk_checks_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                cycle_id,
                _iso(d.timestamp),
                d.symbol,
                d.action,
                d.size_pct,
                d.stop_loss_pct,
                d.take_profit_pct,
                d.reasoning,
                json.dumps(d.signals, sort_keys=True),
                d.model,
                decision_hash,
                int(risked.passed),
                int(risked.clamped),
                risked.final_size_pct,
                risked.final_stop_loss_pct,
                json.dumps(risked.reasons),
                json.dumps(risked.risk_checks, sort_keys=True),
            ),
        )
        return DecisionRow(
            id=int(cur.lastrowid),
            decision=d,
            risked=risked,
            decision_hash=decision_hash,
        )

    # ---------------- fills ----------------

    def record_fill(self, decision_id: int, fill: Fill) -> FillRow:
        fill_hash = canonical_hash(fill)
        cur = self._conn.execute(
            """
            INSERT INTO fills (
                decision_id, filled_at, symbol, side, quantity, fill_price,
                fee_usd, venue, venue_order_id, fill_hash
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                decision_id,
                _iso(fill.filled_at),
                fill.order.symbol,
                fill.order.side,
                fill.order.quantity,
                fill.fill_price,
                fill.fee_usd,
                fill.venue,
                fill.venue_order_id,
                fill_hash,
            ),
        )
        return FillRow(
            id=int(cur.lastrowid),
            decision_id=decision_id,
            fill=fill,
            fill_hash=fill_hash,
        )

    # ---------------- artifacts ----------------

    def record_artifact(
        self,
        *,
        decision_id: int,
        fill_id: int | None,
        artifact: ValidationArtifact,
        artifact_hash: str,
        tx_hash: str | None,
        via: str | None,
        block_number: int | None,
        status: str,
        error: str | None = None,
    ) -> ArtifactRow:
        if status not in ("ok", "unverified"):
            raise ValueError(f"artifact status must be ok|unverified, got {status!r}")
        cur = self._conn.execute(
            """
            INSERT INTO artifacts (
                decision_id, fill_id, artifact_hash, decision_hash, trade_hash,
                pre_state_hash, post_state_hash, reasoning_uri, tx_hash, via,
                block_number, status, error, submitted_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                decision_id,
                fill_id,
                artifact_hash,
                artifact.decision_hash,
                artifact.trade_hash,
                artifact.pre_state_hash,
                artifact.post_state_hash,
                artifact.reasoning_uri,
                tx_hash,
                via,
                block_number,
                status,
                error,
                _now_iso(),
            ),
        )
        return ArtifactRow(
            id=int(cur.lastrowid),
            decision_id=decision_id,
            fill_id=fill_id,
            artifact_hash=artifact_hash,
            tx_hash=tx_hash,
            status=status,
            via=via,
        )

    def mark_artifact_verified(
        self, artifact_id: int, *, tx_hash: str, via: str, block_number: int | None
    ) -> None:
        self._conn.execute(
            """UPDATE artifacts
               SET status='ok', tx_hash=?, via=?, block_number=?, error=NULL
               WHERE id=?""",
            (tx_hash, via, block_number, artifact_id),
        )

    def list_unverified_artifacts(self) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM artifacts WHERE status='unverified' ORDER BY id"
        )
        return list(cur.fetchall())

    # ---------------- counts (for tests / ops) ----------------

    def count(self, table: str) -> int:
        if table not in {"cycles", "decisions", "fills", "artifacts"}:
            raise ValueError(f"unknown table {table!r}")
        cur = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])


def snapshot_state_hash(snapshot: PortfolioSnapshot) -> str:
    """Canonical hash of a PortfolioSnapshot.

    Used as the pre/post state pair on every ValidationArtifact. Changing
    this changes every artifact hash — keep it stable.
    """
    return canonical_hash(snapshot)
