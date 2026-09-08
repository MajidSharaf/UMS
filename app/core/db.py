"""SQLite storage for runs, chain runs and manual scores.

Prompts and objectives are file-backed (see prompts.py / objectives.py) so
they diff cleanly in git and carry natural version history. Runs are
generated artifacts, not source content, so they live in SQLite plus a raw
JSON blob per run under experiments/runs/.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
import uuid
from typing import Any, Iterator, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    objective_id TEXT,
    stage TEXT NOT NULL,
    prompt_stage TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    model TEXT NOT NULL,
    temperature REAL NOT NULL,
    input_ref TEXT NOT NULL,
    input_summary TEXT,
    system_prompt TEXT,
    user_prompt TEXT,
    raw_output TEXT,
    parsed_json TEXT,
    is_valid_json INTEGER NOT NULL,
    schema_errors TEXT,
    error TEXT,
    latency_ms REAL,
    chain_run_id TEXT,
    chain_sequence INTEGER
);

CREATE TABLE IF NOT EXISTS scores (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(id),
    accuracy INTEGER,
    completeness INTEGER,
    hallucination INTEGER,
    structure INTEGER,
    overall INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS chain_runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    objective_id TEXT,
    label TEXT,
    input_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_stage_input ON runs(stage, input_ref);
CREATE INDEX IF NOT EXISTS idx_runs_chain ON runs(chain_run_id);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
"""


@contextlib.contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    # Longer than the 5s default: gives OneDrive (which may briefly lock the
    # file mid-sync) or a concurrently-open Streamlit session room to finish
    # before we give up on a write - matters more over an unattended run.
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def insert_run(row: dict[str, Any]) -> str:
    run_id = row.get("id") or new_id()
    row = dict(row)
    row["id"] = run_id
    row.setdefault("created_at", time.time())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                id, created_at, objective_id, stage, prompt_stage, prompt_name,
                model, temperature, input_ref, input_summary,
                system_prompt, user_prompt, raw_output, parsed_json,
                is_valid_json, schema_errors, error, latency_ms,
                chain_run_id, chain_sequence
            ) VALUES (
                :id, :created_at, :objective_id, :stage, :prompt_stage, :prompt_name,
                :model, :temperature, :input_ref, :input_summary,
                :system_prompt, :user_prompt, :raw_output, :parsed_json,
                :is_valid_json, :schema_errors, :error, :latency_ms,
                :chain_run_id, :chain_sequence
            )
            """,
            {
                "objective_id": None,
                "input_summary": None,
                "raw_output": None,
                "parsed_json": None,
                "schema_errors": None,
                "error": None,
                "latency_ms": None,
                "chain_run_id": None,
                "chain_sequence": None,
                **row,
            },
        )
    return run_id


def list_runs(stage: Optional[str] = None, input_ref: Optional[str] = None,
              chain_run_id: Optional[str] = None) -> list[sqlite3.Row]:
    query = "SELECT * FROM runs WHERE 1=1"
    params: list[Any] = []
    if stage:
        query += " AND stage = ?"
        params.append(stage)
    if input_ref:
        query += " AND input_ref = ?"
        params.append(input_ref)
    if chain_run_id:
        query += " AND chain_run_id = ?"
        params.append(chain_run_id)
    query += " ORDER BY created_at DESC"
    with connect() as conn:
        return conn.execute(query, params).fetchall()


def get_run(run_id: str) -> Optional[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def insert_chain_run(objective_id: Optional[str], label: str, input_summary: str) -> str:
    chain_id = new_id()
    with connect() as conn:
        conn.execute(
            "INSERT INTO chain_runs (id, created_at, objective_id, label, input_summary) "
            "VALUES (?, ?, ?, ?, ?)",
            (chain_id, time.time(), objective_id, label, input_summary),
        )
    return chain_id


def list_chain_runs() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM chain_runs ORDER BY created_at DESC").fetchall()


def upsert_score(run_id: str, accuracy: int, completeness: int, hallucination: int,
                  structure: int, overall: int, notes: str) -> str:
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM scores WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE scores SET accuracy=?, completeness=?, hallucination=?,
                   structure=?, overall=?, notes=?, created_at=? WHERE id=?""",
                (accuracy, completeness, hallucination, structure, overall, notes,
                 time.time(), existing["id"]),
            )
            return existing["id"]
        score_id = new_id()
        conn.execute(
            """INSERT INTO scores (id, created_at, run_id, accuracy, completeness,
               hallucination, structure, overall, notes) VALUES (?,?,?,?,?,?,?,?,?)""",
            (score_id, time.time(), run_id, accuracy, completeness, hallucination,
             structure, overall, notes),
        )
    return score_id


def get_score(run_id: str) -> Optional[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM scores WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()


def scores_for_stage(stage: str) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            """
            SELECT r.id as run_id, r.stage, r.prompt_name, r.model,
                   r.input_ref, s.accuracy, s.completeness, s.hallucination,
                   s.structure, s.overall
            FROM runs r JOIN scores s ON s.run_id = r.id
            WHERE r.stage = ?
            """,
            (stage,),
        ).fetchall()


def save_raw_output(run_id: str, payload: dict[str, Any]) -> None:
    path = config.RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
