"""Flat, portable CSV log of every run — independent of the SQLite DB, so
there's always a plain file to open in Excel/Sheets for a report even if
you never touch the Streamlit Compare page."""
from __future__ import annotations

import csv
import time
from pathlib import Path

from . import config

LOG_PATH = config.EXPERIMENTS_DIR / "results_log.csv"

FIELDNAMES = [
    "timestamp", "run_id", "stage", "prompt_name", "model",
    "temperature", "response_format", "context_variant", "input_ref",
    "is_valid_json", "schema_errors", "latency_ms", "output_excerpt",
    "chain_run_id", "chain_sequence",
]


def append(row: dict) -> None:
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **row})
