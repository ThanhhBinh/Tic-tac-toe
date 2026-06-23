#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lưu kết quả benchmark so sánh AI vào SQLite — tránh chạy lại cùng cấu hình."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Tăng version khi đổi logic benchmark / bộ TH để không dùng cache cũ.
BENCHMARK_CACHE_VERSION = 2

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "models" / "benchmark_cache.db"


class BenchmarkCache:
    """Cache kết quả ``run_benchmark`` theo khóa cấu hình."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._lock = threading.Lock()
        self._init_db()

    @staticmethod
    def make_key(
        scenario_set: str,
        difficulty: str,
        board_size: int,
        double_end_block_rule: bool,
        ai_aggressive: bool,
    ) -> str:
        """Khóa duy nhất cho một tổ hợp tham số benchmark."""
        return (
            f"v{BENCHMARK_CACHE_VERSION}__{scenario_set}__{difficulty.upper()}__"
            f"{board_size}__{int(double_end_block_rule)}__{int(ai_aggressive)}"
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    cache_key TEXT PRIMARY KEY,
                    scenario_set TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    board_size INTEGER NOT NULL,
                    double_end_block_rule INTEGER NOT NULL,
                    ai_aggressive INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    run_elapsed_ms REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Trả payload benchmark đã lưu hoặc None."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT result_json, created_at, run_elapsed_ms FROM benchmark_runs WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["result_json"])
        data["from_cache"] = True
        data["cache_key"] = cache_key
        data["cached_at"] = row["created_at"]
        if row["run_elapsed_ms"] is not None:
            data["run_elapsed_ms"] = row["run_elapsed_ms"]
        return data

    def save(
        self,
        cache_key: str,
        *,
        scenario_set: str,
        difficulty: str,
        board_size: int,
        double_end_block_rule: bool,
        ai_aggressive: bool,
        result: dict[str, Any],
        run_elapsed_ms: float | None = None,
    ) -> None:
        """Ghi đè kết quả cho ``cache_key`` (INSERT OR REPLACE)."""
        payload = dict(result)
        payload.pop("from_cache", None)
        payload.pop("cache_key", None)
        payload.pop("cached_at", None)
        payload.pop("run_elapsed_ms", None)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_runs (
                    cache_key, scenario_set, difficulty, board_size,
                    double_end_block_rule, ai_aggressive,
                    result_json, run_elapsed_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    scenario_set,
                    difficulty.upper(),
                    board_size,
                    int(double_end_block_rule),
                    int(ai_aggressive),
                    json.dumps(payload, ensure_ascii=False),
                    run_elapsed_ms,
                    now,
                ),
            )
            conn.commit()

    def list_keys(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT cache_key FROM benchmark_runs ORDER BY created_at DESC"
            ).fetchall()
        return [str(r["cache_key"]) for r in rows]
