#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chạy benchmark và lưu vào models/benchmark_cache.db (bỏ qua nếu đã có)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai.benchmark import run_benchmark
from config import Difficulty, TacticalConfig
from web.benchmark_store import BenchmarkCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy benchmark và lưu SQLite cache.")
    parser.add_argument("--scenario-set", default="all", choices=("basic", "advanced", "all"))
    parser.add_argument("--difficulty", default="EXPERT")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--force", action="store_true", help="Ghi đè bản đã lưu")
    args = parser.parse_args()

    difficulty = Difficulty[args.difficulty.upper()]
    tactical = TacticalConfig(double_end_block_rule=True, aggressive=True, threat_warnings=True)
    cache = BenchmarkCache()
    key = BenchmarkCache.make_key(
        args.scenario_set,
        args.difficulty,
        args.board_size,
        True,
        True,
    )

    if not args.force:
        hit = cache.get(key)
        if hit is not None:
            w = hit["winner"]
            print(f"Đã có trong DB: {key}")
            print(f"Winner: {w['label']} ({hit['summary'][w['key']]['composite_score']})")
            return

    print(f"Chạy benchmark {args.scenario_set} / {args.difficulty} / {args.board_size}…")
    t0 = time.perf_counter()
    result = run_benchmark(
        difficulty=difficulty,
        board_size=args.board_size,
        tactical=tactical,
        scenario_set=args.scenario_set,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    cache.save(
        key,
        scenario_set=args.scenario_set,
        difficulty=args.difficulty,
        board_size=args.board_size,
        double_end_block_rule=True,
        ai_aggressive=True,
        result=result,
        run_elapsed_ms=elapsed_ms,
    )
    w = result["winner"]
    print(f"Xong {elapsed_ms / 1000:.1f}s — winner {w['label']} ({result['summary'][w['key']]['composite_score']})")
    print(f"Đã lưu: {key}")


if __name__ == "__main__":
    main()
