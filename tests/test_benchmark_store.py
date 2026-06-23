#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test lưu cache benchmark SQLite."""

from __future__ import annotations

from pathlib import Path

import pytest

from web.benchmark_store import BenchmarkCache


@pytest.fixture
def cache(tmp_path: Path) -> BenchmarkCache:
    return BenchmarkCache(tmp_path / "bench.db")


def test_make_key_stable() -> None:
    k1 = BenchmarkCache.make_key("all", "EXPERT", 15, True, True)
    k2 = BenchmarkCache.make_key("all", "expert", 15, True, True)
    assert k1 == k2
    assert k1.startswith("v")


def test_save_and_get(cache: BenchmarkCache) -> None:
    key = BenchmarkCache.make_key("basic", "MEDIUM", 15, True, True)
    payload = {"scenario_count": 10, "winner": {"key": "minimax"}}
    cache.save(
        key,
        scenario_set="basic",
        difficulty="MEDIUM",
        board_size=15,
        double_end_block_rule=True,
        ai_aggressive=True,
        result=payload,
        run_elapsed_ms=1234.5,
    )
    hit = cache.get(key)
    assert hit is not None
    assert hit["from_cache"] is True
    assert hit["cache_key"] == key
    assert hit["scenario_count"] == 10
    assert hit["run_elapsed_ms"] == 1234.5
    assert hit["cached_at"]


def test_get_missing_returns_none(cache: BenchmarkCache) -> None:
    assert cache.get("missing") is None


def test_overwrite(cache: BenchmarkCache) -> None:
    key = BenchmarkCache.make_key("basic", "EASY", 10, True, True)
    cache.save(
        key,
        scenario_set="basic",
        difficulty="EASY",
        board_size=10,
        double_end_block_rule=True,
        ai_aggressive=True,
        result={"n": 1},
    )
    cache.save(
        key,
        scenario_set="basic",
        difficulty="EASY",
        board_size=10,
        double_end_block_rule=True,
        ai_aggressive=True,
        result={"n": 2},
    )
    hit = cache.get(key)
    assert hit is not None
    assert hit["n"] == 2
