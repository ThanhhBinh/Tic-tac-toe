#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test kích thước bàn và benchmark."""

from __future__ import annotations

import pytest

from ai.benchmark import AGENT_KEYS, BENCHMARK_SCENARIOS, run_benchmark
from config import BOARD_SIZES, Difficulty, create_caro_env, win_length_for_board


def test_board_sizes_config() -> None:
    assert BOARD_SIZES == (3, 5, 7, 10, 15)


@pytest.mark.parametrize("size,expected", [(3, 3), (5, 5), (7, 5), (10, 5), (15, 5)])
def test_win_length_for_board(size: int, expected: int) -> None:
    assert win_length_for_board(size) == expected


@pytest.mark.parametrize("size", BOARD_SIZES)
def test_create_caro_env_all_sizes(size: int) -> None:
    env = create_caro_env(size)
    assert env.size == size
    assert env.win_length == win_length_for_board(size)


def test_benchmark_co_10_th() -> None:
    assert len(BENCHMARK_SCENARIOS) == 10


@pytest.mark.parametrize("board_size", [3, 5, 7, 10, 15])
def test_run_benchmark_all_board_sizes(board_size: int) -> None:
    """Benchmark không crash trên mọi kích thước bàn."""
    result = run_benchmark(difficulty=Difficulty.EASY, board_size=board_size)
    assert result["scenario_count"] == 10
    assert result["board_size"] == board_size
    assert set(result["summary"].keys()) == set(AGENT_KEYS)


def test_run_benchmark_structure() -> None:
    result = run_benchmark(difficulty=Difficulty.EASY, board_size=10)
    assert len(result["scenarios"]) == 10
    for sc in result["scenarios"]:
        assert sc["board_size"] == 10
        assert set(sc["agents"].keys()) == set(AGENT_KEYS)


def test_tactical_cases_co_dap_an() -> None:
    tactical = [s for s in BENCHMARK_SCENARIOS if s.expected is not None]
    assert len(tactical) == 3  # TH01-03: tactical; TH04-10: strategic (no expected)
    assert BENCHMARK_SCENARIOS[-1].expected is None
