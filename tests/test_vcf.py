#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test VCF/VCT threat-space search."""

from __future__ import annotations

import time

import numpy as np
import pytest

from config import Player, VCF_ENABLED
from core.caro_env import CaroEnv
from ai.minimax_agent import MinimaxAgent
from ai.vcf import find_forced_defenses, find_threat_moves, vcf_search


def _three_open_four_setup() -> CaroEnv:
    """X có XXX với hai đầu trống — tạo tứ mở rồi thắng sau khi O chặn một đầu."""
    env = CaroEnv(size=10)
    env.board[5, 1] = Player.X
    env.board[5, 2] = Player.X
    env.board[5, 3] = Player.X
    env.current_player = Player.X
    env._move_count = 3  # noqa: SLF001
    env.last_move = (5, 3)
    return env


def test_vcf_finds_forcing_win_from_three_in_row() -> None:
    """VCF tìm chuỗi thắng khi có 3 quân liên tiếp + không gian."""
    env = _three_open_four_setup()
    chain = vcf_search(env, Player.X, max_depth=6)
    assert chain is not None
    assert chain[0] == (5, 4)


def test_vcf_returns_none_without_forcing_line() -> None:
    """Không có chuỗi ép buộc → None."""
    env = CaroEnv(size=10)
    env.board[5, 5] = Player.X
    env.board[6, 6] = Player.O
    env.current_player = Player.X
    env._move_count = 2  # noqa: SLF001
    env.last_move = (6, 6)

    assert vcf_search(env, Player.X, max_depth=8) is None


def test_vcf_defense_blocks_opponent_forcing_line() -> None:
    """Minimax chặn nước đầu chuỗi VCF của đối thủ."""
    env = _three_open_four_setup()
    env.current_player = Player.O

    agent = MinimaxAgent(depth=1, time_budget=None)
    agent.search_only = False
    move = agent.get_move(env)
    assert move in {(5, 4), (5, 0)}


def test_vcf_search_preserves_board() -> None:
    """Bàn cờ không đổi sau vcf_search (push/pop cân bằng)."""
    env = _three_open_four_setup()
    before = env.board.copy()
    before_player = env.current_player
    before_done = env.done

    vcf_search(env, Player.X, max_depth=6)

    assert np.array_equal(env.board, before)
    assert env.current_player is before_player
    assert env.done is before_done
    assert not env._undo_stack  # noqa: SLF001


def test_vcf_search_midgame_under_200ms() -> None:
    """vcf_search trên bàn 15×15 giữa ván < 200 ms."""
    env = CaroEnv(size=15)
    rng = np.random.default_rng(42)
    center = env.size // 2
    moves = [(center, center)]
    for _ in range(21):
        r = center + int(rng.integers(-4, 5))
        c = center + int(rng.integers(-4, 5))
        r = max(0, min(env.size - 1, r))
        c = max(0, min(env.size - 1, c))
        if env.board[r, c] != Player.EMPTY:
            continue
        env.board[r, c] = env.current_player
        env.last_move = (r, c)
        env._move_count += 1  # noqa: SLF001
        env.current_player = env.current_player.opponent
        moves.append((r, c))

    env.current_player = Player.X if len(moves) % 2 == 0 else Player.O

    t0 = time.perf_counter()
    vcf_search(env, env.current_player, max_depth=10, radius=2)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0


def test_find_threat_moves_open_four() -> None:
    env = _three_open_four_setup()
    threats = find_threat_moves(env, Player.X)
    assert (5, 4) in threats


def test_find_forced_defenses_after_open_four() -> None:
    env = _three_open_four_setup()
    env.push((5, 4))
    try:
        blocks = find_forced_defenses(env, (5, 4), Player.O)
        assert set(blocks) == {(5, 0), (5, 5)}
    finally:
        env.pop()


@pytest.mark.skipif(not VCF_ENABLED, reason="VCF tắt trong config")
def test_vcf_enabled_in_config() -> None:
    assert VCF_ENABLED is True
