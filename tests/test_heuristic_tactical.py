#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test luật chiến thuật open-four và Hybrid khi chưa train DQN."""

from __future__ import annotations

import pytest
pytest.importorskip("torch")

from config import Difficulty, Player
from core.caro_env import CaroEnv
from ai.heuristic import (
    find_open_four_block,
    find_open_four_move,
    find_tactical_move,
)
from ai.hybrid_agent import HybridAgent


def test_chan_tu_mo_doi_thu() -> None:
    """Phải chặn ô đối thủ dùng để tạo 4 mở (trước khi thành 5)."""
    env = CaroEnv(size=10)
    # O: O O O _ trên hàng 5 (cột 1-3), ô (5,4) trống → O đánh (5,4) tạo tứ mở.
    env.board[5, 1] = Player.O
    env.board[5, 2] = Player.O
    env.board[5, 3] = Player.O
    env.current_player = Player.X
    env._move_count = 3  # noqa: SLF001

    block = find_open_four_block(env, Player.X)
    assert block == (5, 4)


def test_tao_tu_mo_tan_cong() -> None:
    """X tạo tứ mở khi có 3 quân liên tiếp và ô mở đủ."""
    env = CaroEnv(size=10)
    env.board[5, 1] = Player.X
    env.board[5, 2] = Player.X
    env.board[5, 3] = Player.X
    env.current_player = Player.X
    env._move_count = 3  # noqa: SLF001

    attack = find_open_four_move(env, Player.X)
    assert attack == (5, 4)


def test_hybrid_khong_dqn_dung_heuristic_la() -> None:
    """Chưa train DQN → node lá không gọi mạng, dùng heuristic."""
    env = CaroEnv(size=10)
    agent = HybridAgent.from_difficulty(Difficulty.EXPERT, board_size=10)
    assert not agent.dqn._model_loaded

    calls = 0
    original = agent.dqn._predict_q_numpy

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    agent.dqn._predict_q_numpy = spy  # type: ignore[method-assign]
    agent.get_move(env)
    assert calls == 0


def test_hybrid_expert_co_tactical() -> None:
    """Expert Hybrid vẫn chặn thắng ngay qua find_tactical_move."""
    env = CaroEnv(size=10)
    env.board[3, 3] = Player.O
    env.board[3, 4] = Player.O
    env.board[3, 5] = Player.O
    env.board[3, 6] = Player.O
    env.current_player = Player.X
    env._move_count = 4  # noqa: SLF001
    env.last_move = (3, 6)

    move = find_tactical_move(env, Player.X)
    assert move in {(3, 2), (3, 7)}
