#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test cho MinimaxAgent và heuristic.

Kiểm tra: nước đi hợp lệ, bắt thắng ngay, chặn thua bắt buộc, và ưu tiên
ô trung tâm trên bàn trống.
"""

from __future__ import annotations

import numpy as np

from config import Difficulty, Player
from core.caro_env import CaroEnv
from ai.heuristic import find_blocking_move, find_winning_move
from ai.minimax_agent import MinimaxAgent


def _fill_board(env: CaroEnv, moves: list[tuple[int, int]]) -> None:
    """Đặt lần lượt danh sách nước đi (bỏ qua kiểm tra lượt — dùng cho setup)."""
    for row, col in moves:
        env.board[row, col] = env.current_player
        env._move_count += 1  # noqa: SLF001 — setup test cần tăng bộ đếm
        env.current_player = env.current_player.opponent
    if moves:
        env.last_move = moves[-1]


def test_minimax_tra_ve_nuoc_hop_le() -> None:
    """Mọi nước do Minimax chọn phải hợp lệ trên bàn trống."""
    env = CaroEnv(size=10)
    agent = MinimaxAgent(depth=2)
    move = agent.get_move(env)
    assert env.is_legal(move)


def test_minimax_uu_tien_o_trung_tam_ban_trong() -> None:
    """Bàn trống -> nước đầu nên gần trung tâm (ô 5,5 trên bàn 10x10)."""
    env = CaroEnv(size=10)
    agent = MinimaxAgent(depth=1)
    row, col = agent.get_move(env)
    assert (row, col) == (5, 5)


def test_tim_nuoc_thang_ngay() -> None:
    """Heuristic phát hiện nước thắng ngay (4 quân + 1 ô trống)."""
    env = CaroEnv(size=10)
    # X có 4 quân ngang tại hàng 5, cột 0-3; lượt X.
    _fill_board(
        env,
        [(5, 0), (0, 0), (5, 1), (0, 1), (5, 2), (0, 2), (5, 3), (0, 3)],
    )
    env.current_player = Player.X
    win = find_winning_move(env, Player.X)
    assert win == (5, 4)


def test_tim_nuoc_chan_thua() -> None:
    """Heuristic phát hiện nước chặn đối thủ thắng ngay."""
    env = CaroEnv(size=10)
    # O có 4 quân ngang; lượt X phải chặn.
    _fill_board(
        env,
        [(5, 0), (0, 0), (5, 1), (0, 1), (5, 2), (0, 2), (5, 3), (0, 3)],
    )
    env.current_player = Player.X
    # Giả lập O sắp thắng: đổi quân hàng 5 thành O (trừ ô trống).
    env.board[5, 0] = Player.O
    env.board[5, 1] = Player.O
    env.board[5, 2] = Player.O
    env.board[5, 3] = Player.O
    block = find_blocking_move(env, Player.X)
    assert block == (5, 4)


def test_minimax_chon_thang_khi_co_co_hoi() -> None:
    """Minimax phải chọn nước thắng ngay, không bỏ lỡ."""
    env = CaroEnv(size=10)
    env.board[5, 0] = Player.X
    env.board[5, 1] = Player.X
    env.board[5, 2] = Player.X
    env.board[5, 3] = Player.X
    env.current_player = Player.X
    env._move_count = 4  # noqa: SLF001
    env.last_move = (5, 3)

    agent = MinimaxAgent(depth=1)
    assert agent.get_move(env) == (5, 4)


def test_minimax_chan_khi_bi_doa() -> None:
    """Minimax phải chặn khi đối thủ có 4 quân mở."""
    env = CaroEnv(size=10)
    env.board[3, 3] = Player.O
    env.board[3, 4] = Player.O
    env.board[3, 5] = Player.O
    env.board[3, 6] = Player.O
    env.current_player = Player.X
    env._move_count = 4  # noqa: SLF001
    env.last_move = (3, 6)

    agent = MinimaxAgent(depth=1)
    move = agent.get_move(env)
    assert move in {(3, 2), (3, 7)}


def test_minimax_depth_tu_difficulty() -> None:
    """Factory difficulty map đúng sang độ sâu."""
    agent = MinimaxAgent.from_difficulty(Difficulty.HARD)
    assert agent.depth == 3


def test_heuristic_diem_cao_hon_khi_co_bon_mo() -> None:
    """Bàn có bốn mở phải có điểm cao hơn bàn trống."""
    from ai.heuristic import evaluate_board

    empty = np.zeros((10, 10), dtype=np.int8)
    with_four = empty.copy()
    with_four[5, 1] = Player.X
    with_four[5, 2] = Player.X
    with_four[5, 3] = Player.X
    with_four[5, 4] = Player.X

    assert evaluate_board(with_four, Player.X) > evaluate_board(empty, Player.X)
