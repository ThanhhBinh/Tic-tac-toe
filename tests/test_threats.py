#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test phân tích đe dọa và luật chặn 2 đầu."""

from __future__ import annotations

from config import Player, TacticalConfig
from core.caro_env import CaroEnv
from ai.heuristic import find_tactical_move
from ai.threats import analyze_threats


def test_canh_bao_thang_ngay() -> None:
    """Phải báo ô thắng ngay cho người chơi."""
    env = CaroEnv(size=10)
    env.board[5, 1] = Player.X
    env.board[5, 2] = Player.X
    env.board[5, 3] = Player.X
    env.board[5, 4] = Player.X
    env.current_player = Player.X
    env._move_count = 4  # noqa: SLF001

    analysis = analyze_threats(env, Player.X)
    assert (5, 5) in analysis.win_moves or (5, 0) in analysis.win_moves
    assert analysis.message


def test_ai_tan_cong_truoc_chan_tam_mo() -> None:
    """Chế độ aggressive: tấn công tứ mở trước chặn tam mở nhẹ."""
    env = CaroEnv(size=10)
    env.board[5, 1] = Player.X
    env.board[5, 2] = Player.X
    env.board[5, 3] = Player.X
    env.current_player = Player.X
    env._move_count = 3  # noqa: SLF001

    move = find_tactical_move(
        env,
        Player.X,
        config=TacticalConfig(aggressive=True, double_end_block_rule=True),
    )
    assert move == (5, 4)


def test_double_end_canh_bao() -> None:
    """Luật chặn 2 đầu bật → highlight hai đầu tam mở đối thủ."""
    env = CaroEnv(size=10)
    env.board[5, 2] = Player.O
    env.board[5, 3] = Player.O
    env.board[5, 4] = Player.O
    env.current_player = Player.X
    env._move_count = 3  # noqa: SLF001

    on = analyze_threats(
        env, Player.X, config=TacticalConfig(double_end_block_rule=True)
    )
    off = analyze_threats(
        env, Player.X, config=TacticalConfig(double_end_block_rule=False)
    )
    assert len(on.double_end_blocks) >= 1
    assert len(on.threat_stones) >= 3
    assert "Chặn" in on.message or "đe dọa" in on.message.lower() or on.message
    assert len(off.double_end_blocks) == 0
