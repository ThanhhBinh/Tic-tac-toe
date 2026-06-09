#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test ước lượng xác suất thắng heuristic."""

from __future__ import annotations

from config import Player
from core.caro_env import CaroEnv
from ai.win_probability import estimate_win_probability
from ai.minimax_agent import MinimaxAgent


def test_heuristic_win_prob_gan_50_ban_trong() -> None:
    """Bàn trống → xác suất gần 50%."""
    env = CaroEnv(size=10)
    prob = estimate_win_probability(env, Player.X)
    assert 0.45 <= prob <= 0.55


def test_heuristic_win_prob_thang() -> None:
    """Đã thắng → 100%."""
    env = CaroEnv(size=10)
    env.board[5, 1] = Player.X
    env.board[5, 2] = Player.X
    env.board[5, 3] = Player.X
    env.board[5, 4] = Player.X
    env.board[5, 5] = Player.X
    env.done = True
    env.winner = Player.X
    assert estimate_win_probability(env, Player.X) == 1.0


def test_minimax_co_win_probability() -> None:
    """Minimax luôn trả heuristic win prob."""
    env = CaroEnv(size=10)
    agent = MinimaxAgent(depth=2)
    prob = agent.get_win_probability(env)
    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_web_session_co_win_prob() -> None:
    """API web trả win_probability khi chưa train DQN."""
    from config import AIType, Difficulty, GameMode
    from web.session import GameSession, SessionSettings

    session = GameSession(
        SessionSettings(
            mode=GameMode.PVA,
            ai_type=AIType.HYBRID,
            difficulty=Difficulty.EASY,
            board_size=10,
        )
    )
    data = session.to_dict()
    assert data["win_probability"] is not None
    assert data["win_probability_source"] == "heuristic"
