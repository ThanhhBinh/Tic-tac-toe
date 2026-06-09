#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đánh giá sức mạnh agent bằng cách cho đấu thử nhiều ván.

Dùng bởi skill ``evaluate_model`` trong ``agent_tools.py`` và có thể gọi trực
tiếp từ script huấn luyện.
"""

from __future__ import annotations

from ai.base_agent import Agent
from ai.factory import create_agent
from config import DEFAULT_BOARD_SIZE, AIType, Difficulty, Player
from core.caro_env import CaroEnv


def _resolve_agent(name: str, board_size: int) -> Agent:
    """Tạo agent từ tên chuỗi (vd: 'minimax', 'dqn', 'hybrid', 'random').

    Args:
        name: Tên loại agent (không phân biệt hoa thường).
        board_size: Kích thước bàn cờ.

    Returns:
        Thể hiện Agent tương ứng.
    """
    key = name.strip().lower()
    mapping: dict[str, AIType] = {
        "minimax": AIType.MINIMAX,
        "dqn": AIType.DQN,
        "hybrid": AIType.HYBRID,
        "random": AIType.DQN,  # fallback tạm — random qua factory cũ
    }
    if key == "random":
        from ai.random_agent import RandomAgent

        return RandomAgent()
    ai_type = mapping.get(key, AIType.MINIMAX)
    return create_agent(ai_type, Difficulty.MEDIUM, board_size=board_size)


def play_game(agent_a: Agent, agent_b: Agent, board_size: int = DEFAULT_BOARD_SIZE) -> Player | None:
    """Đấu một ván: agent_a đi quân X, agent_b đi quân O.

    Args:
        agent_a: Tác nhân đi trước (X).
        agent_b: Tác nhân đi sau (O).
        board_size: Kích thước bàn cờ.

    Returns:
        Người thắng hoặc None nếu hòa.
    """
    env = CaroEnv(size=board_size)
    env.reset()
    agents: dict[Player, Agent] = {Player.X: agent_a, Player.O: agent_b}

    max_moves = board_size * board_size + 1
    moves = 0
    while not env.done and moves < max_moves:
        agent = agents[env.current_player]
        move = agent.get_move(env)
        env.step(move)
        moves += 1
    return env.winner


def play_match(
    agent_a: str,
    agent_b: str,
    num_games: int = 20,
    board_size: int = DEFAULT_BOARD_SIZE,
) -> dict[str, float | int | str]:
    """Đấu thử nhiều ván và thống kê tỷ lệ thắng.

    Args:
        agent_a: Tên agent phe X (vd: 'hybrid', 'minimax', 'dqn').
        agent_b: Tên agent phe O.
        num_games: Số ván đấu.
        board_size: Kích thước bàn cờ.

    Returns:
        Dict chứa win_rate_a, wins_a, wins_b, draws, games.
    """
    a = _resolve_agent(agent_a, board_size)
    b = _resolve_agent(agent_b, board_size)

    wins_a = wins_b = draws = 0
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(a, b, board_size)
            if winner is Player.X:
                wins_a += 1
            elif winner is Player.O:
                wins_b += 1
            else:
                draws += 1
        else:
            winner = play_game(b, a, board_size)
            if winner is Player.X:
                wins_b += 1
            elif winner is Player.O:
                wins_a += 1
            else:
                draws += 1

    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "games": num_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / num_games if num_games else 0.0,
        "board_size": board_size,
    }
