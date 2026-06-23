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


def play_match_agents(
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
    board_size: int,
) -> dict[str, int | float]:
    """Đấu ``num_games`` ván giữa hai agent đã khởi tạo, đổi màu xen kẽ.

    Args:
        agent_a: Agent thứ nhất.
        agent_b: Agent thứ hai.
        num_games: Số ván (nửa đi trước, nửa đi sau).
        board_size: Kích thước bàn cờ.

    Returns:
        Dict thống kê wins_a / wins_b / draws / win_rate_a (kể cả nửa hoà).
    """
    wins_a = wins_b = draws = 0
    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(agent_a, agent_b, board_size)
            if winner is Player.X:
                wins_a += 1
            elif winner is Player.O:
                wins_b += 1
            else:
                draws += 1
        else:
            winner = play_game(agent_b, agent_a, board_size)
            if winner is Player.X:
                wins_b += 1
            elif winner is Player.O:
                wins_a += 1
            else:
                draws += 1
    return {
        "games": num_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / num_games if num_games else 0.0,
        "win_rate_b": wins_b / num_games if num_games else 0.0,
    }


def round_robin(
    difficulty: Difficulty = Difficulty.MEDIUM,
    board_size: int = DEFAULT_BOARD_SIZE,
    num_games: int = 20,
    keys: tuple[str, ...] = ("minimax", "dqn", "hybrid"),
    tactical_config: object | None = None,
) -> dict[str, object]:
    """Giải đấu vòng tròn thật giữa các agent — thước đo sức mạnh đúng bản chất.

    Mỗi cặp đấu ``num_games`` ván (đổi màu). Trả bảng đối đầu + điểm tổng (số ván
    thắng trên toàn giải) để xếp hạng. Đây là cái nên dùng để khẳng định
    "Hybrid > Minimax", thay cho điểm heuristic-1-nước của benchmark.

    Args:
        difficulty: Độ khó áp cho cả ba agent.
        board_size: Kích thước bàn cờ.
        num_games: Số ván mỗi cặp.
        keys: Tên các agent tham gia.
        tactical_config: Luật chiến thuật tuỳ chọn (None = mặc định).

    Returns:
        Dict gồm ``matrix`` (win-rate A so với từng đối thủ), ``standings``
        (tổng thắng, xếp hạng) và ``ranking`` (danh sách key theo thứ tự mạnh→yếu).
    """
    from ai.factory import create_agent

    type_map = {
        "minimax": AIType.MINIMAX,
        "dqn": AIType.DQN,
        "hybrid": AIType.HYBRID,
    }
    agents: dict[str, Agent] = {
        k: create_agent(type_map[k], difficulty, board_size, tactical_config)  # type: ignore[arg-type]
        for k in keys
    }

    matrix: dict[str, dict[str, dict[str, int | float]]] = {k: {} for k in keys}
    total_wins: dict[str, int] = {k: 0 for k in keys}
    total_games: dict[str, int] = {k: 0 for k in keys}

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            res = play_match_agents(agents[a], agents[b], num_games, board_size)
            matrix[a][b] = {**res}
            matrix[b][a] = {
                "games": res["games"],
                "wins_a": res["wins_b"],
                "wins_b": res["wins_a"],
                "draws": res["draws"],
                "win_rate_a": res["win_rate_b"],
                "win_rate_b": res["win_rate_a"],
            }
            total_wins[a] += int(res["wins_a"])
            total_wins[b] += int(res["wins_b"])
            total_games[a] += num_games
            total_games[b] += num_games

    ranking = sorted(keys, key=lambda k: total_wins[k], reverse=True)
    standings = {
        k: {
            "total_wins": total_wins[k],
            "total_games": total_games[k],
            "win_rate": total_wins[k] / total_games[k] if total_games[k] else 0.0,
            "rank": ranking.index(k) + 1,
            "name": agents[k].name,
        }
        for k in keys
    }
    return {
        "difficulty": difficulty.name,
        "board_size": board_size,
        "num_games_per_pair": num_games,
        "matrix": matrix,
        "standings": standings,
        "ranking": list(ranking),
        "winner": ranking[0] if ranking else None,
    }


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
