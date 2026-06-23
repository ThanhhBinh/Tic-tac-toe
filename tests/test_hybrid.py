#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test cho HybridAgent (Minimax depth+1 + DQN ordering)."""

from __future__ import annotations

from ai.factory import create_agent
from ai.heuristic import evaluate_position
from ai.hybrid_agent import HybridAgent
from ai.minimax_agent import MinimaxAgent
from config import AIType, Difficulty, Player, hybrid_depth_for
from core.caro_env import CaroEnv


def test_hybrid_tra_ve_nuoc_hop_le() -> None:
    env = CaroEnv(size=10)
    agent = HybridAgent.from_difficulty(Difficulty.MEDIUM, board_size=10)
    move = agent.get_move(env)
    assert env.is_legal(move)


def test_hybrid_ke_thua_minimax_tactical() -> None:
    env = CaroEnv(size=10)
    env.board[3, 3] = Player.O
    env.board[3, 4] = Player.O
    env.board[3, 5] = Player.O
    env.board[3, 6] = Player.O
    env.current_player = Player.X
    env._move_count = 4  # noqa: SLF001
    env.last_move = (3, 6)

    agent = HybridAgent(depth=2, board_size=10)
    move = agent.get_move(env)
    assert move in {(3, 2), (3, 7)}


def test_hybrid_sau_hon_minimax_1_ply() -> None:
    """Hybrid tìm sâu hơn Minimax đúng HYBRID_EXTRA_DEPTH ply (mạnh hơn thật)."""
    from config import HYBRID_EXTRA_DEPTH, HYBRID_MAX_DEPTH

    assert HYBRID_EXTRA_DEPTH >= 1
    for diff in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
        hybrid = HybridAgent.from_difficulty(diff, board_size=10)
        minimax = MinimaxAgent.from_difficulty(diff)
        assert hybrid.depth == hybrid_depth_for(diff)
        assert hybrid.depth == min(int(diff) + HYBRID_EXTRA_DEPTH, HYBRID_MAX_DEPTH)
        assert hybrid.depth > minimax.depth


def test_hybrid_dqn_reorder_khi_co_model() -> None:
    """DQN được gọi để sắp xếp nước (ordering), không override get_move trực tiếp."""
    env = CaroEnv(size=10)
    agent = HybridAgent(depth=2, board_size=10, max_branch=None)
    agent.dqn._model_loaded = True

    calls: list[int] = []
    original = agent.dqn._predict_q_numpy

    def spy(env_arg: CaroEnv, player: Player) -> object:
        calls.append(1)
        return original(env_arg, player)

    agent.dqn._predict_q_numpy = spy  # type: ignore[method-assign]
    agent._dqn_reorder_root = True
    agent._ordered_moves(env, Player.X)
    assert len(calls) >= 1


def test_hybrid_co_win_probability() -> None:
    env = CaroEnv(size=10)
    agent = HybridAgent(depth=2, board_size=10)
    prob = agent.get_win_probability(env)
    assert prob is not None
    assert 0.0 <= prob <= 1.0

    agent.dqn._model_loaded = True
    prob_dqn = agent.get_win_probability(env)
    assert prob_dqn is not None
    assert 0.0 <= prob_dqn <= 1.0


def test_factory_tao_hybrid() -> None:
    agent = create_agent(AIType.HYBRID, Difficulty.EASY, board_size=10)
    assert isinstance(agent, HybridAgent)
    assert agent.depth == hybrid_depth_for(Difficulty.EASY)


def test_hybrid_leaf_heuristic() -> None:
    env = CaroEnv(size=10)
    agent = HybridAgent(depth=1, board_size=10)
    score = agent._evaluate_leaf(env, Player.X)
    assert isinstance(score, float)


def test_hybrid_khong_yeu_hon_minimax_doi_khang() -> None:
    """Thước đo THẬT: đối kháng → Hybrid (sâu hơn) không yếu hơn Minimax.

    Dùng head-to-head thay cho điểm heuristic-1-nước của benchmark (thước đo đó
    thiên vị nước có heuristic tức thời cao nên không phản ánh sức mạnh thật).
    Agent tất định + bàn nhỏ → kết quả ổn định.
    """
    from ai.evaluate import play_match_agents

    board_size = 7
    minimax = MinimaxAgent(
        depth=2, candidate_radius=2, max_branch=8, time_budget=None
    )
    hybrid = HybridAgent(
        depth=3,
        board_size=board_size,
        candidate_radius=2,
        max_branch=8,
        time_budget=None,
    )
    res = play_match_agents(hybrid, minimax, num_games=2, board_size=board_size)
    assert res["wins_a"] >= res["wins_b"]
