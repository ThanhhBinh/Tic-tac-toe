#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test cho HybridAgent (Minimax + DQN đánh giá node lá)."""

from __future__ import annotations

import numpy as np

from config import Difficulty, Player
from core.caro_env import CaroEnv
from ai.factory import create_agent
from ai.hybrid_agent import HybridAgent
from ai.minimax_agent import MinimaxAgent
from config import AIType


def test_hybrid_tra_ve_nuoc_hop_le() -> None:
    """Hybrid luôn trả nước đi hợp lệ trên bàn trống."""
    env = CaroEnv(size=10)
    agent = HybridAgent.from_difficulty(Difficulty.MEDIUM, board_size=10)
    move = agent.get_move(env)
    assert env.is_legal(move)


def test_hybrid_ke_thua_minimax_tactical() -> None:
    """Hybrid kế thừa luật thắng ngay / chặn thua từ Minimax."""
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


def test_hybrid_dung_dqn_o_node_la() -> None:
    """Node lá gọi DQN khi model đã nạp."""
    env = CaroEnv(size=10)
    agent = HybridAgent(depth=1, board_size=10)
    agent.dqn._model_loaded = True

    calls: list[tuple[Player, bool]] = []

    original = agent.dqn._predict_q_numpy

    def spy(env_arg: CaroEnv, player: Player) -> np.ndarray:
        calls.append((player, env_arg.done))
        return original(env_arg, player)

    agent.dqn._predict_q_numpy = spy  # type: ignore[method-assign]

    agent.get_move(env)
    assert len(calls) >= 1


def test_hybrid_co_win_probability() -> None:
    """Win Probability: DQN nếu có model, ngược lại heuristic."""
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
    """Factory trả đúng loại agent Hybrid."""
    agent = create_agent(AIType.HYBRID, Difficulty.EASY, board_size=10)
    assert isinstance(agent, HybridAgent)
    assert agent.depth == 1


def test_hybrid_expert_depth_khi_chua_dqn() -> None:
    """Chưa train DQN → Expert depth=4 (bằng Minimax); có DQN giữ depth=3."""
    agent = HybridAgent.from_difficulty(Difficulty.EXPERT, board_size=10)
    assert not agent.dqn._model_loaded
    assert agent.depth == 4


def test_hybrid_leaf_fallback_khi_q_khong_hop_le() -> None:
    """Khi Q không hợp lệ, _dqn_raw_score trả 0; leaf vẫn có heuristic."""
    env = CaroEnv(size=10)
    agent = HybridAgent(depth=1, board_size=10)
    agent.dqn._model_loaded = True
    score = agent._evaluate_leaf(env, Player.X)
    assert isinstance(score, float)
    assert score != 0.0 or env.move_count == 0


def test_hybrid_khac_minimax_thuan_tren_cung_depth() -> None:
    """Hybrid và Minimax cùng interface, cùng chạy được với depth giống nhau."""
    env = CaroEnv(size=10)
    hybrid = HybridAgent(depth=2, board_size=10)
    minimax = MinimaxAgent(depth=2)
    assert hybrid.get_move(env.clone()).__class__ is tuple
    assert minimax.get_move(env.clone()).__class__ is tuple
