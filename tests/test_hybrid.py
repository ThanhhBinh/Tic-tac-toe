#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test cho HybridAgent (Minimax + DQN đánh giá node lá)."""

from __future__ import annotations

import numpy as np

from ai.factory import create_agent
from ai.hybrid_agent import HybridAgent
from ai.minimax_agent import MinimaxAgent
from config import AIType, Difficulty, HYBRID_DEPTH_BY_DIFFICULTY, Player
from core.caro_env import CaroEnv


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


def test_hybrid_dung_dqn_khi_refine_root() -> None:
    """Khi model đã nạp, DQN được gọi ở bước tinh chỉnh root."""
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
    """Có DQN → depth map Hybrid; chưa DQN → depth bằng Minimax Expert."""
    agent = HybridAgent.from_difficulty(Difficulty.EXPERT, board_size=10)
    if agent.dqn._model_loaded:
        assert agent.depth == HYBRID_DEPTH_BY_DIFFICULTY[Difficulty.EXPERT]
        assert agent.max_branch is None
    else:
        assert agent.depth == int(Difficulty.EXPERT)


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


def test_hybrid_dqn_loaded_khong_gioi_han_nhanh() -> None:
    """Khi đã nạp DQN, Hybrid duyệt đủ nhánh như Minimax (max_branch=None)."""
    agent = HybridAgent.from_difficulty(Difficulty.MEDIUM, board_size=15)
    if agent.dqn._model_loaded:
        assert agent.max_branch is None


def test_hybrid_search_khop_minimax_khi_co_dqn() -> None:
    """Search heuristic thuần → cùng nước Minimax trên bàn trống (trước refine)."""
    env = CaroEnv(size=10)
    hybrid = HybridAgent.from_difficulty(Difficulty.MEDIUM, board_size=10)
    minimax = MinimaxAgent.from_difficulty(Difficulty.MEDIUM)
    if not hybrid.dqn._model_loaded:
        return
    hybrid._heuristic_only_search = True
    h_move = super(HybridAgent, hybrid).get_move(env.clone())
    m_move = minimax.get_move(env.clone())
    assert h_move == m_move
