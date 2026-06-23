#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test cho DQN: encoder, mạng, agent, replay buffer và trainer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from config import Difficulty, Player
from core.caro_env import CaroEnv
from ai.board_encoder import (
    action_to_move,
    encode_board,
    legal_action_mask,
    move_to_action,
)
from ai.dqn_agent import DQNAgent
from ai.dqn_model import DQNNetwork
from ai.dqn_trainer import DQNTrainer
from ai.replay_buffer import ReplayBuffer, Transition
from config import dqn_model_path


def test_encode_board_shape() -> None:
    """Tensor mã hoá phải có shape (3, H, W)."""
    env = CaroEnv(size=10)
    env.step((5, 5))
    state = encode_board(env.board, Player.X)
    assert state.shape == (3, 10, 10)
    assert state.dtype == np.float32


def test_move_action_bieu_thuc_nguoc() -> None:
    """move_to_action và action_to_move phải nghịch đảo nhau."""
    move = (3, 7)
    action = move_to_action(move, 10)
    assert action_to_move(action, 10) == move


def test_dqn_network_forward() -> None:
    """Mạng CNN trả Q-value đúng kích thước."""
    net = DQNNetwork(10)
    x = torch.randn(4, 3, 10, 10)
    out = net(x)
    assert out.shape == (4, 100)


def test_dqn_agent_tra_ve_nuoc_hop_le() -> None:
    """Agent luôn trả nước đi hợp lệ."""
    env = CaroEnv(size=10)
    agent = DQNAgent(board_size=10, epsilon=0.0)
    move = agent.get_move(env)
    assert env.is_legal(move)


def test_dqn_win_probability_trong_khoang() -> None:
    """Xác suất thắng ước lượng phải nằm trong [0, 1]."""
    env = CaroEnv(size=10)
    agent = DQNAgent(board_size=10)
    prob = agent.get_win_probability(env)
    assert prob is not None
    assert 0.0 <= prob <= 1.0


def test_dqn_save_load() -> None:
    """Lưu và nạp checkpoint phải giữ nguyên trọng số."""
    agent = DQNAgent(board_size=10, epsilon=0.0)
    path = dqn_model_path(10)
    agent.save(path)

    env = CaroEnv(size=10)
    q_before = agent._predict_q_numpy(env, Player.X)

    agent2 = DQNAgent(board_size=10, epsilon=0.0)
    agent2.load(path)
    q_after = agent2._predict_q_numpy(env, Player.X)

    assert np.allclose(q_before, q_after, atol=1e-5)
    path.unlink(missing_ok=True)


def test_replay_buffer_sample() -> None:
    """Buffer lấy mẫu đúng kích thước batch."""
    buf = ReplayBuffer(100, seed=0)
    state = np.zeros((3, 5, 5), dtype=np.float32)
    for i in range(20):
        buf.push(Transition(state, i, 0.0, state, False))
    batch = buf.sample(8)
    assert len(batch) == 8


def test_trainer_chay_vong_ngan() -> None:
    """Trainer chạy vài episode trên bàn nhỏ không lỗi."""
    trainer = DQNTrainer(board_size=5, buffer_capacity=500, batch_size=16, seed=42)
    trainer.train(num_episodes=3)
    assert trainer.stats.episodes == 3
    assert trainer.stats.total_steps > 0


def test_dqn_from_difficulty() -> None:
    """Factory difficulty map sang epsilon chơi."""
    agent = DQNAgent.from_difficulty(Difficulty.EXPERT, board_size=10)
    assert agent.epsilon == 0.0


def test_legal_action_mask() -> None:
    """Mặt nạ hành động chỉ True tại ô trống."""
    env = CaroEnv(size=5)
    env.step((2, 2))
    mask = legal_action_mask(env.board)
    assert mask.sum() == 5 * 5 - 1
    assert not mask[move_to_action((2, 2), 5)]
