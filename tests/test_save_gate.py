#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test save gate và chất lượng ván online."""

from __future__ import annotations

import pytest
pytest.importorskip("torch")

from collections import deque

from ai.board_encoder import encode_board
from ai.dqn_trainer import DQNTrainer
from ai.online_learner import (
    GameMoveRecorder,
    assess_game_quality,
    gradient_steps_for_quality,
)
from ai.replay_buffer import ReplayBuffer, Transition
from ai.save_gate import loss_spike_rejected, transitions_show_improvement
from config import (
    ONLINE_LEARN_DRAW_MIN_AI_MOVES,
    ONLINE_LEARN_GRADIENT_STEPS,
    ONLINE_LEARN_GRADIENT_STEPS_HIGH,
    ONLINE_LEARN_GRADIENT_STEPS_LOW,
    ONLINE_LEARN_MIN_AI_MOVES,
    Player,
)
from core.caro_env import CaroEnv


def test_assess_game_quality_skip_short_ai_win() -> None:
    """AI thắng ván quá ngắn — bỏ qua học."""
    assert assess_game_quality("ai_win", ONLINE_LEARN_MIN_AI_MOVES - 1) == "skip"
    assert assess_game_quality("ai_win", ONLINE_LEARN_MIN_AI_MOVES) != "skip"


def test_assess_game_quality_high_on_long_loss() -> None:
    """AI thua ván dài — chất lượng cao."""
    assert assess_game_quality("ai_loss", 12) == "high"


def test_assess_game_quality_draw() -> None:
    """Ván hòa ngắn skip; hòa dài high."""
    assert assess_game_quality("draw", ONLINE_LEARN_DRAW_MIN_AI_MOVES - 1) == "skip"
    assert assess_game_quality("draw", ONLINE_LEARN_DRAW_MIN_AI_MOVES) == "high"


def test_gradient_steps_by_quality() -> None:
    """Số bước gradient tăng theo chất lượng ván."""
    assert gradient_steps_for_quality("high") == ONLINE_LEARN_GRADIENT_STEPS_HIGH
    assert gradient_steps_for_quality("low") == ONLINE_LEARN_GRADIENT_STEPS_LOW
    assert gradient_steps_for_quality("medium") == ONLINE_LEARN_GRADIENT_STEPS


def test_loss_credit_assignment_three_moves() -> None:
    """Khi thua, 3 nước cuối nhận reward âm dần."""
    recorder = GameMoveRecorder()
    env = CaroEnv(size=10)
    env.reset()
    moves = [(2, 2), (3, 3), (4, 4)]
    for move in moves:
        before = env.clone()
        env.step(move)
        recorder.record_ai_move(before, move, Player.X, env)

    transitions = recorder.build_transitions("ai_loss")
    assert len(transitions) == 3
    assert transitions[-1].reward == -1.0
    assert transitions[-2].reward == -0.5
    assert transitions[-3].reward == -0.25
    assert transitions[-1].done is True
    assert transitions[-2].done is False


def test_replay_buffer_prioritized_favors_high_reward() -> None:
    """Buffer ưu tiên transition |reward| lớn."""
    buf = ReplayBuffer(100, seed=0)
    env = CaroEnv(size=5)
    env.reset()
    low = Transition(
        state=encode_board(env.board, Player.X),
        action=0,
        reward=-0.005,
        next_state=encode_board(env.board, Player.O),
        done=False,
    )
    high = Transition(
        state=encode_board(env.board, Player.X),
        action=1,
        reward=-1.0,
        next_state=encode_board(env.board, Player.O),
        done=True,
    )
    for _ in range(50):
        buf.push(low)
    for _ in range(5):
        buf.push(high, priority=2.0)

    batch = buf.sample(20)
    high_count = sum(1 for t in batch if t.reward <= -1.0)
    assert high_count >= 1


def test_loss_spike_rejected() -> None:
    """Loss đột biến bị từ chối."""
    history: deque[float] = deque([0.1, 0.12, 0.11], maxlen=10)
    assert loss_spike_rejected(0.5, history) is True
    assert loss_spike_rejected(0.15, history) is False


def test_transitions_show_improvement_ai_win_always_ok() -> None:
    """Ván thắng không cần transition gate."""
    trainer = DQNTrainer(board_size=10, buffer_capacity=64, seed=1)
    assert transitions_show_improvement(trainer, [], "ai_win") is True
