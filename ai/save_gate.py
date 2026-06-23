#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save gate — chỉ ghi checkpoint khi model thực sự cải thiện hoặc không regress."""

from __future__ import annotations

import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ai.benchmark import BENCHMARK_SCENARIOS, _make_env, _run_agent_on_scenario

try:
    from ai.dqn_agent import DQNAgent
    from ai.dqn_trainer import DQNTrainer
except ModuleNotFoundError:
    DQNAgent = None  # type: ignore[assignment,misc]
    DQNTrainer = None  # type: ignore[assignment,misc]
from ai.learning_inspector import _dqn_greedy_move, _env_from_transition
from ai.replay_buffer import Transition
from config import (
    ONLINE_LOSS_SPIKE_RATIO,
    ONLINE_SAVE_GATE_ENABLED,
    TacticalConfig,
    dqn_model_backup_path,
    dqn_model_path,
)

GateReason = Literal[
    "gate_disabled",
    "no_backup_first_save",
    "loss_spike",
    "tactical_regressed",
    "transition_improved",
    "tactical_ok",
    "no_improvement",
]


@dataclass(slots=True)
class SaveGateResult:
    """Kết quả quyết định lưu checkpoint."""

    saved: bool
    rolled_back: bool
    reason: GateReason
    tactical_before_pct: float | None = None
    tactical_after_pct: float | None = None


def _agent_from_trainer(trainer: DQNTrainer) -> DQNAgent:
    """Tạo DQNAgent tạm từ policy_net trong trainer (không đọc file)."""
    agent = DQNAgent(board_size=trainer.board_size, epsilon=0.0)
    agent.network.load_state_dict(trainer.policy_net.state_dict())
    agent.network.eval()
    agent._model_loaded = True
    return agent


def tactical_benchmark_score(
    agent: DQNAgent,
    board_size: int,
    tactical: TacticalConfig | None = None,
) -> tuple[int, int, float]:
    """Đếm số TH chiến thuật đúng trên benchmark cố định."""
    cfg = tactical or TacticalConfig()
    correct = 0
    total = 0
    for scenario in BENCHMARK_SCENARIOS:
        _env, expected_moves = _make_env(scenario, board_size, cfg)
        if expected_moves is None:
            continue
        total += 1
        result = _run_agent_on_scenario(agent, scenario, board_size, cfg, expected_moves)
        if result.get("is_expected") is True:
            correct += 1
    pct = (correct / total * 100.0) if total else 0.0
    return correct, total, pct


def transitions_show_improvement(
    trainer: DQNTrainer,
    transitions: list[Transition],
    outcome: str,
) -> bool:
    """True nếu model không còn lặp nước sai từ ván vừa học (ai_loss)."""
    if outcome != "ai_loss" or not transitions:
        return True

    agent = _agent_from_trainer(trainer)
    for transition in transitions:
        if transition.reward > -0.1:
            continue
        env, _perspective, move = _env_from_transition(transition, trainer.board_size)
        greedy = _dqn_greedy_move(agent, env)
        if greedy != move:
            return True
    return False


def loss_spike_rejected(avg_loss: float, recent_losses: deque[float]) -> bool:
    """Từ chối lưu nếu loss đột biến so với lịch sử gần đây."""
    if avg_loss <= 0 or len(recent_losses) < 2:
        return False
    baseline = sum(recent_losses) / len(recent_losses)
    if baseline <= 0:
        return False
    return avg_loss > baseline * ONLINE_LOSS_SPIKE_RATIO


def restore_checkpoint_from_backup(board_size: int, trainer: DQNTrainer) -> bool:
    """Khôi phục trainer + file chính từ bản backup."""
    backup = dqn_model_backup_path(board_size)
    if not backup.exists():
        return False
    trainer.load_checkpoint(backup)
    shutil.copy2(backup, dqn_model_path(board_size))
    return True


def evaluate_online_save_gate(
    trainer: DQNTrainer,
    board_size: int,
    transitions: list[Transition],
    outcome: str,
    avg_loss: float,
    recent_losses: deque[float],
    agent_before: DQNAgent | None = None,
) -> SaveGateResult:
    """Đánh giá có nên giữ checkpoint sau gradient update online."""
    if not ONLINE_SAVE_GATE_ENABLED:
        return SaveGateResult(saved=True, rolled_back=False, reason="gate_disabled")

    if loss_spike_rejected(avg_loss, recent_losses):
        return SaveGateResult(saved=False, rolled_back=False, reason="loss_spike")

    agent_after = _agent_from_trainer(trainer)
    _, _, after_pct = tactical_benchmark_score(agent_after, board_size)

    if agent_before is None or not agent_before._model_loaded:
        return SaveGateResult(
            saved=True,
            rolled_back=False,
            reason="no_backup_first_save",
            tactical_after_pct=after_pct,
        )

    _, _, before_pct = tactical_benchmark_score(agent_before, board_size)

    if after_pct < before_pct:
        return SaveGateResult(
            saved=False,
            rolled_back=True,
            reason="tactical_regressed",
            tactical_before_pct=before_pct,
            tactical_after_pct=after_pct,
        )

    if transitions_show_improvement(trainer, transitions, outcome):
        return SaveGateResult(
            saved=True,
            rolled_back=False,
            reason="transition_improved",
            tactical_before_pct=before_pct,
            tactical_after_pct=after_pct,
        )

    if after_pct >= before_pct:
        return SaveGateResult(
            saved=True,
            rolled_back=False,
            reason="tactical_ok",
            tactical_before_pct=before_pct,
            tactical_after_pct=after_pct,
        )

    return SaveGateResult(
        saved=False,
        rolled_back=False,
        reason="no_improvement",
        tactical_before_pct=before_pct,
        tactical_after_pct=after_pct,
    )


def commit_online_checkpoint(
    trainer: DQNTrainer,
    board_size: int,
    transitions: list[Transition],
    outcome: str,
    avg_loss: float,
    recent_losses: deque[float],
    agent_before: DQNAgent | None,
) -> SaveGateResult:
    """Lưu checkpoint nếu pass gate; rollback nếu tactical regress."""
    gate = evaluate_online_save_gate(
        trainer,
        board_size,
        transitions,
        outcome,
        avg_loss,
        recent_losses,
        agent_before,
    )

    if gate.saved:
        trainer.save_agent()
        return gate

    if gate.rolled_back:
        restore_checkpoint_from_backup(board_size, trainer)

    return gate


def evaluate_train_save_gate(
    trainer: DQNTrainer,
    board_size: int,
    win_rate: float,
    best_win_rate: float,
) -> bool:
    """Train.py: chỉ lưu khi win rate eval >= best hiện tại."""
    if win_rate + 1e-9 >= best_win_rate:
        return True
    agent = _agent_from_trainer(trainer)
    _, _, pct = tactical_benchmark_score(agent, board_size)
    return pct >= 70.0 and win_rate >= best_win_rate - 0.05


def snapshot_trainer_agent(trainer: DQNTrainer) -> DQNAgent:
    """Chụp trọng số policy hiện tại trước khi train (để so sánh gate)."""
    return _agent_from_trainer(trainer)


def load_agent_from_path(path: Path, board_size: int) -> DQNAgent | None:
    """Nạp agent từ checkpoint nếu file tồn tại."""
    if not path.exists():
        return None
    return DQNAgent(board_size=board_size, epsilon=0.0, model_path=path)
