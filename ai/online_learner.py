#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Học online từ ván Người vs AI (DQN / Hybrid).

Khi người chơi thắng, ghi lại các nước AI đã đi, gán phần thưởng âm cho
nước cuối, đưa vào replay buffer và cập nhật mạng DQN — giúp AI học từ sai lầm.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from ai.base_agent import Agent
from ai.board_encoder import encode_board, move_to_action
from ai.dqn_agent import DQNAgent
from ai.dqn_trainer import DQNTrainer
from ai.hybrid_agent import HybridAgent
from ai.replay_buffer import Transition
from config import (
    DQN_BATCH_SIZE,
    DQN_BUFFER_CAPACITY,
    ONLINE_LEARN_ENABLED,
    ONLINE_LEARN_GRADIENT_STEPS,
    ONLINE_LEARN_MIN_SAMPLES,
    Player,
    dqn_model_path,
)
from core.caro_env import CaroEnv
from core.constants import Move

GameOutcome = Literal["ai_loss", "ai_win", "draw"]


def extract_dqn_agent(agent: Agent | None) -> DQNAgent | None:
    """Lấy DQNAgent từ DQN thuần hoặc Hybrid; Minimax trả None."""
    if agent is None:
        return None
    if isinstance(agent, DQNAgent):
        return agent
    if isinstance(agent, HybridAgent):
        return agent.dqn
    return None


@dataclass(slots=True)
class LearnResult:
    """Kết quả một lần học online."""

    outcome: GameOutcome
    ai_moves: int
    gradient_steps: int
    avg_loss: float
    model_saved: bool
    buffered_only: bool = False


class GameMoveRecorder:
    """Ghi các transition của AI trong một ván PvA."""

    def __init__(self) -> None:
        self._transitions: list[Transition] = []
        self._invalidated: bool = False

    def reset(self) -> None:
        """Xoá bản ghi khi bắt đầu ván mới."""
        self._transitions.clear()
        self._invalidated = False

    def invalidate(self) -> None:
        """Huỷ bản ghi sau undo — không học từ ván đã chỉnh sửa."""
        self._transitions.clear()
        self._invalidated = True

    @property
    def ai_move_count(self) -> int:
        """Số nước AI đã ghi."""
        return len(self._transitions)

    def record_ai_move(
        self,
        env_before: CaroEnv,
        move: Move,
        ai_player: Player,
        env_after: CaroEnv,
    ) -> None:
        """Lưu một nước AI vào bộ nhớ tạm của ván."""
        if self._invalidated:
            return

        state = encode_board(env_before.board, ai_player)
        action = move_to_action(move, env_before.size)
        reward = DQNTrainer._compute_reward(env_after, ai_player, move)
        next_perspective = (
            env_after.current_player if not env_after.done else ai_player
        )
        next_state = encode_board(env_after.board, next_perspective)
        self._transitions.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=env_after.done,
            )
        )

    def build_transitions(self, outcome: GameOutcome) -> list[Transition]:
        """Chuẩn hoá phần thưởng cuối ván trước khi đưa vào buffer."""
        if self._invalidated or not self._transitions:
            return []

        items = list(self._transitions)
        if outcome == "ai_loss":
            last = items[-1]
            items[-1] = Transition(
                state=last.state,
                action=last.action,
                reward=-1.0,
                next_state=last.next_state,
                done=True,
            )
        elif outcome == "ai_win":
            last = items[-1]
            items[-1] = Transition(
                state=last.state,
                action=last.action,
                reward=1.0,
                next_state=last.next_state,
                done=True,
            )
        return items


class OnlineLearner:
    """Quản lý replay buffer + gradient update sau mỗi ván PvA."""

    _instances: dict[int, OnlineLearner] = {}
    _instances_lock = threading.Lock()

    def __init__(self, board_size: int) -> None:
        self.board_size = board_size
        self.trainer = DQNTrainer(
            board_size=board_size,
            buffer_capacity=DQN_BUFFER_CAPACITY,
        )
        self._lock = threading.Lock()
        model_path = dqn_model_path(board_size)
        if model_path.exists():
            self.trainer.load_checkpoint(model_path)

    @classmethod
    def for_board(cls, board_size: int) -> OnlineLearner:
        """Singleton theo kích thước bàn cờ."""
        with cls._instances_lock:
            if board_size not in cls._instances:
                cls._instances[board_size] = cls(board_size)
            return cls._instances[board_size]

    def learn_from_game(
        self,
        recorder: GameMoveRecorder,
        outcome: GameOutcome,
    ) -> LearnResult | None:
        """Đưa ván vào buffer và chạy vài bước gradient.

        Returns:
            LearnResult nếu có học; None nếu tắt hoặc không có dữ liệu.
        """
        if not ONLINE_LEARN_ENABLED or outcome == "draw":
            return None

        transitions = recorder.build_transitions(outcome)
        if not transitions:
            return None

        with self._lock:
            for transition in transitions:
                self.trainer.buffer.push(transition)

            buffer_len = len(self.trainer.buffer)
            effective_batch = min(DQN_BATCH_SIZE, buffer_len)
            can_train = effective_batch >= ONLINE_LEARN_MIN_SAMPLES

            steps = 0
            loss_sum = 0.0
            if can_train:
                for _ in range(ONLINE_LEARN_GRADIENT_STEPS):
                    if len(self.trainer.buffer) < ONLINE_LEARN_MIN_SAMPLES:
                        break
                    batch_size = min(DQN_BATCH_SIZE, len(self.trainer.buffer))
                    loss_sum += self.trainer._optimize(batch_size=batch_size)
                    steps += 1
                    self.trainer._maybe_sync_target()

            saved = False
            if steps > 0:
                self.trainer.save_agent()
                saved = True

        avg_loss = loss_sum / steps if steps > 0 else 0.0
        return LearnResult(
            outcome=outcome,
            ai_moves=len(transitions),
            gradient_steps=steps,
            avg_loss=avg_loss,
            model_saved=saved,
            buffered_only=not can_train,
        )

    @staticmethod
    def reload_agent_weights(agent: Agent | None) -> None:
        """Nạp lại checkpoint mới nhất vào agent đang chơi."""
        dqn = extract_dqn_agent(agent)
        if dqn is None:
            return

        path = dqn_model_path(dqn.board_size)
        if not path.exists():
            return

        dqn.load(path)
        if isinstance(agent, HybridAgent):
            agent._eval_cache.clear()
            if dqn._model_loaded:
                agent.name = f"Hybrid (depth={agent.depth}, DQN đã nạp)"


def resolve_game_outcome(
    human: Player | None,
    winner: Player | None,
    is_draw: bool,
) -> GameOutcome | None:
    """Xác định kết quả ván từ góc AI (chỉ PvA)."""
    if human is None or is_draw or winner is None:
        return "draw" if is_draw else None
    if winner is human:
        return "ai_loss"
    if winner is human.opponent:
        return "ai_win"
    return None


def learn_from_pva_game(
    recorder: GameMoveRecorder,
    board_size: int,
    human: Player | None,
    winner: Player | None,
    is_draw: bool,
    ai_agent: Agent | None,
) -> LearnResult | None:
    """API cấp cao: học từ ván PvA và nạp lại trọng số agent."""
    if extract_dqn_agent(ai_agent) is None:
        return None

    outcome = resolve_game_outcome(human, winner, is_draw)
    if outcome is None or outcome == "draw":
        return None

    learner = OnlineLearner.for_board(board_size)
    result = learner.learn_from_game(recorder, outcome)
    if result is not None and result.model_saved:
        OnlineLearner.reload_agent_weights(ai_agent)
    return result
