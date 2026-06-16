#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vòng lặp huấn luyện DQN (self-play hoặc đấu với đối thủ cố định).

Module này tách biệt logic train khỏi agent suy luận để ``train.py`` (Bước 5)
chỉ cần gọi API cấp cao, đồng thời cho phép test huấn luyện nhanh trong unit
test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ai.base_agent import Agent
from ai.board_encoder import encode_board, legal_action_mask, move_to_action
from ai.dqn_agent import resolve_device
from ai.dqn_model import DQNNetwork
from ai.replay_buffer import ReplayBuffer, Transition
from ai.heuristic import _is_four_with_open_end, _is_open_three_at, find_tactical_move
from config import (
    DQN_BATCH_SIZE,
    DQN_EPSILON_DECAY,
    DQN_EPSILON_END,
    DQN_EPSILON_START,
    DQN_GAMMA,
    DQN_LEARNING_RATE,
    DQN_MIN_BUFFER,
    DQN_REWARD_OPEN_FOUR,
    DQN_REWARD_OPEN_THREE,
    DQN_REWARD_STEP,
    DQN_TARGET_SYNC_EVERY,
    DQN_TRAIN_EVERY,
    Player,
    TacticalConfig,
    dqn_model_path,
)
from core.caro_env import CaroEnv


@dataclass
class TrainStats:
    """Thống kê tích lũy trong quá trình huấn luyện.

    Attributes:
        episodes: Số episode đã chạy.
        total_steps: Tổng số bước (nước đi) qua tất cả episode.
        losses: Danh sách loss Huber gần nhất (giới hạn bởi deque nội bộ).
    """

    episodes: int = 0
    total_steps: int = 0
    wins_x: int = 0
    losses_x: int = 0
    draws: int = 0
    losses: list[float] = field(default_factory=list)

    @property
    def avg_loss(self) -> float:
        """Loss trung bình trên các bước gradient gần nhất."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)

    @property
    def win_rate_x(self) -> float:
        """Tỷ lệ X thắng qua các episode (ước lượng nhanh khi train)."""
        total = self.wins_x + self.losses_x + self.draws
        if total == 0:
            return 0.0
        return self.wins_x / total


class DQNTrainer:
    """Huấn luyện DQN qua self-play hoặc đấu với agent đối thủ."""

    def __init__(
        self,
        board_size: int,
        device: str | None = None,
        learning_rate: float = DQN_LEARNING_RATE,
        gamma: float = DQN_GAMMA,
        batch_size: int = DQN_BATCH_SIZE,
        buffer_capacity: int = 50_000,
        seed: int | None = None,
    ) -> None:
        """Khởi tạo mạng Q, mạng target và replay buffer.

        Args:
            board_size: Cạnh bàn cờ.
            device: Thiết bị PyTorch.
            learning_rate: Tốc độ học Adam.
            gamma: Hệ số chiết khấu phần thưởng.
            batch_size: Kích thước mini-batch.
            buffer_capacity: Sức chứa replay buffer.
            seed: Hạt giống ngẫu nhiên PyTorch/numpy.
        """
        self.board_size = board_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = resolve_device(device)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.policy_net = DQNNetwork(board_size).to(self.device)
        self.target_net = DQNNetwork(board_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)

        self.epsilon = DQN_EPSILON_START
        self.stats = TrainStats()
        self._steps_since_sync = 0

    def train_episode(self, opponent: Agent | None = None) -> float:
        """Chạy một episode huấn luyện và trả reward cuối từ góc X.

        Args:
            opponent: Agent đối thủ cố định; None = self-play thuần (cả hai
                bên dùng cùng policy_net + epsilon-greedy).

        Returns:
            Phần thưởng tích lũy của Player.X trong episode.
        """
        env = CaroEnv(size=self.board_size)
        env.reset()
        total_reward_x = 0.0

        while not env.done:
            player = env.current_player
            move = self._select_action(env, player, opponent)
            prev_state = encode_board(env.board, player)

            env.step(move)
            reward = self._compute_reward(env, player, move)
            if player is Player.X:
                total_reward_x += reward

            # Trạng thái kế tiếp theo góc nhìn người sắp đi (chuẩn MDP 2 người).
            next_perspective = env.current_player if not env.done else player
            next_state = encode_board(env.board, next_perspective)
            done = env.done

            self.buffer.push(
                Transition(
                    state=prev_state,
                    action=move_to_action(move, self.board_size),
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )

            self.stats.total_steps += 1
            if (
                len(self.buffer) >= DQN_MIN_BUFFER
                and self.stats.total_steps % DQN_TRAIN_EVERY == 0
            ):
                loss = self._optimize()
                self.stats.losses.append(loss)
                if len(self.stats.losses) > 200:
                    self.stats.losses = self.stats.losses[-200:]

            self._decay_epsilon()
            self._maybe_sync_target()

        self.stats.episodes += 1
        self._record_episode_outcome(env)
        return total_reward_x

    def _record_episode_outcome(self, env: CaroEnv) -> None:
        """Cập nhật thống kê thắng/thua/hòa từ góc Player.X."""
        if env.is_draw:
            self.stats.draws += 1
        elif env.winner is Player.X:
            self.stats.wins_x += 1
        elif env.winner is Player.O:
            self.stats.losses_x += 1

    def _select_action(
        self,
        env: CaroEnv,
        player: Player,
        opponent: Agent | None,
    ) -> tuple[int, int]:
        """Chọn nước đi cho ``player`` trong phase huấn luyện."""
        if opponent is not None and player is Player.O:
            return opponent.get_move(env)

        tactical = find_tactical_move(
            env, player, radius=2, config=TacticalConfig(aggressive=True)
        )
        if tactical is not None:
            return tactical

        # Self-play / X: epsilon-greedy trên policy_net.
        legal = env.candidate_moves(radius=2) or env.legal_moves()
        if np.random.random() < self.epsilon:
            return legal[int(np.random.randint(len(legal)))]

        state = encode_board(env.board, player)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        mask = legal_action_mask(env.board)

        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(tensor).squeeze(0).cpu().numpy()
        q[~mask] = -np.inf
        action = int(np.argmax(q))
        row, col = action // self.board_size, action % self.board_size
        return (row, col)

    @staticmethod
    def _compute_reward(env: CaroEnv, player: Player, move: tuple[int, int]) -> float:
        """Tính phần thưởng cho người vừa đi (thắng + thưởng hình tấn công).

        Args:
            env: Môi trường sau nước đi.
            player: Người vừa đặt quân.
            move: Ô vừa đặt.

        Returns:
            Phần thưởng scalar cho replay buffer.
        """
        if env.done:
            if env.winner is player:
                return 1.0
            if env.winner is player.opponent:
                return -1.0
            return 0.0

        row, col = move
        bonus = 0.0
        if _is_four_with_open_end(env.board, row, col, player, env.size):
            bonus += DQN_REWARD_OPEN_FOUR
        elif _is_open_three_at(env.board, row, col, player, env.size):
            bonus += DQN_REWARD_OPEN_THREE
        return DQN_REWARD_STEP + bonus

    def _optimize(self, batch_size: int | None = None) -> float:
        """Một bước gradient trên mini-batch từ replay buffer."""
        bs = batch_size or self.batch_size
        batch = self.buffer.sample(bs)
        states = torch.from_numpy(self.buffer.states_batch(batch)).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(self.buffer.next_states_batch(batch)).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        self.policy_net.train()
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def _decay_epsilon(self) -> None:
        """Giảm epsilon theo hệ số decay."""
        self.epsilon = max(DQN_EPSILON_END, self.epsilon * DQN_EPSILON_DECAY)

    def _maybe_sync_target(self) -> None:
        """Đồng bộ target network định kỳ."""
        self._steps_since_sync += 1
        if self._steps_since_sync >= DQN_TARGET_SYNC_EVERY:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self._steps_since_sync = 0

    def save_agent(self, path: Path | None = None) -> Path:
        """Xuất trọng số policy_net sang file để DQNAgent nạp khi chơi.

        Args:
            path: Đích lưu; None = đường dẫn mặc định theo board_size.

        Returns:
            Path file checkpoint.
        """
        dest = path or dqn_model_path(self.board_size)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"board_size": self.board_size, "state_dict": self.policy_net.state_dict()},
            dest,
        )
        return dest

    def load_checkpoint(self, path: Path) -> None:
        """Nạp trọng số policy/target net từ checkpoint để tiếp tục huấn luyện.

        Args:
            path: File ``.pth`` đã lưu trước đó.

        Raises:
            ValueError: Nếu board_size checkpoint không khớp.
            FileNotFoundError: Nếu file không tồn tại.
        """
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved_size = int(checkpoint.get("board_size", self.board_size))
        if saved_size != self.board_size:
            raise ValueError(
                f"Checkpoint bàn {saved_size}x{saved_size} không khớp trainer "
                f"{self.board_size}x{self.board_size}."
            )
        self.policy_net.load_state_dict(checkpoint["state_dict"])
        self.target_net.load_state_dict(checkpoint["state_dict"])
        self.policy_net.train()
        self.target_net.eval()

    def train(
        self,
        num_episodes: int,
        opponent: Agent | None = None,
        save_every: int = 0,
        log_every: int = 0,
        on_log: "callable[[int, TrainStats, float], None] | None" = None,
    ) -> TrainStats:
        """Huấn luyện nhiều episode liên tiếp.

        Args:
            num_episodes: Số episode cần chạy.
            opponent: Đối thủ cố định (None = self-play).
            save_every: Lưu checkpoint mỗi N episode (0 = không tự lưu).
            log_every: Gọi ``on_log`` mỗi N episode (0 = tắt).
            on_log: Callback ``(episode, stats, epsilon)`` in tiến độ.

        Returns:
            Thống kê huấn luyện tích lũy.
        """
        for ep in range(1, num_episodes + 1):
            reward_x = self.train_episode(opponent=opponent)
            if save_every > 0 and ep % save_every == 0:
                self.save_agent()
            if on_log is not None and log_every > 0 and ep % log_every == 0:
                on_log(ep, self.stats, self.epsilon, reward_x)
        return self.stats
