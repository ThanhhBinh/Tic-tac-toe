#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tác nhân Deep Q-Network (DQN) cho Cờ Caro.

Kết hợp mạng CNN ước lượng Q-value, epsilon-greedy khi chọn nước, và các luật
tactical (thắng ngay / chặn thua) giống Minimax để chơi ổn định kể cả khi
model chưa huấn luyện đủ mạnh.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from ai.base_agent import Agent
from ai.board_encoder import (
    action_to_move,
    encode_board,
    legal_action_mask,
    mask_q_values,
    move_to_action,
)
from ai.dqn_model import DQNNetwork
from ai.heuristic import find_tactical_move
from config import (
    DQN_PLAY_EPSILON,
    Difficulty,
    Player,
    TacticalConfig,
    dqn_model_path,
)
from core.caro_env import CaroEnv
from core.constants import Move


def resolve_device(device: str | None = None) -> torch.device:
    """Chọn thiết bị tính toán (CUDA nếu có, ngược lại CPU).

    Args:
        device: Chuỗi ép buộc ('cpu' / 'cuda') hoặc None để tự phát hiện.

    Returns:
        Đối tượng ``torch.device`` sẵn sàng dùng.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon: MPS nhanh hơn CPU đáng kể cho forward DQN.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DQNAgent(Agent):
    """Agent chơi cờ bằng Deep Q-Network."""

    name = "DQN"

    def __init__(
        self,
        board_size: int,
        epsilon: float = 0.0,
        model_path: Path | None = None,
        device: str | None = None,
        seed: int | None = None,
        tactical_config: TacticalConfig | None = None,
    ) -> None:
        """Khởi tạo mạng Q và nạp checkpoint nếu có.

        Args:
            board_size: Cạnh bàn cờ (phải khớp với môi trường chơi).
            epsilon: Xác suất chọn ngẫu nhiên khi suy luận (0 = thuần greedy).
            model_path: Đường dẫn file ``.pth``; None thì dùng mặc định theo size.
            device: Thiết bị PyTorch ('cpu' / 'cuda' / None).
            seed: Hạt giống cho phần ngẫu nhiên epsilon-greedy.
        """
        self.board_size = board_size
        self.epsilon = epsilon
        self.device = resolve_device(device)
        self._rng = random.Random(seed)
        self.tactical_config = tactical_config or TacticalConfig()

        self.network = DQNNetwork(board_size).to(self.device)
        self._model_loaded = False

        path = model_path or dqn_model_path(board_size)
        if path.exists():
            self.load(path)

        status = "đã nạp model" if self._model_loaded else "chưa huấn luyện"
        self.name = f"DQN ({status}, ε={epsilon:.2f})"

    # ------------------------------------------------------------------
    #  SUY LUẬN
    # ------------------------------------------------------------------
    def get_move(self, env: CaroEnv) -> Move:
        """Chọn nước đi cho ``env.current_player``.

        Args:
            env: Môi trường hiện tại (không bị thay đổi).

        Returns:
            Nước đi hợp lệ.
        """
        player = env.current_player

        tactical = find_tactical_move(
            env,
            player,
            radius=2,
            config=self.tactical_config,
        )
        if tactical is not None:
            return tactical

        legal = env.candidate_moves(radius=2) or env.legal_moves()
        if not legal:
            return env.legal_moves()[0]

        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal)

        action = self._select_greedy_action(env, player)
        move = action_to_move(action, self.board_size)
        if env.is_legal(move):
            return move
        # Dự phòng nếu mạng chọn ô không hợp lệ.
        return legal[0]

    def get_win_probability(
        self, env: CaroEnv, for_player: Player | None = None
    ) -> float:
        """Ước lượng xác suất thắng: DQN (nếu đã train) hoặc heuristic.

        Args:
            env: Môi trường hiện tại.
            for_player: Góc nhìn; None = ``env.current_player``.

        Returns:
            Xác suất trong [0, 1].
        """
        player = for_player if for_player is not None else env.current_player
        if self._model_loaded:
            if for_player is not None and for_player is not env.current_player:
                trial = env.clone()
                trial.current_player = for_player
                return self._dqn_win_probability(trial)
            return self._dqn_win_probability(env)
        from ai.win_probability import estimate_win_probability

        return estimate_win_probability(env, player)

    def _dqn_win_probability(self, env: CaroEnv) -> float:
        """Softmax Q-value hợp lệ → xác suất cho HUD."""
        player = env.current_player
        q = self._predict_q_numpy(env, player)
        mask = legal_action_mask(env.board)
        if not mask.any():
            return 0.5

        legal_q = q[mask]
        if legal_q.size == 0 or not np.isfinite(legal_q).any():
            return 0.5

        shifted = legal_q - legal_q.max()
        exp_q = np.exp(shifted)
        probs = exp_q / exp_q.sum()
        return float(probs.max())

    def _select_greedy_action(self, env: CaroEnv, player: Player) -> int:
        """Chọn hành động có Q-value cao nhất trong các ô hợp lệ."""
        q = self._predict_q_numpy(env, player)
        masked = mask_q_values(q, legal_action_mask(env.board))
        return int(np.argmax(masked))

    def _predict_q_numpy(self, env: CaroEnv, player: Player) -> np.ndarray:
        """Chạy forward pass và trả Q-value dạng numpy 1D."""
        if env.size != self.board_size:
            return np.zeros(env.size * env.size, dtype=np.float32)
        state = encode_board(env.board, player)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.network.eval()
        with torch.inference_mode():
            q = self.network(tensor).squeeze(0).cpu().numpy()
        return q.astype(np.float32)

    # ------------------------------------------------------------------
    #  LƯU / NẠP MODEL
    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        """Lưu trọng số mạng Q ra file ``.pth``.

        Args:
            path: Đích lưu; None thì dùng đường dẫn mặc định theo board_size.

        Returns:
            Path file đã lưu.
        """
        dest = path or dqn_model_path(self.board_size)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "board_size": self.board_size,
                "state_dict": self.network.state_dict(),
            },
            dest,
        )
        return dest

    def load(self, path: Path) -> None:
        """Nạp trọng số từ checkpoint.

        Args:
            path: File ``.pth`` chứa state_dict.

        Raises:
            ValueError: Nếu board_size trong checkpoint không khớp.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved_size = int(checkpoint.get("board_size", self.board_size))
        if saved_size != self.board_size:
            raise ValueError(
                f"Checkpoint bàn {saved_size}x{saved_size} không khớp agent "
                f"{self.board_size}x{self.board_size}."
            )
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()
        self._model_loaded = True

    @classmethod
    def from_difficulty(
        cls,
        difficulty: Difficulty,
        board_size: int,
        device: str | None = None,
        tactical_config: TacticalConfig | None = None,
    ) -> "DQNAgent":
        """Tạo agent với epsilon suy luận theo mức độ khó.

        Args:
            difficulty: Độ khó (ánh xạ sang epsilon chơi).
            board_size: Kích thước bàn cờ.
            device: Thiết bị PyTorch.

        Returns:
            DQNAgent đã cấu hình.
        """
        epsilon = DQN_PLAY_EPSILON.get(difficulty, 0.0)
        return cls(
            board_size=board_size,
            epsilon=epsilon,
            device=device,
            tactical_config=tactical_config,
        )

    @staticmethod
    def action_index(env: CaroEnv, move: Move) -> int:
        """Tiện ích: chuyển nước đi sang chỉ số hành động."""
        return move_to_action(move, env.size)
