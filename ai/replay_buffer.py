#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bộ đệm kinh nghiệm (Experience Replay) cho huấn luyện DQN.

Lưu các bộ (state, action, reward, next_state, done) và lấy mẫu ngẫu nhiên
mini-batch để phá vỡ tương quan thời gian giữa các transition liên tiếp.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random

import numpy as np
from numpy.typing import NDArray

from ai.board_encoder import StateTensor


@dataclass(slots=True)
class Transition:
    """Một transition đơn lẻ trong replay buffer.

    Attributes:
        state: Trạng thái trước khi hành động (3, H, W).
        action: Chỉ số hành động phẳng.
        reward: Phần thưởng nhận được.
        next_state: Trạng thái sau hành động.
        done: True nếu episode kết thúc.
    """

    state: StateTensor
    action: int
    reward: float
    next_state: StateTensor
    done: bool


class ReplayBuffer:
    """Vòng đệm cố định lưu transition với lấy mẫu uniform ngẫu nhiên."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        """Khởi tạo buffer với sức chứa tối đa.

        Args:
            capacity: Số transition tối đa (FIFO khi đầy).
            seed: Hạt giống cho bộ lấy mẫu ngẫu nhiên.
        """
        self.capacity = capacity
        self._data: deque[Transition] = deque(maxlen=capacity)
        self._rng = Random(seed)

    def __len__(self) -> int:
        """Số transition hiện có trong buffer."""
        return len(self._data)

    def push(self, transition: Transition) -> None:
        """Thêm một transition vào buffer.

        Args:
            transition: Bộ dữ liệu cần lưu.
        """
        self._data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Lấy ngẫu nhiên một mini-batch.

        Args:
            batch_size: Kích thước batch (không vượt quá len(buffer)).

        Returns:
            Danh sách transition được lấy mẫu.

        Raises:
            ValueError: Nếu buffer chưa đủ phần tử.
        """
        if len(self._data) < batch_size:
            raise ValueError(
                f"Buffer chỉ có {len(self._data)} phần tử, cần ít nhất {batch_size}."
            )
        return self._rng.sample(list(self._data), batch_size)

    def states_batch(self, transitions: list[Transition]) -> NDArray[np.float32]:
        """Gom state của batch thành ndarray ``(B, 3, H, W)``."""
        return np.stack([t.state for t in transitions], axis=0)

    def next_states_batch(self, transitions: list[Transition]) -> NDArray[np.float32]:
        """Gom next_state của batch thành ndarray ``(B, 3, H, W)``."""
        return np.stack([t.next_state for t in transitions], axis=0)
