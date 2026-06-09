#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tác nhân chọn nước đi ngẫu nhiên — PLACEHOLDER tạm thời.

Mục đích: giúp khung UI (chế độ Player vs AI / AI vs AI) chạy được NGAY trong
giai đoạn xây dựng, trước khi các thuật toán thật (Minimax/DQN/Hybrid) hoàn
thành ở Bước 2-5. Khi đó, agent này sẽ được thay thế.
"""

from __future__ import annotations

import random

from ai.base_agent import Agent
from core.caro_env import CaroEnv
from core.constants import Move


class RandomAgent(Agent):
    """Chọn ngẫu nhiên một nước đi trong số các ô ứng viên gần quân đã đặt."""

    name = "Random (tạm thời)"

    def __init__(self, seed: int | None = None) -> None:
        """Khởi tạo bộ sinh số ngẫu nhiên.

        Args:
            seed: Hạt giống ngẫu nhiên để tái lập (None = ngẫu nhiên thật).
        """
        self._rng = random.Random(seed)

    def get_move(self, env: CaroEnv) -> Move:
        """Trả về một nước đi hợp lệ chọn ngẫu nhiên.

        Ưu tiên các ô ứng viên (gần quân đã đặt) để nước đi đỡ "lạc lõng";
        nếu không có thì lấy bất kỳ ô hợp lệ nào.
        """
        candidates = env.candidate_moves(radius=1) or env.legal_moves()
        return self._rng.choice(candidates)
