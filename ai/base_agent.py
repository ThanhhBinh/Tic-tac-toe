#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interface chung cho mọi tác nhân AI chơi Cờ Caro.

Mọi agent (Minimax, DQN, Hybrid, Random...) đều hiện thực `get_move`, nhận vào
môi trường hiện tại và trả về nước đi muốn thực hiện. UI và vòng lặp game chỉ
phụ thuộc vào interface này, không phụ thuộc cài đặt cụ thể.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from config import Player
from core.caro_env import CaroEnv
from core.constants import Move


class Agent(ABC):
    """Lớp cơ sở trừu tượng cho một tác nhân chơi cờ.

    Attributes:
        name: Tên hiển thị của agent (dùng cho HUD/log).
    """

    name: str = "Agent"

    @abstractmethod
    def get_move(self, env: CaroEnv) -> Move:
        """Chọn nước đi tốt nhất cho người chơi hiện tại của môi trường.

        Args:
            env: Môi trường hiện tại (KHÔNG được thay đổi trạng thái gốc;
                nếu cần mô phỏng hãy dùng ``env.clone()``).

        Returns:
            Nước đi (hàng, cột) hợp lệ.
        """

    def get_win_probability(
        self, env: CaroEnv, for_player: Player | None = None
    ) -> float | None:
        """Ước lượng xác suất thắng (mặc định: không hỗ trợ).

        Args:
            env: Môi trường hiện tại.
            for_player: Góc nhìn; None = ``env.current_player``.

        Returns:
            Xác suất trong [0, 1] hoặc None nếu agent không cung cấp.
        """
        return None
