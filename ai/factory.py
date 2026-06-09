#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Factory tạo tác nhân AI theo cấu hình người dùng chọn.

Tập trung việc khởi tạo agent tại một nơi để UI không phụ thuộc trực tiếp vào
từng lớp agent cụ thể.
"""

from __future__ import annotations

from ai.base_agent import Agent
from ai.dqn_agent import DQNAgent
from ai.hybrid_agent import HybridAgent
from ai.minimax_agent import MinimaxAgent
from config import DEFAULT_BOARD_SIZE, AIType, Difficulty, TacticalConfig


def create_agent(
    ai_type: AIType,
    difficulty: Difficulty,
    board_size: int = DEFAULT_BOARD_SIZE,
    tactical_config: TacticalConfig | None = None,
) -> Agent:
    """Tạo một agent dựa trên loại AI, độ khó và kích thước bàn cờ.

    Args:
        ai_type: Loại AI (Minimax / DQN / Hybrid).
        difficulty: Độ khó (depth Minimax hoặc epsilon DQN).
        board_size: Cạnh bàn cờ (10 hoặc 15).
        tactical_config: Luật chặn 2 đầu / chế độ tấn công.

    Returns:
        Một thể hiện Agent sẵn sàng chơi.
    """
    cfg = tactical_config or TacticalConfig()

    if ai_type is AIType.MINIMAX:
        return MinimaxAgent.from_difficulty(difficulty, tactical_config=cfg)

    if ai_type is AIType.DQN:
        return DQNAgent.from_difficulty(
            difficulty, board_size=board_size, tactical_config=cfg
        )

    if ai_type is AIType.HYBRID:
        return HybridAgent.from_difficulty(
            difficulty, board_size=board_size, tactical_config=cfg
        )

    return MinimaxAgent.from_difficulty(difficulty, tactical_config=cfg)
