#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ước lượng xác suất thắng cho HUD (heuristic fallback khi chưa có DQN)."""

from __future__ import annotations

import math

from config import Player
from core.caro_env import CaroEnv

from ai.heuristic import evaluate_position

# Hệ số scale: điểm heuristic (tam/tứ mở ~ hàng nghìn) → tanh mềm về [0, 1].
_HEURISTIC_SCALE: float = 8_000.0
# Tránh 0%/100% khi ván đang diễn ra (chưa kết thúc).
_LIVE_PROB_MIN: float = 0.05
_LIVE_PROB_MAX: float = 0.95


def estimate_win_probability(env: CaroEnv, for_player: Player) -> float:
    """Chuyển điểm heuristic sang xác suất thắng trong [0, 1].

    Dùng khi chưa train DQN hoặc agent Minimax thuần — chỉ là chỉ báo tương
    đối trên HUD, không phải xác suất thống kê thật.

    Args:
        env: Môi trường hiện tại.
        for_player: Góc nhìn người chơi cần ước lượng.

    Returns:
        Xác suất trong [0, 1].
    """
    if env.done:
        if env.is_draw:
            return 0.5
        if env.winner is for_player:
            return 1.0
        if env.winner is for_player.opponent:
            return 0.0
        return 0.5

    score = evaluate_position(env.winner, env.board, for_player)
    # tanh: điểm dương lớn → gần 1, âm lớn → gần 0, quanh 0 → ~50%.
    scaled = math.tanh(score / _HEURISTIC_SCALE)
    prob = 0.5 + 0.5 * scaled
    return max(_LIVE_PROB_MIN, min(_LIVE_PROB_MAX, prob))
