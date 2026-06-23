#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gói `ai` — các tác nhân chơi cờ (Minimax, DQN, Hybrid).

Tầng này KHÔNG import pygame để có thể chạy huấn luyện & đánh giá độc lập.
"""

from ai.base_agent import Agent
from ai.minimax_agent import MinimaxAgent
from ai.random_agent import RandomAgent

try:
    from ai.dqn_agent import DQNAgent
    from ai.hybrid_agent import HybridAgent
except ModuleNotFoundError:
    DQNAgent = None  # type: ignore[assignment,misc]
    HybridAgent = None  # type: ignore[assignment,misc]

__all__ = ["Agent", "DQNAgent", "HybridAgent", "MinimaxAgent", "RandomAgent"]
