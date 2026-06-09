#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gói `core` — logic game Cờ Caro thuần (backend).

Tách biệt hoàn toàn với giao diện: KHÔNG import pygame ở bất kỳ đâu trong gói
này, để có thể dùng cho huấn luyện, kiểm thử và đánh giá AI mà không cần UI.
"""

from core.caro_env import CaroEnv
from core.constants import Board, Move

__all__ = ["CaroEnv", "Board", "Move"]
