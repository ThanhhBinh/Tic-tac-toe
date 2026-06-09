#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiến trúc mạng nơ-ron CNN cho Deep Q-Network (DQN).

Mạng nhận tensor ``(batch, 3, H, W)`` và trả về Q-value cho mọi ô trên bàn
 cờ (``batch, H*W``). Dùng CNN nhẹ để trích xuất đặc trưng cục bộ quanh quân.
"""

from __future__ import annotations

import torch
from torch import nn


class DQNNetwork(nn.Module):
    """Mạng CNN ước lượng Q(s, a) cho mọi hành động trên bàn cờ."""

    def __init__(self, board_size: int, in_channels: int = 3) -> None:
        """Khởi tạo các lớp convolution và fully-connected.

        Args:
            board_size: Cạnh bàn cờ (H = W = board_size).
            in_channels: Số kênh đầu vào (mặc định 3: ta/đối/thừa).
        """
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        flat_dim = 128 * board_size * board_size
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Tính Q-value cho mọi ô.

        Args:
            state: Tensor ``(B, 3, H, W)`` hoặc ``(3, H, W)``.

        Returns:
            Tensor Q-value ``(B, H*W)`` hoặc ``(H*W,)`` nếu batch=1 1D input.
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
        batch = self.features(state)
        batch = batch.view(batch.size(0), -1)
        return self.head(batch)
