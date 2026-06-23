#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiến trúc mạng nơ-ron CNN cho Deep Q-Network (DQN).

Mạng nhận tensor ``(batch, 3, H, W)`` và trả về Q-value cho mọi ô trên bàn
cờ (``batch, H*W``).

THIẾT KẾ (fully-convolutional Q-head):
    Thân CNN nhiều tầng 3×3 (giữ nguyên kích thước nhờ padding) trích xuất đặc
    trưng cục bộ, rồi một lớp Conv 1×1 sinh đúng MỘT Q-value cho mỗi ô bàn cờ.

    Ưu điểm so với head Linear khổng lồ trước đây
    (``Linear(128·H·W, 256)`` ~ 7.4 triệu tham số cho bàn 15×15):
      • Bất biến tịnh tiến — học một "mẫu thế cờ" áp dụng được ở mọi vị trí,
        nên hội tụ với ÍT dữ liệu hơn nhiều bậc độ lớn.
      • Nhẹ (~vài trăm nghìn tham số) → suy luận & huấn luyện nhanh hơn hẳn.
      • Checkpoint nhỏ (vài trăm KB thay vì 30 MB).
"""

from __future__ import annotations

import torch
from torch import nn


class DQNNetwork(nn.Module):
    """Mạng CNN ước lượng Q(s, a) cho mọi hành động trên bàn cờ."""

    def __init__(
        self,
        board_size: int,
        in_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 5,
    ) -> None:
        """Khởi tạo thân convolution và Q-head 1×1.

        Args:
            board_size: Cạnh bàn cờ (H = W = board_size).
            in_channels: Số kênh đầu vào (mặc định 3: ta/đối/trống).
            channels: Số kênh đặc trưng mỗi tầng conv.
            num_blocks: Số tầng conv 3×3 (trường tiếp nhận ≈ 1 + 2·num_blocks).
        """
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(1, num_blocks) - 1):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)

        # Q-head: Conv 1×1 → 1 kênh = Q-value tại mỗi ô (bất biến tịnh tiến).
        self.q_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Tính Q-value cho mọi ô.

        Args:
            state: Tensor ``(B, 3, H, W)`` hoặc ``(3, H, W)``.

        Returns:
            Tensor Q-value ``(B, H*W)``.
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
        feat = self.features(state)
        q = self.q_head(feat)  # (B, 1, H, W)
        return q.view(q.size(0), -1)
