#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mã hoá trạng thái bàn cờ thành tensor đầu vào cho mạng DQN.

Biểu diễn 3 kênh theo góc nhìn người chơi đang đi:
    - Kênh 0: quân của ta
    - Kênh 1: quân đối thủ
    - Kênh 2: ô trống
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import Player
from core.constants import Board, Move

StateTensor = NDArray[np.float32]


def encode_board(board: Board, for_player: Player) -> StateTensor:
    """Chuyển bàn cờ sang tensor float32 shape ``(3, H, W)``.

    Args:
        board: Ma trận trạng thái int8.
        for_player: Góc nhìn người chơi (ta / đối thủ / trống).

    Returns:
        Tensor 3 kênh dtype float32.
    """
    opponent = for_player.opponent
    own = (board == for_player).astype(np.float32)
    opp = (board == opponent).astype(np.float32)
    empty = (board == Player.EMPTY).astype(np.float32)
    return np.stack([own, opp, empty], axis=0)


def move_to_action(move: Move, board_size: int) -> int:
    """Ánh xạ nước đi (row, col) sang chỉ số hành động phẳng.

    Args:
        move: Nước đi (hàng, cột).
        board_size: Cạnh bàn cờ.

    Returns:
        ``row * board_size + col``.
    """
    row, col = move
    return row * board_size + col


def action_to_move(action: int, board_size: int) -> Move:
    """Giải mã chỉ số hành động phẳng thành (row, col).

    Args:
        action: Chỉ số hành động.
        board_size: Cạnh bàn cờ.

    Returns:
        Nước đi tương ứng.
    """
    return (action // board_size, action % board_size)


def legal_action_mask(board: Board) -> NDArray[np.bool_]:
    """Tạo mặt nạ bool cho các hành động hợp lệ (ô trống).

    Args:
        board: Bàn cờ hiện tại.

    Returns:
        Vector bool length ``size*size``; True tại ô trống.
    """
    flat = (board.reshape(-1) == Player.EMPTY)
    return flat.astype(np.bool_)


def mask_q_values(
    q_values: NDArray[np.float32],
    mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Gán ``-inf`` cho Q-value của hành động không hợp lệ.

    Args:
        q_values: Vector Q-value gốc.
        mask: Mặt nạ hành động hợp lệ.

    Returns:
        Vector Q-value đã che (cùng shape).
    """
    masked = q_values.copy()
    masked[~mask] = -np.inf
    return masked
