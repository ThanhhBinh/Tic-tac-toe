#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VCF / VCT — tìm chuỗi thắng ép buộc (Victory by Continuous Forcing/Threat).

Chỉ xét nước tạo đe dọa (thắng ngay, tứ mở/kín, tam mở) nên branching factor
nhỏ (3–5) thay vì 30+ ô ứng viên — cho phép search sâu 8–10 ply xen kẽ nhanh.
"""

from __future__ import annotations

import time

from config import Player
from core.caro_env import CaroEnv
from core.constants import Board, DIRECTIONS, Move

from ai.heuristic import (
    _count_stones,
    _is_open_three_at,
    _wins_with_move,
    find_winning_move,
)


def _open_end_cells(
    board: Board, row: int, col: int, dr: int, dc: int, player: Player, size: int
) -> list[Move]:
    """Các ô trống ở hai đầu hàng quân ``player`` qua (row, col)."""
    fwd = _count_stones(board, row, col, dr, dc, player, size)
    bwd = _count_stones(board, row, col, -dr, -dc, player, size)
    ends: list[Move] = []
    er, ec = row + dr * (fwd + 1), col + dc * (fwd + 1)
    br, bc = row - dr * (bwd + 1), col - dc * (bwd + 1)
    if 0 <= er < size and 0 <= ec < size and board[er, ec] == Player.EMPTY:
        ends.append((er, ec))
    if 0 <= br < size and 0 <= bc < size and board[br, bc] == Player.EMPTY:
        ends.append((br, bc))
    return ends


def _line_info_at(
    board: Board, row: int, col: int, dr: int, dc: int, player: Player, size: int
) -> tuple[int, list[Move]] | None:
    """(số quân liên tiếp, ô trống ở hai đầu) nếu (row,col) thuộc hàng ``player``."""
    if board[row, col] != player:
        return None
    fwd = _count_stones(board, row, col, dr, dc, player, size)
    bwd = _count_stones(board, row, col, -dr, -dc, player, size)
    total = 1 + fwd + bwd
    if total < 3:
        return None
    return total, _open_end_cells(board, row, col, dr, dc, player, size)


def _is_open_four_both_ends(
    board: Board, row: int, col: int, player: Player, size: int
) -> bool:
    """True nếu quân tại (row,col) nằm trên tứ mở hai đầu."""
    if board[row, col] != player:
        return False
    for dr, dc in DIRECTIONS:
        info = _line_info_at(board, row, col, dr, dc, player, size)
        if info is None:
            continue
        total, ends = info
        if total == 4 and len(ends) >= 2:
            return True
    return False


def _is_closed_four_at(
    board: Board, row: int, col: int, player: Player, size: int
) -> bool:
    """True nếu quân tại (row,col) nằm trên tứ kín (đúng 4 quân, 1 đầu mở)."""
    if board[row, col] != player:
        return False
    for dr, dc in DIRECTIONS:
        info = _line_info_at(board, row, col, dr, dc, player, size)
        if info is None:
            continue
        total, ends = info
        if total == 4 and len(ends) == 1:
            return True
    return False


def _creates_open_four(
    board: Board, row: int, col: int, player: Player, size: int
) -> bool:
    """True sau khi đặt quân tại (row,col) tạo tứ mở (≥1 đầu mở, thường 2)."""
    if board[row, col] != player:
        return False
    for dr, dc in DIRECTIONS:
        info = _line_info_at(board, row, col, dr, dc, player, size)
        if info is None:
            continue
        total, ends = info
        if total == 4 and len(ends) >= 1:
            return True
    return False


def _creates_open_three(
    board: Board, row: int, col: int, player: Player, size: int
) -> bool:
    """True sau khi đặt quân tại (row,col) tạo tam mở."""
    return _is_open_three_at(board, row, col, player, size)


def _unique_moves(moves: list[Move]) -> list[Move]:
    seen: set[Move] = set()
    out: list[Move] = []
    for move in moves:
        if move not in seen:
            seen.add(move)
            out.append(move)
    return out


def find_threat_moves(
    env: CaroEnv,
    player: Player,
    *,
    include_threes: bool = False,
    radius: int = 2,
) -> list[Move]:
    """Các nước tấn công hợp lệ cho VCF/VCT (thắng ngay → tứ → tam).

    Dùng đặt/gỡ quân tại chỗ trên ``env.board`` — không clone env.
    """
    board = env.board
    size = env.size
    wins: list[Move] = []
    open_fours: list[Move] = []
    closed_fours: list[Move] = []
    open_threes: list[Move] = []

    for move in env.candidate_moves(radius=radius):
        row, col = move
        if board[row, col] != Player.EMPTY:
            continue
        if _wins_with_move(env, move, player):
            wins.append(move)
            continue

        board[row, col] = player
        if _creates_open_four(board, row, col, player, size):
            if _is_open_four_both_ends(board, row, col, player, size):
                open_fours.append(move)
            else:
                closed_fours.append(move)
        elif include_threes and _creates_open_three(board, row, col, player, size):
            open_threes.append(move)
        board[row, col] = Player.EMPTY

    if wins:
        return _unique_moves(wins)
    return _unique_moves(open_fours + closed_fours + open_threes)


def _collect_four_blocks_from_move(
    board: Board, last_move: Move, attacker: Player, size: int
) -> list[Move]:
    """Ô bắt buộc chặn tứ mở/kín do ``attacker`` vừa tạo tại ``last_move``."""
    row, col = last_move
    blocks: list[Move] = []
    for dr, dc in DIRECTIONS:
        info = _line_info_at(board, row, col, dr, dc, attacker, size)
        if info is None:
            continue
        total, ends = info
        if total == 4 and ends:
            blocks.extend(ends)
    return _unique_moves(blocks)


def _collect_three_blocks_from_move(
    board: Board, last_move: Move, attacker: Player, size: int
) -> list[Move]:
    """Ô chặn tam mở (các đầu mở của hàng 3) do ``attacker`` vừa tạo."""
    row, col = last_move
    blocks: list[Move] = []
    for dr, dc in DIRECTIONS:
        info = _line_info_at(board, row, col, dr, dc, attacker, size)
        if info is None:
            continue
        total, ends = info
        if total == 3 and ends:
            blocks.extend(ends)
    return _unique_moves(blocks)


def find_forced_defenses(
    env: CaroEnv,
    attacker_move: Move,
    defender: Player,
    *,
    include_threes: bool = False,
    radius: int = 2,
) -> list[Move]:
    """Sau khi đối thủ đi ``attacker_move``, nước phòng thủ BẮT BUỘC.

    ``env`` phải ở trạng thái SAU nước tấn công. Trả ``[]`` nếu không có đe dọa
    bắt buộc → nhánh VCF/VCT thất bại.
    """
    win = find_winning_move(env, defender, radius=radius)
    if win is not None:
        return [win]

    attacker = defender.opponent
    board = env.board
    size = env.size

    blocks = _collect_four_blocks_from_move(board, attacker_move, attacker, size)
    if blocks:
        return blocks

    if include_threes:
        three_blocks = _collect_three_blocks_from_move(
            board, attacker_move, attacker, size
        )
        if three_blocks:
            return three_blocks

    return []


def vcf_search(
    env: CaroEnv,
    player: Player,
    max_depth: int = 10,
    deadline: float | None = None,
    *,
    radius: int = 2,
) -> list[Move] | None:
    """Tìm chuỗi thắng ép buộc (chỉ nước tạo tứ / thắng ngay).

    ``max_depth`` = tối đa số nước xen kẽ (tấn công + phòng thủ) còn lại.
    """
    return _threat_space_search(
        env,
        player,
        max_depth,
        include_threes=False,
        radius=radius,
        deadline=deadline,
    )


def vct_search(
    env: CaroEnv,
    player: Player,
    max_depth: int = 12,
    deadline: float | None = None,
    *,
    radius: int = 2,
) -> list[Move] | None:
    """VCT — mở rộng VCF với nước tạo tam mở."""
    return _threat_space_search(
        env,
        player,
        max_depth,
        include_threes=True,
        radius=radius,
        deadline=deadline,
    )


def _threat_space_search(
    env: CaroEnv,
    player: Player,
    max_depth: int,
    *,
    include_threes: bool,
    radius: int,
    deadline: float | None = None,
) -> list[Move] | None:
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    if max_depth <= 0 or env.done:
        return None

    attacks = find_threat_moves(
        env, player, include_threes=include_threes, radius=radius
    )
    if not attacks:
        return None

    defender = player.opponent

    for attack in attacks:
        env.push(attack)
        try:
            if env.winner is player:
                return [attack]

            defenses = find_forced_defenses(
                env,
                attack,
                defender,
                include_threes=include_threes,
                radius=radius,
            )
            if not defenses:
                continue

            continuation: list[Move] | None = None
            all_defenses_beaten = True

            for defense in defenses:
                env.push(defense)
                try:
                    if env.winner is defender:
                        all_defenses_beaten = False
                        break
                    sub = _threat_space_search(
                        env,
                        player,
                        max_depth - 2,
                        include_threes=include_threes,
                        radius=radius,
                        deadline=deadline,
                    )
                finally:
                    env.pop()

                if sub is None:
                    all_defenses_beaten = False
                    break
                continuation = sub

            if all_defenses_beaten and continuation is not None:
                return [attack] + continuation
        finally:
            env.pop()

    return None
