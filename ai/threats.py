#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phân tích đe dọa trên bàn cờ — cảnh báo sắp thắng / chặn 2 đầu cho UI."""

from __future__ import annotations

from dataclasses import dataclass

from config import Player, TacticalConfig
from core.caro_env import CaroEnv
from core.constants import Board, DIRECTIONS, Move

from ai.heuristic import (
    _count_stones,
    _is_four_with_open_end,
    _is_open_three_at,
    find_blocking_move,
    find_open_four_block,
    find_winning_move,
)


@dataclass
class ThreatAnalysis:
    """Kết quả phân tích đe dọa cho một người chơi (thường là người thật).

    Attributes:
        win_moves: Các ô đặt quân sẽ thắng ngay lượt này.
        block_moves: Các ô bắt buộc chặn (đối thủ thắng nếu không chặn).
        double_end_blocks: Ô thuộc mối đe dọa «chặn 2 đầu» (tứ/tam mở hai đầu).
        threat_stones: Quân đối thủ nằm trên hàng đe dọa (tô đỏ trên UI).
        message: Mô tả ngắn cho HUD.
    """

    win_moves: list[Move]
    block_moves: list[Move]
    double_end_blocks: list[Move]
    threat_stones: list[Move]
    message: str


def _open_end_cells(
    board: Board, row: int, col: int, dr: int, dc: int, player: Player, size: int
) -> list[Move]:
    """Trả về các ô trống ở hai đầu hàng quân ``player`` qua (row,col)."""
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


def _collect_double_end_cells(
    board: Board, player: Player, size: int, min_open_ends: int = 2
) -> list[Move]:
    """Gom ô trống cần chú ý khi ``player`` có hàng 3/4 mở đủ đầu.

    Args:
        board: Bàn cờ hiện tại.
        player: Màu quân đang xét (thường là đối thủ).
        size: Cạnh bàn.
        min_open_ends: Số đầu mở tối thiểu (2 = luật chặn 2 đầu).
    """
    found: set[Move] = set()
    for row in range(size):
        for col in range(size):
            if board[row, col] != player:
                continue
            for dr, dc in DIRECTIONS:
                fwd = _count_stones(board, row, col, dr, dc, player, size)
                bwd = _count_stones(board, row, col, -dr, -dc, player, size)
                total = 1 + fwd + bwd
                if total not in (3, 4):
                    continue
                ends = _open_end_cells(board, row, col, dr, dc, player, size)
                if len(ends) >= min_open_ends:
                    found.update(ends)
    return sorted(found)


def _collect_double_end_stone_cells(
    board: Board, player: Player, size: int, min_open_ends: int = 2
) -> list[Move]:
    """Gom quân ``player`` trên hàng tam/tứ mở hai đầu (để UI tô đỏ)."""
    found: set[Move] = set()
    for row in range(size):
        for col in range(size):
            if board[row, col] != player:
                continue
            for dr, dc in DIRECTIONS:
                fwd = _count_stones(board, row, col, dr, dc, player, size)
                bwd = _count_stones(board, row, col, -dr, -dc, player, size)
                total = 1 + fwd + bwd
                if total not in (3, 4):
                    continue
                ends = _open_end_cells(board, row, col, dr, dc, player, size)
                if len(ends) < min_open_ends:
                    continue
                for i in range(-bwd, fwd + 1):
                    nr, nc = row + i * dr, col + i * dc
                    if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == player:
                        found.add((nr, nc))
    return sorted(found)


def _unique_moves(moves: list[Move]) -> list[Move]:
    """Loại trùng, giữ thứ tự."""
    seen: set[Move] = set()
    out: list[Move] = []
    for move in moves:
        if move not in seen:
            seen.add(move)
            out.append(move)
    return out


def analyze_threats(
    env: CaroEnv,
    player: Player,
    config: TacticalConfig | None = None,
) -> ThreatAnalysis:
    """Phân tích đe dọa cho ``player`` (thắng ngay, chặn bắt buộc, chặn 2 đầu).

    Args:
        env: Môi trường hiện tại.
        player: Người chơi cần cảnh báo (thường là người thật).
        config: Cấu hình luật chiến thuật.

    Returns:
        ThreatAnalysis dùng cho web/Pygame highlight.
    """
    cfg = config or TacticalConfig()
    opponent = player.opponent
    radius = 3

    win_moves: list[Move] = []
    for move in env.candidate_moves(radius=radius):
        if env.board[move[0], move[1]] != Player.EMPTY:
            continue
        trial = env.clone()
        trial.current_player = player
        trial.step(move)
        if trial.winner is player:
            win_moves.append(move)
    win_moves = _unique_moves(win_moves)

    block_moves: list[Move] = []
    block_one = find_blocking_move(env, player, radius=radius)
    if block_one is not None:
        for move in env.candidate_moves(radius=radius):
            if env.board[move[0], move[1]] != Player.EMPTY:
                continue
            trial = env.clone()
            trial.current_player = opponent
            trial.step(move)
            if trial.winner is opponent:
                block_moves.append(move)
    block_moves = _unique_moves(block_moves)

    double_end_blocks: list[Move] = []
    threat_stones: list[Move] = []
    if cfg.double_end_block_rule:
        double_end_blocks = _collect_double_end_cells(
            env.board, opponent, env.size, min_open_ends=2
        )
        # Bổ sung ô chặn tứ mở sắp hình thành (một nước nữa là thua).
        open_four = find_open_four_block(env, player, radius=radius)
        if open_four is not None and open_four not in double_end_blocks:
            double_end_blocks.append(open_four)
        double_end_blocks = _unique_moves(double_end_blocks)
        threat_stones = _collect_double_end_stone_cells(
            env.board, opponent, env.size, min_open_ends=2
        )

    message = _build_message(win_moves, block_moves, double_end_blocks, cfg)
    return ThreatAnalysis(
        win_moves=win_moves,
        block_moves=block_moves,
        double_end_blocks=double_end_blocks,
        threat_stones=threat_stones,
        message=message,
    )


def _build_message(
    win_moves: list[Move],
    block_moves: list[Move],
    double_end_blocks: list[Move],
    config: TacticalConfig,
) -> str:
    """Tạo chuỗi cảnh báo ngắn cho HUD."""
    if win_moves:
        return f"Bạn có thể thắng ngay ({len(win_moves)} ô)"
    if block_moves:
        return f"Chặn ngay — đối thủ sắp thắng ({len(block_moves)} ô)"
    if config.double_end_block_rule and len(double_end_blocks) >= 2:
        return f"Chặn 2 đầu — {len(double_end_blocks)} ô nguy hiểm"
    if config.double_end_block_rule and double_end_blocks:
        return "Đe dọa tứ/tam mở — nên chặn sớm"
    return ""


def _is_open_three_both_ends(
    board: Board, row: int, col: int, player: Player, size: int
) -> bool:
    """True nếu quân tại (row,col) nằm trên tam mở hai đầu."""
    if board[row, col] != player:
        return False
    for dr, dc in DIRECTIONS:
        fwd = _count_stones(board, row, col, dr, dc, player, size)
        bwd = _count_stones(board, row, col, -dr, -dc, player, size)
        if 1 + fwd + bwd != 3:
            continue
        ends = _open_end_cells(board, row, col, dr, dc, player, size)
        if len(ends) >= 2:
            return True
    return False


def find_open_three_attack(env: CaroEnv, player: Player, radius: int = 2) -> Move | None:
    """Tìm nước tạo tam mở (ưu tiên tam mở hai đầu) để tấn công.

    Args:
        env: Môi trường hiện tại.
        player: Người tấn công.
        radius: Bán kính ứng viên.

    Returns:
        Nước tấn công tốt nhất hoặc None.
    """
    from ai.heuristic import _wins_with_move, move_priority

    best: Move | None = None
    best_score = float("-inf")
    board = env.board
    size = env.size
    for move in env.candidate_moves(radius=radius):
        row, col = move
        if board[row, col] != Player.EMPTY:
            continue
        if _wins_with_move(env, move, player):
            return move
        board[row, col] = player
        is_three = _is_open_three_at(board, row, col, player, size)
        both_ends = is_three and _is_open_three_both_ends(
            board, row, col, player, size
        )
        board[row, col] = Player.EMPTY
        if not is_three:
            continue
        score = move_priority(board, move, player)
        if both_ends:
            score += 8000.0
        if score > best_score:
            best_score = score
            best = move
    return best


def find_open_three_block_double_end(
    env: CaroEnv, player: Player, radius: int = 2
) -> Move | None:
    """Chặn tam/tứ mở hai đầu của đối thủ (luật chặn 2 đầu).

    Args:
        env: Môi trường hiện tại.
        player: Người phòng thủ.
        radius: Bán kính ứng viên.

    Returns:
        Nước chặn hoặc None.
    """
    from ai.heuristic import _wins_with_move, move_priority

    opponent = player.opponent
    best: Move | None = None
    best_score = float("-inf")
    board = env.board
    size = env.size
    for move in env.candidate_moves(radius=radius):
        row, col = move
        if board[row, col] != Player.EMPTY:
            continue
        if _wins_with_move(env, move, opponent):
            return move
        board[row, col] = opponent
        threat = _is_four_with_open_end(
            board, row, col, opponent, size
        ) or _is_open_three_both_ends(board, row, col, opponent, size)
        board[row, col] = Player.EMPTY
        if not threat:
            continue
        score = move_priority(board, move, player)
        if score > best_score:
            best_score = score
            best = move
    return best
