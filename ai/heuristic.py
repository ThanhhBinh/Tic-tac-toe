#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hàm lượng giá (heuristic) cho bàn cờ Caro.

Chiến lược: quét mọi hàng/cột/đường chéo, chuyển thành chuỗi ký tự rồi khớp
các mẫu quân liên tiếp (5, mở 4, kín 4, mở 3...) theo góc nhìn từng người
chơi. Điểm cuối = điểm ta − điểm đối thủ (cân bằng tấn công/phòng thủ).
"""

from __future__ import annotations

import re

from config import Player, TacticalConfig
from core.caro_env import CaroEnv
from core.constants import DIRECTIONS, Board, Move

# Điểm cho từng mẫu (quân ta = '1', đối thủ = '2', trống = '0', biên = '3').
# Thứ tự khớp từ cao xuống thấp để ưu tiên mẫu mạnh hơn.
_PATTERN_SCORES: tuple[tuple[str, int], ...] = (
    ("11111", 1_000_000),      # Năm quân — thắng
    ("011110", 50_000),        # Bốn mở (hai đầu trống)
    ("211110", 5_000),         # Bốn kín (một đầu bị chặn)
    ("011112", 5_000),
    ("011100", 3_000),         # Ba mở
    ("001110", 3_000),
    ("011010", 1_500),         # Ba nhảy (broken three)
    ("010110", 1_500),
    ("211100", 500),           # Ba kín
    ("001112", 500),
    ("01100", 200),            # Hai mở
    ("00110", 200),
    ("01010", 80),             # Hai nhảy
    ("0100", 20),              # Một mở
    ("0010", 20),
)

# Biên compile regex một lần để tăng tốc khi quét nhiều dòng.
_COMPILED_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = tuple(
    (re.compile(pat), score) for pat, score in _PATTERN_SCORES
)


def _player_line(line: str, player: Player, opponent: Player) -> str:
    """Chuyển chuỗi ô số sang '0/1/2/3' theo góc nhìn ``player``.

    Args:
        line: Chuỗi ký tự digit ('0','1','2') biểu diễn một hàng/cột/chéo.
        player: Người chơi đang được chấm điểm.
        opponent: Đối thủ.

    Returns:
        Chuỗi chỉ gồm '0','1','2','3' (3 = biên ảo ngoài bàn cờ).
    """
    mapped = line.replace(str(int(player)), "1").replace(str(int(opponent)), "2")
    return mapped.replace("0", "0")


def _score_line(line: str) -> int:
    """Cộng điểm mọi mẫu khớp trên một dòng đã chuẩn hoá ('0/1/2/3')."""
    total = 0
    for pattern, value in _COMPILED_PATTERNS:
        total += len(pattern.findall(line)) * value
    return total


def _extract_lines(board: Board, size: int) -> list[str]:
    """Trích mọi hàng, cột và đường chéo thành chuỗi digit.

    Mỗi đường chéo được bọc thêm ký tự biên '3' hai đầu để phân biệt mẫu
    bị chặn sát mép bàn cờ với mẫu mở thật sự.
    """
    lines: list[str] = []

    # Hàng ngang & cột dọc.
    for row in range(size):
        lines.append("".join(str(int(board[row, col])) for col in range(size)))
    for col in range(size):
        lines.append("".join(str(int(board[row, col])) for row in range(size)))

    # Chéo xuống-phải: duyệt mọi đường chéo (tổng cộng 2*size-1 đường).
    for start_col in range(size):
        chars: list[str] = ["3"]
        row, col = 0, start_col
        while row < size and col < size:
            chars.append(str(int(board[row, col])))
            row += 1
            col += 1
        chars.append("3")
        lines.append("".join(chars))

    for start_row in range(1, size):
        chars = ["3"]
        row, col = start_row, 0
        while row < size and col < size:
            chars.append(str(int(board[row, col])))
            row += 1
            col += 1
        chars.append("3")
        lines.append("".join(chars))

    # Chéo lên-phải.
    for start_col in range(size):
        chars = ["3"]
        row, col = size - 1, start_col
        while row >= 0 and col < size:
            chars.append(str(int(board[row, col])))
            row -= 1
            col += 1
        chars.append("3")
        lines.append("".join(chars))

    for start_row in range(size - 2, -1, -1):
        chars = ["3"]
        row, col = start_row, 0
        while row >= 0 and col < size:
            chars.append(str(int(board[row, col])))
            row -= 1
            col += 1
        chars.append("3")
        lines.append("".join(chars))

    return lines


def score_player(board: Board, player: Player) -> int:
    """Tính tổng điểm mẫu quân cho một người chơi trên toàn bàn cờ.

    Args:
        board: Ma trận trạng thái bàn cờ.
        player: Người chơi cần chấm.

    Returns:
        Tổng điểm heuristic (càng cao càng có lợi thế).
    """
    if player is Player.EMPTY:
        return 0
    opponent = player.opponent
    size = board.shape[0]
    total = 0
    for raw_line in _extract_lines(board, size):
        total += _score_line(_player_line(raw_line, player, opponent))
    return total


def evaluate_board(board: Board, for_player: Player) -> float:
    """Lượng giá thế cờ từ góc nhìn ``for_player`` (ta − đối thủ).

    Args:
        board: Ma trận bàn cờ hiện tại.
        for_player: Người chơi cần tối đa hoá điểm.

    Returns:
        Điểm số float; dương = có lợi cho ``for_player``.
    """
    opponent = for_player.opponent
    return float(score_player(board, for_player) - score_player(board, opponent))


def evaluate_position(env_winner: Player | None, board: Board, for_player: Player) -> float:
    """Lượng giá vị trí kèm xử lý trạng thái kết thúc (thắng/thua).

    Args:
        env_winner: Người thắng nếu ván đã kết thúc, None nếu chưa.
        board: Bàn cờ hiện tại.
        for_player: Góc nhìn đánh giá.

    Returns:
        Điểm lớn ± nếu thắng/thua, ngược lại heuristic thường.
    """
    win_score = 10.0**9
    if env_winner is for_player:
        return win_score
    if env_winner is for_player.opponent:
        return -win_score
    return evaluate_board(board, for_player)


def _lines_through_cell(
    board: Board, row: int, col: int, size: int
) -> tuple[str, str, str, str]:
    """Trả 4 chuỗi digit của 4 đường (ngang/dọc/2 chéo) đi qua (row, col).

    Đường chéo được bọc biên '3' hai đầu — đồng nhất với ``_extract_lines`` để
    điểm cục bộ khớp phong cách chấm của heuristic toàn cục.
    """
    horiz = "".join(str(int(board[row, c])) for c in range(size))
    vert = "".join(str(int(board[r, col])) for r in range(size))

    # Chéo xuống-phải qua ô: lùi về đầu đường rồi quét tới cuối.
    r, c = row, col
    while r > 0 and c > 0:
        r -= 1
        c -= 1
    chars = ["3"]
    while r < size and c < size:
        chars.append(str(int(board[r, c])))
        r += 1
        c += 1
    chars.append("3")
    diag_dr = "".join(chars)

    # Chéo lên-phải qua ô.
    r, c = row, col
    while r < size - 1 and c > 0:
        r += 1
        c -= 1
    chars = ["3"]
    while r >= 0 and c < size:
        chars.append(str(int(board[r, c])))
        r -= 1
        c += 1
    chars.append("3")
    diag_ur = "".join(chars)

    return horiz, vert, diag_dr, diag_ur


def move_priority(board: Board, move: tuple[int, int], player: Player) -> float:
    """Ước lượng nhanh chất lượng một nước đi (dùng sắp xếp nước đi).

    Tối ưu: chỉ chấm 4 đường thẳng đi qua ô vừa đặt (ta − đối thủ) thay vì
    ``board.copy()`` + quét toàn bộ bàn cờ. Đây là proxy cục bộ của Δ điểm
    heuristic, đủ tốt để xếp hạng nước cho Alpha-Beta và nhanh hơn nhiều bậc.

    Args:
        board: Bàn cờ trước khi đặt (được khôi phục nguyên trạng sau khi tính).
        move: Nước đi cần xếp hạng.
        player: Người chơi sắp đặt quân.

    Returns:
        Điểm ưu tiên càng cao càng nên thử trước trong Alpha-Beta.
    """
    row, col = move
    opponent = player.opponent
    size = board.shape[0]
    old = board[row, col]
    board[row, col] = player
    total = 0
    for raw in _lines_through_cell(board, row, col, size):
        total += _score_line(_player_line(raw, player, opponent))
        total -= _score_line(_player_line(raw, opponent, player))
    board[row, col] = old
    return float(total)


def _wins_with_move(env: CaroEnv, move: Move, player: Player) -> bool:
    """True nếu ``player`` đặt quân tại ``move`` sẽ thắng ngay (theo luật env).

    Dùng push/pop (O(win_length)) thay vì ``env.clone()`` — không cấp phát bàn
    cờ mới, vẫn tôn trọng ``double_end_block_rule``.
    """
    row, col = move
    if env.done or env.board[row, col] != Player.EMPTY:
        return False
    prev_player = env.current_player
    env.current_player = player
    env.push(move)
    won = env.winner is player
    env.pop()
    env.current_player = prev_player
    return won


def find_winning_move(env: CaroEnv, player: Player, radius: int = 2) -> Move | None:
    """Tìm nước đi thắng ngay nếu có (trong các ô ứng viên).

    Mô phỏng ``player`` đặt quân (bất kể lượt hiện tại của env) để phát hiện
    nước thắng tức thì hoặc nước chặn bắt buộc. Dùng push/pop thay vì clone.

    Args:
        env: Môi trường hiện tại.
        player: Người chơi cần tìm nước thắng.

    Returns:
        Nước thắng hoặc None.
    """
    for move in env.candidate_moves(radius=radius):
        if _wins_with_move(env, move, player):
            return move
    return None


def find_blocking_move(env: CaroEnv, player: Player, radius: int = 2) -> Move | None:
    """Tìm nước chặn đối thủ thắng ở lượt kế.

    Args:
        env: Môi trường hiện tại.
        player: Người chơi cần phòng thủ.

    Returns:
        Nước chặn bắt buộc hoặc None.
    """
    opponent = player.opponent
    return find_winning_move(env, opponent, radius=radius)


def _count_stones(
    board: Board, row: int, col: int, dr: int, dc: int, player: Player, size: int
) -> int:
    """Đếm số quân ``player`` liên tiếp theo hướng (dr, dc), không tính ô (row,col)."""
    count = 0
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and board[r, c] == player:
        count += 1
        r += dr
        c += dc
    return count


def _is_four_with_open_end(
    board: Board, row: int, col: int, player: Player, size: int
) -> bool:
    """True nếu quân tại (row,col) nằm trên hàng 4 mở (có thể nối thành 5).

    Args:
        board: Bàn cờ hiện tại.
        row: Hàng quân vừa xét.
        col: Cột quân vừa xét.
        player: Màu quân.
        size: Cạnh bàn cờ.
    """
    if board[row, col] != player:
        return False
    for dr, dc in DIRECTIONS:
        fwd = _count_stones(board, row, col, dr, dc, player, size)
        bwd = _count_stones(board, row, col, -dr, -dc, player, size)
        total = 1 + fwd + bwd
        if total != 4:
            continue
        er, ec = row + dr * (fwd + 1), col + dc * (fwd + 1)
        br, bc = row - dr * (bwd + 1), col - dc * (bwd + 1)
        open_ends = 0
        if 0 <= er < size and 0 <= ec < size and board[er, ec] == Player.EMPTY:
            open_ends += 1
        if 0 <= br < size and 0 <= bc < size and board[br, bc] == Player.EMPTY:
            open_ends += 1
        if open_ends >= 1:
            return True
    return False


def find_open_four_move(env: CaroEnv, player: Player, radius: int = 2) -> Move | None:
    """Tìm nước tạo tứ mở (open four) — ép đối thủ phòng thủ.

    Args:
        env: Môi trường hiện tại.
        player: Người chơi tấn công.

    Returns:
        Nước tạo tứ mở hoặc None.
    """
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
        is_four = _is_four_with_open_end(board, row, col, player, size)
        board[row, col] = Player.EMPTY
        if is_four:
            score = move_priority(board, move, player)
            if score > best_score:
                best_score = score
                best = move
    return best


def find_open_four_block(env: CaroEnv, player: Player, radius: int = 2) -> Move | None:
    """Chặn nước đối thủ tạo tứ mở (threat tấn công mạnh).

    Args:
        env: Môi trường hiện tại.
        player: Người chơi phòng thủ.

    Returns:
        Nước chặn tứ mở hoặc None.
    """
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
        is_four = _is_four_with_open_end(board, row, col, opponent, size)
        board[row, col] = Player.EMPTY
        if is_four:
            score = move_priority(board, move, player)
            if score > best_score:
                best_score = score
                best = move
    return best


def _is_open_three_at(
    board: Board, row: int, col: int, player: Player, size: int
) -> bool:
    """True nếu quân tại (row,col) nằm trên hàng 3 mở (có thể phát triển thành tứ mở)."""
    if board[row, col] != player:
        return False
    for dr, dc in DIRECTIONS:
        fwd = _count_stones(board, row, col, dr, dc, player, size)
        bwd = _count_stones(board, row, col, -dr, -dc, player, size)
        total = 1 + fwd + bwd
        if total != 3:
            continue
        er, ec = row + dr * (fwd + 1), col + dc * (fwd + 1)
        br, bc = row - dr * (bwd + 1), col - dc * (bwd + 1)
        open_ends = 0
        if 0 <= er < size and 0 <= ec < size and board[er, ec] == Player.EMPTY:
            open_ends += 1
        if 0 <= br < size and 0 <= bc < size and board[br, bc] == Player.EMPTY:
            open_ends += 1
        if open_ends >= 1:
            return True
    return False


def find_open_three_block(env: CaroEnv, player: Player, radius: int = 2) -> Move | None:
    """Chặn đối thủ tạo tam mở — giảm tốc độ tấn công trước khi thành tứ mở.

    Args:
        env: Môi trường hiện tại.
        player: Người chơi phòng thủ.
        radius: Bán kính quét nước ứng viên.

    Returns:
        Nước chặn tam mở hoặc None.
    """
    opponent = player.opponent
    best: Move | None = None
    best_score = float("-inf")
    board = env.board
    size = env.size
    for move in env.candidate_moves(radius=radius):
        row, col = move
        if board[row, col] != Player.EMPTY:
            continue
        board[row, col] = opponent
        is_three = _is_open_three_at(board, row, col, opponent, size)
        board[row, col] = Player.EMPTY
        if is_three:
            score = move_priority(board, move, player)
            if score > best_score:
                best_score = score
                best = move
    return best


def find_tactical_move(
    env: CaroEnv,
    player: Player,
    radius: int = 2,
    config: TacticalConfig | None = None,
) -> Move | None:
    """Luật chiến thuật trước Minimax: thắng → chặn thua → phòng tứ/tam mở → tấn công.

    Thứ tự ưu tiên: chặn đe dọa đối thủ (tứ/tam mở) trước khi tự tấn công,
    kể cả khi bật chế độ aggressive — tránh bỏ qua tam mở đối thủ.

    Args:
        env: Môi trường hiện tại.
        player: Người chơi cần chọn nước.
        radius: Bán kính quét nước ứng viên.
        config: Luật chặn 2 đầu / chế độ tấn công.

    Returns:
        Nước chiến thuật ưu tiên cao, hoặc None để tìm kiếm sâu.
    """
    cfg = config or TacticalConfig()

    win = find_winning_move(env, player, radius=radius)
    if win is not None:
        return win
    block = find_blocking_move(env, player, radius=radius)
    if block is not None:
        return block

    defend_four = find_open_four_block(env, player, radius=radius)
    if defend_four is not None:
        return defend_four

    defend_three = find_open_three_block(env, player, radius=radius)
    if defend_three is not None:
        return defend_three

    if cfg.double_end_block_rule:
        from ai.threats import find_open_three_block_double_end

        defend_double = find_open_three_block_double_end(env, player, radius=radius)
        if defend_double is not None:
            return defend_double

    if cfg.aggressive:
        attack_four = find_open_four_move(env, player, radius=radius)
        if attack_four is not None:
            return attack_four
        from ai.threats import find_open_three_attack

        attack_three = find_open_three_attack(env, player, radius=radius)
        if attack_three is not None:
            return attack_three

    return None
