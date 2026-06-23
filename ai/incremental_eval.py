#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluator tăng dần (incremental) cho Cờ Caro.

Ý tưởng (giống các engine Gomoku online): KHÔNG quét lại toàn bộ bàn cờ ở mỗi
node lá. Thay vào đó duy trì điểm số theo TỪNG đường thẳng (hàng/cột/2 đường
chéo). Mỗi ô chỉ nằm trên đúng 4 đường; khi đặt/gỡ một quân ta chỉ tính lại 4
đường đó (O(4·size)) thay vì ~90 đường (O(size²)).

Tổng điểm của evaluator này KHỚP CHÍNH XÁC với ``heuristic.evaluate_board`` vì:
    score_player(board, p) = Σ _score_line(_player_line(line, p, opp))
và evaluator chỉ cache từng số hạng của tổng đó rồi cập nhật cục bộ. Nhờ vậy
kết quả tìm kiếm (nước đi chọn) KHÔNG đổi — chỉ nhanh hơn nhiều.
"""

from __future__ import annotations

from config import Player
from core.constants import Board, Move
from ai.heuristic import _player_line, _score_line


class IncrementalEvaluator:
    """Giữ điểm heuristic của bàn cờ và cập nhật tăng dần theo từng nước.

    Cách dùng trong tìm kiếm:
        ev = IncrementalEvaluator(env.board)   # board là cùng một mảng numpy
        env.push(move); ev.touch(move)          # sau khi đặt quân
        ...                                     # ev.value(p) luôn đúng
        env.pop();      ev.touch(move)          # sau khi gỡ quân

    ``touch`` đọc trạng thái HIỆN TẠI của board nên đặt hay gỡ đều xử lý chung —
    không cần lưu stack riêng.
    """

    def __init__(self, board: Board) -> None:
        """Gắn evaluator vào một mảng bàn cờ và tính điểm ban đầu.

        Args:
            board: Mảng numpy int8 (size×size). Evaluator GIỮ THAM CHIẾU tới
                mảng này — mọi thay đổi tại chỗ (push/pop) sẽ được phản ánh khi
                gọi ``touch``.
        """
        self.board = board
        self.size = int(board.shape[0])
        self._lines: list[list[Move]] = []          # mỗi line = danh sách ô (r,c)
        self._line_bordered: list[bool] = []          # có bọc biên '3' hay không
        self._cell_lines: dict[Move, list[int]] = {}  # ô -> các line đi qua
        self._build_lines()
        self._sx = [0] * len(self._lines)             # điểm theo góc nhìn X
        self._so = [0] * len(self._lines)             # điểm theo góc nhìn O
        self.total_x = 0
        self.total_o = 0
        self.recompute_all()

    # ------------------------------------------------------------------
    #  XÂY DỰNG DANH SÁCH ĐƯỜNG (khớp heuristic._extract_lines)
    # ------------------------------------------------------------------
    def _add_line(self, cells: list[Move], bordered: bool) -> None:
        idx = len(self._lines)
        self._lines.append(cells)
        self._line_bordered.append(bordered)
        for cell in cells:
            self._cell_lines.setdefault(cell, []).append(idx)

    def _build_lines(self) -> None:
        size = self.size
        # Hàng ngang & cột dọc (không bọc biên — giống _extract_lines).
        for row in range(size):
            self._add_line([(row, col) for col in range(size)], bordered=False)
        for col in range(size):
            self._add_line([(row, col) for row in range(size)], bordered=False)

        # Chéo xuống-phải (bọc biên '3').
        for start_col in range(size):
            cells, r, c = [], 0, start_col
            while r < size and c < size:
                cells.append((r, c)); r += 1; c += 1
            self._add_line(cells, bordered=True)
        for start_row in range(1, size):
            cells, r, c = [], start_row, 0
            while r < size and c < size:
                cells.append((r, c)); r += 1; c += 1
            self._add_line(cells, bordered=True)

        # Chéo lên-phải (bọc biên '3').
        for start_col in range(size):
            cells, r, c = [], size - 1, start_col
            while r >= 0 and c < size:
                cells.append((r, c)); r -= 1; c += 1
            self._add_line(cells, bordered=True)
        for start_row in range(size - 2, -1, -1):
            cells, r, c = [], start_row, 0
            while r >= 0 and c < size:
                cells.append((r, c)); r -= 1; c += 1
            self._add_line(cells, bordered=True)

    # ------------------------------------------------------------------
    #  TÍNH ĐIỂM
    # ------------------------------------------------------------------
    def _line_string(self, idx: int) -> str:
        """Chuỗi digit của một đường (kèm biên '3' nếu là đường chéo)."""
        board = self.board
        core = "".join(str(int(board[r, c])) for r, c in self._lines[idx])
        if self._line_bordered[idx]:
            return "3" + core + "3"
        return core

    def _score_line_idx(self, idx: int) -> tuple[int, int]:
        """Điểm (X-view, O-view) cho một đường."""
        raw = self._line_string(idx)
        sx = _score_line(_player_line(raw, Player.X, Player.O))
        so = _score_line(_player_line(raw, Player.O, Player.X))
        return sx, so

    def recompute_all(self) -> None:
        """Tính lại điểm mọi đường (gọi một lần khi khởi tạo)."""
        self.total_x = 0
        self.total_o = 0
        for idx in range(len(self._lines)):
            sx, so = self._score_line_idx(idx)
            self._sx[idx] = sx
            self._so[idx] = so
            self.total_x += sx
            self.total_o += so

    def touch(self, move: Move) -> None:
        """Cập nhật điểm sau khi ô ``move`` vừa thay đổi trên board.

        Gọi sau MỖI push (đặt quân) và MỖI pop (gỡ quân) với cùng ``move``.
        """
        for idx in self._cell_lines.get(move, ()):  # type: ignore[arg-type]
            old_x, old_o = self._sx[idx], self._so[idx]
            new_x, new_o = self._score_line_idx(idx)
            if new_x != old_x:
                self.total_x += new_x - old_x
                self._sx[idx] = new_x
            if new_o != old_o:
                self.total_o += new_o - old_o
                self._so[idx] = new_o

    def value(self, for_player: Player) -> float:
        """Điểm heuristic theo góc nhìn ``for_player`` (ta − đối thủ).

        Bằng đúng ``evaluate_board(board, for_player)``.
        """
        if for_player is Player.X:
            return float(self.total_x - self.total_o)
        return float(self.total_o - self.total_x)
