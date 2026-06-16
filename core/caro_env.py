#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Môi trường game Cờ Caro (Gomoku) — class `CaroEnv`.

Đây là "backend" thuần logic, không phụ thuộc giao diện. UI và các agent AI
chỉ ĐỌC trạng thái từ class này để vẽ/ra quyết định.

Tối ưu hiệu năng (theo .cursorrules mục 7):
    - Bàn cờ lưu bằng ``numpy.ndarray`` dtype ``int8``.
    - Kiểm tra thắng chỉ quét quanh nước vừa đánh theo 4 hướng, KHÔNG quét lại
      toàn bộ bàn cờ mỗi lượt -> O(WIN_LENGTH) thay vì O(n^2).
"""

from __future__ import annotations

import numpy as np

from config import WIN_LENGTH, Player
from core.constants import DIRECTIONS, Board, Move


class CaroEnv:
    """Quản lý trạng thái và luật chơi của một ván Cờ Caro.

    Attributes:
        size: Kích thước bàn cờ (size x size).
        win_length: Số quân liên tiếp cần để thắng.
        board: Ma trận trạng thái bàn cờ (int8).
        current_player: Người chơi sắp đi ở lượt hiện tại.
        last_move: Nước đi gần nhất, None nếu chưa có.
        winner: Người thắng (Player.X/O) hoặc None nếu chưa kết thúc/hòa.
        done: True nếu ván đã kết thúc (có người thắng hoặc hòa).
        winning_line: Danh sách ô tạo thành đường thắng (để UI highlight).
    """

    def __init__(
        self,
        size: int = 15,
        win_length: int = WIN_LENGTH,
        double_end_block_rule: bool = False,
    ) -> None:
        """Khởi tạo môi trường với kích thước bàn cờ cho trước.

        Args:
            size: Cạnh bàn cờ (vd: 10 hoặc 15).
            win_length: Số quân liên tiếp để thắng (mặc định 5).
            double_end_block_rule: Nếu True, đúng 5 quân liên tiếp bị chặn
                hai đầu bởi đối phương (hoặc biên bàn) không được tính thắng.

        Raises:
            ValueError: Nếu size nhỏ hơn win_length.
        """
        if size < win_length:
            raise ValueError(
                f"Kích thước bàn cờ ({size}) phải >= số quân thắng ({win_length})."
            )
        self.size: int = size
        self.win_length: int = win_length
        self.double_end_block_rule: bool = double_end_block_rule
        self.board: Board = np.zeros((size, size), dtype=np.int8)
        self.current_player: Player = Player.X
        self.last_move: Move | None = None
        self.winner: Player | None = None
        self.done: bool = False
        self.winning_line: list[Move] = []
        self._move_count: int = 0
        # Stack hoàn tác cho Minimax (push/pop) — tránh clone bàn cờ mỗi nhánh.
        self._undo_stack: list[dict[str, object]] = []

    # ----------------------------------------------------------------------
    #  VÒNG ĐỜI VÁN ĐẤU
    # ----------------------------------------------------------------------
    def reset(self) -> Board:
        """Đặt lại bàn cờ về trạng thái ban đầu (X đi trước).

        Returns:
            Bản sao trạng thái bàn cờ sau khi reset.
        """
        self.board.fill(Player.EMPTY)
        self.current_player = Player.X
        self.last_move = None
        self.winner = None
        self.done = False
        self.winning_line = []
        self._move_count = 0
        self._undo_stack.clear()
        return self.board.copy()

    # ----------------------------------------------------------------------
    #  KIỂM TRA & SINH NƯỚC ĐI
    # ----------------------------------------------------------------------
    def in_bounds(self, row: int, col: int) -> bool:
        """Kiểm tra (row, col) có nằm trong bàn cờ không."""
        return 0 <= row < self.size and 0 <= col < self.size

    def is_legal(self, move: Move) -> bool:
        """Kiểm tra một nước đi có hợp lệ không.

        Hợp lệ khi: ván chưa kết thúc, ô nằm trong bàn cờ và đang trống.

        Args:
            move: Nước đi (hàng, cột).

        Returns:
            True nếu nước đi hợp lệ.
        """
        row, col = move
        return (
            not self.done
            and self.in_bounds(row, col)
            and self.board[row, col] == Player.EMPTY
        )

    def legal_moves(self) -> list[Move]:
        """Trả về danh sách tất cả các ô trống (nước đi hợp lệ)."""
        if self.done:
            return []
        rows, cols = np.where(self.board == Player.EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))

    def candidate_moves(self, radius: int = 1) -> list[Move]:
        """Sinh các nước đi "ứng viên" gần các quân đã đặt (để AI tìm kiếm).

        Giới hạn không gian tìm kiếm của Minimax: chỉ xét các ô trống nằm trong
        bán kính ``radius`` quanh ít nhất một quân đã có. Khi bàn cờ trống,
        trả về ô trung tâm.

        Args:
            radius: Bán kính lân cận tính theo ô.

        Returns:
            Danh sách nước đi ứng viên (đã loại trùng).
        """
        if self._move_count == 0:
            center = self.size // 2
            return [(center, center)]

        occupied = np.argwhere(self.board != Player.EMPTY)
        candidates: set[Move] = set()
        for r, c in occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = int(r) + dr, int(c) + dc
                    if self.in_bounds(nr, nc) and self.board[nr, nc] == Player.EMPTY:
                        candidates.add((nr, nc))
        return list(candidates)

    # ----------------------------------------------------------------------
    #  THỰC HIỆN NƯỚC ĐI
    # ----------------------------------------------------------------------
    def step(self, move: Move) -> tuple[Board, Player | None, bool]:
        """Đặt một quân của người chơi hiện tại lên bàn cờ.

        Sau khi đặt: kiểm tra thắng quanh nước vừa đánh, cập nhật trạng thái và
        đổi lượt nếu ván chưa kết thúc.

        Args:
            move: Nước đi (hàng, cột).

        Returns:
            Bộ ba (bàn cờ mới, người thắng hoặc None, ván đã kết thúc?).

        Raises:
            ValueError: Nếu nước đi không hợp lệ.
        """
        if not self.is_legal(move):
            raise ValueError(f"Nước đi không hợp lệ: {move}.")

        row, col = move
        player = self.current_player
        self.board[row, col] = player
        self.last_move = move
        self._move_count += 1

        # Kiểm tra thắng chỉ quanh nước vừa đánh (tối ưu).
        if self._check_win_from(row, col, player):
            self.winner = player
            self.done = True
        elif self._move_count == self.size * self.size:
            # Hết ô trống mà chưa ai thắng -> hòa.
            self.done = True
        else:
            self.current_player = player.opponent

        return self.board.copy(), self.winner, self.done

    def push(self, move: Move) -> None:
        """Đặt quân tạm thời cho tìm kiếm Minimax (dùng kèm ``pop``).

        Ghi lại trạng thái trước khi đặt để hoàn tác nhanh, tránh ``clone()``.

        Args:
            move: Nước đi hợp lệ trên bàn hiện tại.

        Raises:
            ValueError: Nếu nước đi không hợp lệ.
        """
        if not self.is_legal(move):
            raise ValueError(f"Nước đi push không hợp lệ: {move}.")

        self._undo_stack.append(
            {
                "move": move,
                "player": self.current_player,
                "last_move": self.last_move,
                "winner": self.winner,
                "done": self.done,
                "winning_line": list(self.winning_line),
                "move_count": self._move_count,
            }
        )

        row, col = move
        player = self.current_player
        self.board[row, col] = player
        self.last_move = move
        self._move_count += 1

        if self._check_win_from(row, col, player):
            self.winner = player
            self.done = True
        elif self._move_count == self.size * self.size:
            self.done = True
        else:
            self.current_player = player.opponent

    def pop(self) -> None:
        """Hoàn tác nước đi gần nhất do ``push`` (LIFO).

        Raises:
            RuntimeError: Nếu stack undo rỗng.
        """
        if not self._undo_stack:
            raise RuntimeError("pop() khi undo stack rỗng.")

        record = self._undo_stack.pop()
        move = record["move"]
        assert isinstance(move, tuple)
        row, col = move
        self.board[row, col] = Player.EMPTY
        self.current_player = record["player"]  # type: ignore[assignment]
        self.last_move = record["last_move"]  # type: ignore[assignment]
        self.winner = record["winner"]  # type: ignore[assignment]
        self.done = bool(record["done"])
        self.winning_line = list(record["winning_line"])  # type: ignore[arg-type]
        self._move_count = int(record["move_count"])  # type: ignore[arg-type]

    def _is_cell_blocked_for_player(self, row: int, col: int, player: Player) -> bool:
        """True nếu ô ngoài biên hoặc bị quân đối phương chiếm (không mở)."""
        if not self.in_bounds(row, col):
            return True
        return self.board[row, col] == player.opponent

    def _check_win_from(self, row: int, col: int, player: Player) -> bool:
        """Kiểm tra nước vừa đặt tại (row, col) có tạo thành đường thắng không.

        Với mỗi hướng trong 4 hướng, đếm số quân liên tiếp cùng màu về cả hai
        phía. Nếu tổng >= win_length thì thắng và lưu lại ``winning_line``.

        Khi bật ``double_end_block_rule``: đúng ``win_length`` quân liên tiếp
        chỉ thắng nếu ít nhất một đầu còn mở (ô trống). Hai đầu đều bị đối
        phương hoặc biên bàn chặn thì không tính thắng.

        Args:
            row: Hàng của quân vừa đặt.
            col: Cột của quân vừa đặt.
            player: Người chơi vừa đặt quân.

        Returns:
            True nếu tạo thành đường thắng hợp lệ.
        """
        for dr, dc in DIRECTIONS:
            line: list[Move] = [(row, col)]

            # Đếm về phía thuận (dr, dc).
            r, c = row + dr, col + dc
            while self.in_bounds(r, c) and self.board[r, c] == player:
                line.append((r, c))
                r += dr
                c += dc

            # Đếm về phía ngược (-dr, -dc).
            r, c = row - dr, col - dc
            while self.in_bounds(r, c) and self.board[r, c] == player:
                line.insert(0, (r, c))
                r -= dr
                c -= dc

            if len(line) < self.win_length:
                continue

            if (
                self.double_end_block_rule
                and len(line) == self.win_length
            ):
                first_r, first_c = line[0]
                last_r, last_c = line[-1]
                before_blocked = self._is_cell_blocked_for_player(
                    first_r - dr, first_c - dc, player
                )
                after_blocked = self._is_cell_blocked_for_player(
                    last_r + dr, last_c + dc, player
                )
                if before_blocked and after_blocked:
                    continue

            # Cắt đúng win_length ô liên tiếp chứa nước vừa đánh để highlight.
            self.winning_line = (
                line[: self.win_length] if len(line) > self.win_length else line
            )
            return True
        return False

    # ----------------------------------------------------------------------
    #  TIỆN ÍCH CHO AI
    # ----------------------------------------------------------------------
    def clone(self) -> "CaroEnv":
        """Tạo bản sao sâu của môi trường (cho tìm kiếm Minimax/giả lập).

        Returns:
            Một CaroEnv mới độc lập với trạng thái giống hệt hiện tại.
        """
        cloned = CaroEnv(
            self.size,
            self.win_length,
            double_end_block_rule=self.double_end_block_rule,
        )
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        cloned.last_move = self.last_move
        cloned.winner = self.winner
        cloned.done = self.done
        cloned.winning_line = list(self.winning_line)
        cloned._move_count = self._move_count
        return cloned

    def copy_state_from(self, other: "CaroEnv") -> None:
        """Ghi đè trạng thái hiện tại từ bản sao (phục vụ undo/redo trên UI).

        Args:
            other: Môi trường nguồn (cùng ``size`` và ``win_length``).
        """
        if self.size != other.size or self.win_length != other.win_length:
            raise ValueError("Không thể copy_state_from môi trường khác kích thước.")
        self.board = other.board.copy()
        self.current_player = other.current_player
        self.last_move = other.last_move
        self.winner = other.winner
        self.done = other.done
        self.winning_line = list(other.winning_line)
        self._move_count = other._move_count
        self._undo_stack.clear()

    @property
    def move_count(self) -> int:
        """Số quân đã được đặt trên bàn cờ."""
        return self._move_count

    @property
    def is_draw(self) -> bool:
        """True nếu ván kết thúc mà không có người thắng (hòa)."""
        return self.done and self.winner is None

    def __repr__(self) -> str:
        """Biểu diễn ngắn gọn để debug."""
        return (
            f"CaroEnv(size={self.size}, moves={self._move_count}, "
            f"turn={self.current_player.name}, done={self.done})"
        )
