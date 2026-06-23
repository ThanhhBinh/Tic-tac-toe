#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tác nhân Minimax với Alpha-Beta Pruning cho Cờ Caro.

Luồng ra quyết định:
    1. Thắng ngay nếu có thể.
    2. Chặn đối thủ thắng ngay nếu buộc phải chặn.
    3. Tìm kiếm Minimax + Alpha-Beta trên không gian nước đi ứng viên, sắp xếp
       nước đi theo heuristic để cắt tỉa hiệu quả hơn.

Tối ưu hiệu năng:
    - ``push``/``pop`` thay ``clone()`` khi duyệt cây tìm kiếm.
    - Transposition table (TT) cache điểm số theo trạng thái bàn + depth.
    - ``max_branch`` giới hạn số nhánh mở rộng mỗi node (Hybrid dùng).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from config import Difficulty, Player, TacticalConfig
from core.caro_env import CaroEnv
from core.constants import Move

from ai.base_agent import Agent
from ai.heuristic import (
    evaluate_position,
    find_tactical_move,
    move_priority,
)


# Loại cận của giá trị lưu trong transposition table.
_TT_EXACT = 0   # Giá trị chính xác (node duyệt hết, nằm trong cửa sổ).
_TT_LOWER = 1   # Cận dưới (node bị cắt beta — giá trị thật ≥ score).
_TT_UPPER = 2   # Cận trên (node fail-low — giá trị thật ≤ score).

# Điểm coi như "thắng/thua chắc chắn" — gặp giá trị này có thể dừng sớm.
_WIN_THRESHOLD = 10.0**8


class _SearchTimeout(Exception):
    """Báo hiệu vượt ngân sách thời gian — huỷ vòng lặp deepening hiện tại."""


@dataclass
class _TTEntry:
    """Một mục trong bảng transposition (điểm số + loại cận đã tính trước)."""

    depth: int
    score: float
    best_move: Move | None
    flag: int = _TT_EXACT


class MinimaxAgent(Agent):
    """Agent chơi cờ bằng Minimax có Alpha-Beta và heuristic pattern-based."""

    name = "Minimax"

    def __init__(
        self,
        depth: int = Difficulty.MEDIUM,
        candidate_radius: int = 2,
        max_branch: int | None = None,
        tactical_config: TacticalConfig | None = None,
        time_budget: float | None = None,
    ) -> None:
        """Khởi tạo agent với độ sâu tìm kiếm.

        Args:
            depth: Độ sâu Minimax (số ply). Thường map từ ``Difficulty``.
            candidate_radius: Bán kính sinh nước ứng viên quanh quân đã đặt.
            max_branch: Giới hạn số nhánh mở rộng mỗi node; None = không giới hạn.
            tactical_config: Luật chặn 2 đầu và độ tấn công.
            time_budget: Ngân sách thời gian (giây) cho mỗi nước khi dùng
                iterative deepening; None = tìm hết ``depth`` (tất định).
        """
        self.depth = max(1, int(depth))
        self.candidate_radius = candidate_radius
        self.max_branch = max_branch
        self.tactical_config = tactical_config or TacticalConfig()
        self.time_budget = time_budget
        self._tt: dict[tuple[bytes, int, int, int], _TTEntry] = {}
        # Killer moves theo độ sâu còn lại — nước gây cắt tỉa beta, thử sớm lần sau.
        self._killers: dict[int, list[Move]] = {}
        self.name = f"Minimax (depth={self.depth})"

    def get_move(self, env: CaroEnv) -> Move:
        """Chọn nước đi tốt nhất cho ``env.current_player``.

        Dùng iterative deepening: tìm lần lượt depth = 1, 2, …, self.depth, tận
        dụng transposition table + killer move để sắp xếp nước tốt lên trước
        (cắt tỉa sâu hơn). Nếu có ``time_budget`` và vượt quá, trả về nước tốt
        nhất của lần lặp sâu nhất đã hoàn tất.

        Args:
            env: Môi trường hiện tại (không bị thay đổi).

        Returns:
            Nước hợp lệ.
        """
        self._tt.clear()
        self._killers.clear()
        player = env.current_player

        if not getattr(self, "search_only", False):
            tactical = find_tactical_move(
                env,
                player,
                radius=self.candidate_radius,
                config=self.tactical_config,
            )
            if tactical is not None:
                return tactical

        fallback = env.candidate_moves(radius=self.candidate_radius) or env.legal_moves()
        if not fallback:
            return env.legal_moves()[0]

        deadline = (
            time.perf_counter() + self.time_budget
            if self.time_budget is not None
            else None
        )

        best_move: Move = fallback[0]
        for current_depth in range(1, self.depth + 1):
            try:
                value, move = self._alpha_beta(
                    env=env,
                    depth=current_depth,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    ai_player=player,
                    deadline=deadline,
                )
            except _SearchTimeout:
                break  # Giữ best_move của lần lặp trước (đã hoàn tất).

            if move is not None:
                best_move = move
            # Tìm thấy thắng/thua ép buộc thì không cần đào sâu thêm.
            if abs(value) >= _WIN_THRESHOLD:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

        return best_move

    def _ordered_moves(self, env: CaroEnv, player: Player) -> list[Move]:
        """Sinh nước ứng viên và sắp xếp giảm dần theo điểm heuristic.

        Move ordering giúp Alpha-Beta cắt tỉa sớm hơn, đặc biệt ở depth cao.
        """
        moves = env.candidate_moves(radius=self.candidate_radius)
        if not moves:
            moves = env.legal_moves()
        moves.sort(
            key=lambda m: move_priority(env.board, m, player),
            reverse=True,
        )
        if self.max_branch is not None:
            moves = moves[: self.max_branch]
        return moves

    def _reorder_for_search(
        self, moves: list[Move], tt_move: Move | None, depth: int
    ) -> list[Move]:
        """Đưa PV-move (TT) rồi killer move lên đầu danh sách (chỉ đổi thứ tự).

        Sắp xếp lại không thay đổi kết quả alpha-beta, chỉ giúp cắt tỉa sớm hơn.
        """
        if not moves:
            return moves
        priority: list[Move] = []
        if tt_move is not None and tt_move in moves:
            priority.append(tt_move)
        for killer in self._killers.get(depth, ()):  # type: ignore[arg-type]
            if killer in moves and killer not in priority:
                priority.append(killer)
        if not priority:
            return moves
        priority_set = set(priority)
        return priority + [m for m in moves if m not in priority_set]

    def _record_killer(self, move: Move, depth: int) -> None:
        """Ghi nhớ nước gây cắt tỉa ở độ sâu này (tối đa 2 killer/độ sâu)."""
        killers = self._killers.setdefault(depth, [])
        if move in killers:
            return
        killers.insert(0, move)
        if len(killers) > 2:
            killers.pop()

    def _tt_key(
        self, env: CaroEnv, depth: int, ai_player: Player
    ) -> tuple[bytes, int, int, int]:
        """Khóa TT: bàn cờ + depth còn lại + người chơi AI + lượt hiện tại."""
        return (
            env.board.tobytes(),
            depth,
            int(ai_player),
            int(env.current_player),
        )

    def _alpha_beta(
        self,
        env: CaroEnv,
        depth: int,
        alpha: float,
        beta: float,
        ai_player: Player,
        deadline: float | None = None,
    ) -> tuple[float, Move | None]:
        """Minimax với Alpha-Beta pruning, push/pop và transposition table.

        Args:
            env: Trạng thái hiện tại (mutated tạm thời bằng push/pop).
            depth: Số ply còn lại.
            alpha: Ngưỡng alpha.
            beta: Ngưỡng beta.
            ai_player: Người chơi mà ta muốn tối đa hoá điểm.
            deadline: Mốc ``time.perf_counter()`` phải dừng; None = không giới hạn.

        Returns:
            Cặp (điểm số, nước đi tốt nhất ở node gốc).

        Raises:
            _SearchTimeout: Nếu vượt ``deadline`` (vòng deepening sẽ bỏ kết quả dở).
        """
        if deadline is not None and time.perf_counter() >= deadline:
            raise _SearchTimeout

        if env.done or depth == 0:
            score = self._evaluate_leaf(env, ai_player)
            return score, None

        alpha_orig = alpha
        beta_orig = beta

        tt_key = self._tt_key(env, depth, ai_player)
        cached = self._tt.get(tt_key)
        tt_move: Move | None = None
        if cached is not None:
            tt_move = cached.best_move
            if cached.depth >= depth:
                # Dùng loại cận để hoặc trả ngay, hoặc thu hẹp cửa sổ [alpha,beta]
                # → tạo thêm nhiều lần cắt tỉa mà vẫn đúng (an toàn về cận).
                if cached.flag == _TT_EXACT:
                    return cached.score, cached.best_move
                if cached.flag == _TT_LOWER:
                    alpha = max(alpha, cached.score)
                elif cached.flag == _TT_UPPER:
                    beta = min(beta, cached.score)
                if alpha >= beta:
                    return cached.score, cached.best_move

        current = env.current_player
        maximizing = current is ai_player
        moves = self._ordered_moves(env, current)
        # PV-move (TT) + killer move lên trước → cắt tỉa sớm hơn (chỉ đổi thứ tự,
        # không đổi kết quả alpha-beta).
        moves = self._reorder_for_search(moves, tt_move, depth)

        best_move: Move | None = None

        if maximizing:
            value = float("-inf")
            for move in moves:
                env.push(move)
                try:
                    if env.done:
                        child_score = self._evaluate_leaf(env, ai_player)
                    else:
                        child_score, _ = self._alpha_beta(
                            env, depth - 1, alpha, beta, ai_player, deadline
                        )
                finally:
                    env.pop()

                if child_score > value:
                    value = child_score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self._record_killer(move, depth)
                    break  # Beta cut
        else:
            value = float("inf")
            for move in moves:
                env.push(move)
                try:
                    if env.done:
                        child_score = self._evaluate_leaf(env, ai_player)
                    else:
                        child_score, _ = self._alpha_beta(
                            env, depth - 1, alpha, beta, ai_player, deadline
                        )
                finally:
                    env.pop()

                if child_score < value:
                    value = child_score
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    self._record_killer(move, depth)
                    break  # Alpha cut

        # Xác định loại cận để lưu (đúng cho cả node max lẫn min nhờ cửa sổ
        # [alpha,beta] đối xứng): fail-low → UPPER, fail-high → LOWER, else EXACT.
        if value <= alpha_orig:
            flag = _TT_UPPER
        elif value >= beta_orig:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt[tt_key] = _TTEntry(depth, value, best_move, flag)
        return value, best_move

    def _evaluate_leaf(self, env: CaroEnv, ai_player: Player) -> float:
        """Lượng giá node lá (depth=0 hoặc kết thúc).

        Mặc định dùng heuristic pattern-based. ``HybridAgent`` ghi đè để dùng
        Q-value từ DQN tại node lá.

        Args:
            env: Trạng thái bàn cờ hiện tại.
            ai_player: Người chơi mà ta muốn tối đa hoá điểm.

        Returns:
            Điểm số float (cao = có lợi cho ``ai_player``).
        """
        return evaluate_position(env.winner, env.board, ai_player)

    def get_win_probability(
        self, env: CaroEnv, for_player: Player | None = None
    ) -> float:
        """Ước lượng xác suất thắng từ heuristic pattern-based.

        Args:
            env: Môi trường hiện tại.
            for_player: Góc nhìn; None = ``env.current_player``.

        Returns:
            Xác suất trong [0, 1].
        """
        from ai.win_probability import estimate_win_probability

        player = for_player if for_player is not None else env.current_player
        return estimate_win_probability(env, player)

    @classmethod
    def from_difficulty(
        cls,
        difficulty: Difficulty,
        tactical_config: TacticalConfig | None = None,
        time_budget: float | None = -1.0,
    ) -> "MinimaxAgent":
        """Tạo agent với độ sâu lấy trực tiếp từ enum ``Difficulty``.

        Args:
            difficulty: Mức độ khó (EASY=1 .. EXPERT=4 ply).
            tactical_config: Luật chiến thuật tùy chọn.
            time_budget: Ngân sách thời gian/nước. ``-1.0`` (mặc định) = dùng
                ``MINIMAX_PLAY_TIME_BUDGET_SEC`` để chơi tương tác an toàn;
                truyền ``None`` để tìm hết độ sâu (tất định, dùng khi test/đo).

        Returns:
            MinimaxAgent đã cấu hình.
        """
        from config import MINIMAX_PLAY_TIME_BUDGET_SEC

        budget = (
            MINIMAX_PLAY_TIME_BUDGET_SEC if time_budget == -1.0 else time_budget
        )
        return cls(
            depth=int(difficulty),
            tactical_config=tactical_config,
            time_budget=budget,
        )
