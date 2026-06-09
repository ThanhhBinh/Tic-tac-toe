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


@dataclass
class _TTEntry:
    """Một mục trong bảng transposition (điểm số đã tính trước)."""

    depth: int
    score: float
    best_move: Move | None


class MinimaxAgent(Agent):
    """Agent chơi cờ bằng Minimax có Alpha-Beta và heuristic pattern-based."""

    name = "Minimax"

    def __init__(
        self,
        depth: int = Difficulty.MEDIUM,
        candidate_radius: int = 2,
        max_branch: int | None = None,
        tactical_config: TacticalConfig | None = None,
    ) -> None:
        """Khởi tạo agent với độ sâu tìm kiếm.

        Args:
            depth: Độ sâu Minimax (số ply). Thường map từ ``Difficulty``.
            candidate_radius: Bán kính sinh nước ứng viên quanh quân đã đặt.
            max_branch: Giới hạn số nhánh mở rộng mỗi node; None = không giới hạn.
            tactical_config: Luật chặn 2 đầu và độ tấn công.
        """
        self.depth = max(1, int(depth))
        self.candidate_radius = candidate_radius
        self.max_branch = max_branch
        self.tactical_config = tactical_config or TacticalConfig()
        self._tt: dict[tuple[bytes, int, int, int], _TTEntry] = {}
        self.name = f"Minimax (depth={self.depth})"

    def get_move(self, env: CaroEnv) -> Move:
        """Chọn nước đi tốt nhất cho ``env.current_player``.

        Args:
            env: Môi trường hiện tại (không bị thay đổi).

        Returns:
            Nước đi hợp lệ.
        """
        self._tt.clear()
        player = env.current_player

        tactical = find_tactical_move(
            env,
            player,
            radius=self.candidate_radius,
            config=self.tactical_config,
        )
        if tactical is not None:
            return tactical

        moves = self._ordered_moves(env, player)
        if not moves:
            return env.legal_moves()[0]

        _, best_move = self._alpha_beta(
            env=env,
            depth=self.depth,
            alpha=float("-inf"),
            beta=float("inf"),
            ai_player=player,
        )
        return best_move if best_move is not None else moves[0]

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
    ) -> tuple[float, Move | None]:
        """Minimax với Alpha-Beta pruning, push/pop và transposition table.

        Args:
            env: Trạng thái hiện tại (mutated tạm thời bằng push/pop).
            depth: Số ply còn lại.
            alpha: Ngưỡng alpha.
            beta: Ngưỡng beta.
            ai_player: Người chơi mà ta muốn tối đa hoá điểm.

        Returns:
            Cặp (điểm số, nước đi tốt nhất ở node gốc).
        """
        if env.done or depth == 0:
            score = self._evaluate_leaf(env, ai_player)
            return score, None

        tt_key = self._tt_key(env, depth, ai_player)
        cached = self._tt.get(tt_key)
        if cached is not None and cached.depth >= depth:
            return cached.score, cached.best_move

        current = env.current_player
        maximizing = current is ai_player
        moves = self._ordered_moves(env, current)

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
                            env, depth - 1, alpha, beta, ai_player
                        )
                finally:
                    env.pop()

                if child_score > value:
                    value = child_score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cut
            self._tt[tt_key] = _TTEntry(depth, value, best_move)
            return value, best_move

        value = float("inf")
        for move in moves:
            env.push(move)
            try:
                if env.done:
                    child_score = self._evaluate_leaf(env, ai_player)
                else:
                    child_score, _ = self._alpha_beta(
                        env, depth - 1, alpha, beta, ai_player
                    )
            finally:
                env.pop()

            if child_score < value:
                value = child_score
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cut
        self._tt[tt_key] = _TTEntry(depth, value, best_move)
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
    ) -> "MinimaxAgent":
        """Tạo agent với độ sâu lấy trực tiếp từ enum ``Difficulty``.

        Args:
            difficulty: Mức độ khó (EASY=1 .. EXPERT=4 ply).
            tactical_config: Luật chiến thuật tùy chọn.

        Returns:
            MinimaxAgent đã cấu hình.
        """
        return cls(depth=int(difficulty), tactical_config=tactical_config)
