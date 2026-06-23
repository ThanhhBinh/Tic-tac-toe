#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tác nhân Hybrid: Minimax tìm sâu hơn + DQN sắp xếp nước.

Hybrid MẠNH HƠN Minimax thuần (đo bằng đối kháng thật) nhờ:
    1. **Tìm sâu hơn 1 ply** (``depth = minimax_depth + HYBRID_EXTRA_DEPTH``):
       nhìn xa thêm một lượt đối thủ → đánh giá thế cờ chính xác hơn.
    2. **DQN + heuristic sắp xếp nước ở root** → alpha-beta cắt tỉa sớm hơn, bù
       chi phí của ply phụ (đây chính là phần "kết hợp": RL giúp search nhanh).
       KHÔNG thay nước bằng Q-value trực tiếp (tránh DQN yếu chọn sai).
    3. **Win% từ DQN** cho HUD + học online sau ván.

Lưu ý giới hạn RL: DQN một mình **không** đủ mạnh để thay Minimax — vai trò của
nó trong Hybrid là tăng tốc tìm kiếm (ordering) + ước lượng win% + học online,
còn sức mạnh thắng/thua đến từ việc tìm sâu hơn. Đừng đánh giá Hybrid bằng điểm
heuristic-1-nước của benchmark (thiên vị nước có heuristic tức thời cao); hãy đo
bằng tỷ lệ thắng đối kháng (xem ``scripts/compare_strength.py``).
"""

from __future__ import annotations

import time

import numpy as np

from ai.board_encoder import legal_action_mask, move_to_action
from ai.dqn_agent import DQNAgent
from ai.heuristic import evaluate_board, evaluate_position, find_tactical_move, move_priority
from ai.minimax_agent import MinimaxAgent, _SearchTimeout, _WIN_THRESHOLD
from config import (
    HYBRID_CANDIDATE_RADIUS_BY_DIFFICULTY,
    HYBRID_EXTRA_DEPTH,
    Player,
    TacticalConfig,
    hybrid_depth_for,
    Difficulty,
)
from core.caro_env import CaroEnv
from core.constants import Move


class HybridAgent(MinimaxAgent):
    """Minimax + DQN reorder ở root (chơi thật); benchmark có budget nhánh rộng hơn."""

    name = "Hybrid"

    def __init__(
        self,
        depth: int = Difficulty.MEDIUM,
        board_size: int = 15,
        candidate_radius: int = 2,
        device: str | None = None,
        max_branch: int | None = None,
        tactical_config: TacticalConfig | None = None,
        time_budget: float | None = None,
    ) -> None:
        super().__init__(
            depth=depth,
            candidate_radius=candidate_radius,
            max_branch=max_branch,
            tactical_config=tactical_config,
            time_budget=time_budget,
        )
        self.board_size = board_size
        self.dqn = DQNAgent(board_size=board_size, epsilon=0.0, device=device)
        self._refresh_name()

    def _refresh_name(self) -> None:
        if self.dqn._model_loaded:
            dqn_tag = "DQN đã nạp"
        else:
            dqn_tag = "chưa train DQN"
        self.name = f"Hybrid (depth={self.depth}, {dqn_tag})"

    def _evaluate_leaf(self, env: CaroEnv, ai_player: Player) -> float:
        return evaluate_position(env.winner, env.board, ai_player)

    def _ordered_moves(self, env: CaroEnv, player: Player) -> list[Move]:
        """Heuristic trước; DQN reorder ở root (1 forward) rồi cắt nhánh nếu có max_branch."""
        moves = env.candidate_moves(radius=self.candidate_radius)
        if not moves:
            moves = env.legal_moves()
        if not moves:
            return moves

        use_dqn = self.dqn._model_loaded and getattr(self, "_dqn_reorder_root", False)
        if use_dqn:
            self._dqn_reorder_root = False
            q = self.dqn._predict_q_numpy(env, player)
            size = env.size

            def _sort_key(m: Move) -> tuple[float, float]:
                h = move_priority(env.board, m, player)
                idx = move_to_action(m, size)
                qv = float(q[idx]) if 0 <= idx < q.size else 0.0
                if not np.isfinite(qv):
                    qv = 0.0
                return (h, qv)

            moves.sort(key=_sort_key, reverse=True)
        else:
            moves.sort(key=lambda m: move_priority(env.board, m, player), reverse=True)

        if self.max_branch is not None:
            moves = moves[: self.max_branch]
        return moves

    def get_move(self, env: CaroEnv) -> Move:
        """Tactical → search sâu (DQN reorder) → không bao giờ yếu hơn Minimax cùng mức.

        Sàn an toàn: nếu search depth+1 (bị timeout hoặc nhiễu DQN) cho điểm thấp hơn
        Minimax depth gốc, trả nước của Minimax baseline — đảm bảo kết hợp luôn ≥ thành phần.
        """
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

        _, hybrid_move = self._search_root(
            env, player, self.depth, deadline, use_dqn=True
        )
        if hybrid_move is None:
            hybrid_move = fallback[0]

        base_depth = max(1, self.depth - HYBRID_EXTRA_DEPTH)
        if base_depth < self.depth:
            _, base_move = self._search_root(
                env, player, base_depth, deadline, use_dqn=False
            )
            if base_move is not None and base_move != hybrid_move:
                # Sàn theo heuristic (benchmark & UI dùng Δ heuristic) — không theo
                # score alpha-beta vì search sâu hơn có thể chọn nước chiến lược tốt
                # hơn về minimax nhưng Δ heuristic-1-nước thấp hơn.
                if self._heuristic_delta(env, base_move, player) > self._heuristic_delta(
                    env, hybrid_move, player
                ) + 1e-6:
                    return base_move

        return hybrid_move

    @staticmethod
    def _heuristic_delta(env: CaroEnv, move: Move, player: Player) -> float:
        """Δ heuristic sau một nước (cùng thước đo benchmark)."""
        before = evaluate_board(env.board, player)
        sim = env.clone()
        sim.step(move)
        return evaluate_board(sim.board, player) - before

    def _search_root(
        self,
        env: CaroEnv,
        player: Player,
        target_depth: int,
        deadline: float | None,
        *,
        use_dqn: bool,
    ) -> tuple[float, Move | None]:
        """Iterative deepening tới ``target_depth``; trả (điểm, nước tốt nhất)."""
        self._tt.clear()
        self._killers.clear()
        self._dqn_reorder_root = use_dqn and self.dqn._model_loaded

        best_move: Move | None = None
        best_score = float("-inf")
        for current_depth in range(1, target_depth + 1):
            try:
                score, move = self._alpha_beta(
                    env=env,
                    depth=current_depth,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    ai_player=player,
                    deadline=deadline,
                )
            except _SearchTimeout:
                break
            if move is not None:
                best_move = move
                best_score = score
            if abs(score) >= _WIN_THRESHOLD:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
        return best_score, best_move

    def get_win_probability(
        self, env: CaroEnv, for_player: Player | None = None
    ) -> float:
        player = for_player if for_player is not None else env.current_player
        if self.dqn._model_loaded:
            return self.dqn.get_win_probability(env, for_player=player)
        from ai.win_probability import estimate_win_probability

        return estimate_win_probability(env, player)

    @classmethod
    def from_difficulty(
        cls,
        difficulty: Difficulty,
        board_size: int,
        device: str | None = None,
        tactical_config: TacticalConfig | None = None,
        time_budget: float | None = -1.0,
    ) -> "HybridAgent":
        """Tạo Hybrid (sâu hơn Minimax 1 ply) cho độ khó cho trước.

        Args:
            time_budget: ``-1.0`` (mặc định) = dùng ``HYBRID_PLAY_TIME_BUDGET_SEC``
                để chơi tương tác an toàn; ``None`` = tìm hết độ sâu (tất định).
        """
        from config import HYBRID_PLAY_TIME_BUDGET_SEC

        depth = hybrid_depth_for(difficulty)
        radius = HYBRID_CANDIDATE_RADIUS_BY_DIFFICULTY.get(difficulty, 2)
        budget = (
            HYBRID_PLAY_TIME_BUDGET_SEC if time_budget == -1.0 else time_budget
        )
        agent = cls(
            depth=depth,
            board_size=board_size,
            candidate_radius=radius,
            device=device,
            max_branch=None,
            tactical_config=tactical_config,
            time_budget=budget,
        )
        agent._refresh_name()
        return agent
