#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tác nhân Hybrid: Minimax (Alpha-Beta) + DQN đánh giá node lá.

Ý tưởng: giữ tìm kiếm có giới hạn độ sâu (depth 2–3) của Minimax để nhìn xa
cục bộ, kết hợp heuristic pattern-based (ổn định) với Q-value DQN khi đã train.

LƯU Ý: Nếu chưa có checkpoint DQN, node lá dùng THUẦN heuristic — mạng ngẫu nhiên
sẽ làm AI yếu hơn Minimax thuần nếu không blend.
"""

from __future__ import annotations

import numpy as np

from ai.board_encoder import legal_action_mask
from ai.dqn_agent import DQNAgent
from ai.heuristic import evaluate_position
from ai.minimax_agent import MinimaxAgent
from config import (
    HYBRID_CANDIDATE_RADIUS_BY_DIFFICULTY,
    HYBRID_DEPTH_BY_DIFFICULTY,
    HYBRID_LEAF_HEURISTIC_WEIGHT,
    HYBRID_MAX_BRANCH_BY_DIFFICULTY,
    Difficulty,
    Player,
    TacticalConfig,
)
from core.caro_env import CaroEnv
from core.constants import Move


class HybridAgent(MinimaxAgent):
    """Minimax + Alpha-Beta với lượng giá lá heuristic + DQN (có cache)."""

    name = "Hybrid"

    def __init__(
        self,
        depth: int = Difficulty.MEDIUM,
        board_size: int = 15,
        candidate_radius: int = 2,
        device: str | None = None,
        cache_size: int = 8_192,
        max_branch: int | None = None,
        leaf_heuristic_weight: float = HYBRID_LEAF_HEURISTIC_WEIGHT,
        tactical_config: TacticalConfig | None = None,
    ) -> None:
        """Khởi tạo Hybrid: Minimax depth cố định + mạng DQN suy luận.

        Args:
            depth: Độ sâu Minimax (ply); nên <= 3 cho Hybrid.
            board_size: Cạnh bàn cờ (phải khớp checkpoint DQN).
            candidate_radius: Bán kính sinh nước ứng viên.
            device: Thiết bị PyTorch cho DQN ('cpu' / 'cuda' / None).
            cache_size: Số lượng giá trị DQN cache tối đa tại node lá.
            max_branch: Giới hạn nhánh mở rộng mỗi node (None = không giới hạn).
            leaf_heuristic_weight: Trọng số heuristic khi trộn với DQN (0–1).
        """
        super().__init__(
            depth=depth,
            candidate_radius=candidate_radius,
            max_branch=max_branch,
            tactical_config=tactical_config,
        )
        self.board_size = board_size
        self.dqn = DQNAgent(board_size=board_size, epsilon=0.0, device=device)
        self._eval_cache: dict[tuple[bytes, int, int], float] = {}
        self._cache_size = cache_size
        self._leaf_heuristic_weight = leaf_heuristic_weight

        if self.dqn._model_loaded:
            dqn_tag = "DQN đã nạp"
        else:
            dqn_tag = "heuristic (chưa train DQN)"
        self.name = f"Hybrid (depth={self.depth}, {dqn_tag})"

    def _heuristic_score(self, env: CaroEnv, ai_player: Player) -> float:
        """Điểm heuristic pattern-based tại node lá."""
        return evaluate_position(env.winner, env.board, ai_player)

    def _dqn_raw_score(self, env: CaroEnv, ai_player: Player) -> float:
        """Điểm Q-value thô từ mạng (không cache)."""
        current = env.current_player
        q = self.dqn._predict_q_numpy(env, current)
        mask = legal_action_mask(env.board)
        if not mask.any():
            return 0.0
        legal_q = q[mask]
        if legal_q.size == 0 or not np.isfinite(legal_q).any():
            return 0.0
        best_q = float(np.max(legal_q))
        return best_q if current is ai_player else -best_q

    def _scale_dqn_to_heuristic(self, dqn_score: float, heuristic: float) -> float:
        """Scale Q-value nhỏ về cùng magnitude với heuristic (~10^3)."""
        if abs(dqn_score) <= 10.0:
            scale = max(5000.0, abs(heuristic) * 0.5, 1.0)
            return dqn_score * scale
        return dqn_score

    def _evaluate_leaf(self, env: CaroEnv, ai_player: Player) -> float:
        """Đánh giá node lá: heuristic (+ DQN nếu đã train).

        Khi chưa có model DQN, chỉ dùng heuristic — tránh mạng ngẫu nhiên phá search.
        """
        if env.done:
            return self._heuristic_score(env, ai_player)

        heuristic = self._heuristic_score(env, ai_player)

        if not self.dqn._model_loaded:
            return heuristic

        cache_key = (env.board.tobytes(), int(env.current_player), int(ai_player))
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        dqn = self._scale_dqn_to_heuristic(
            self._dqn_raw_score(env, ai_player), heuristic
        )
        w = self._leaf_heuristic_weight
        score = w * heuristic + (1.0 - w) * dqn

        if len(self._eval_cache) >= self._cache_size:
            self._eval_cache.clear()
        self._eval_cache[cache_key] = score
        return score

    def get_move(self, env: CaroEnv) -> Move:
        """Xóa cache DQN trước mỗi lượt để tránh stale state."""
        self._eval_cache.clear()
        return super().get_move(env)

    def get_win_probability(
        self, env: CaroEnv, for_player: Player | None = None
    ) -> float:
        """DQN nếu đã nạp model; ngược lại heuristic."""
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
    ) -> "HybridAgent":
        """Tạo HybridAgent; depth lấy từ map riêng (Expert = 3, không phải 4).

        Args:
            difficulty: Mức độ khó trong Settings.
            board_size: Kích thước bàn cờ.
            device: Thiết bị PyTorch.

        Returns:
            HybridAgent với depth đã giới hạn cho hiệu năng UI.
        """
        depth = HYBRID_DEPTH_BY_DIFFICULTY.get(difficulty, 2)
        max_branch = HYBRID_MAX_BRANCH_BY_DIFFICULTY.get(difficulty, 8)
        radius = HYBRID_CANDIDATE_RADIUS_BY_DIFFICULTY.get(difficulty, 2)
        agent = cls(
            depth=depth,
            board_size=board_size,
            candidate_radius=radius,
            device=device,
            max_branch=max_branch,
            tactical_config=tactical_config,
        )
        # Chưa train DQN → node lá chỉ heuristic, không tốn forward mạng.
        # Tăng depth lên bằng Minimax thuần (Expert=4) để không yếu hơn đáng kể.
        if not agent.dqn._model_loaded:
            agent.depth = max(agent.depth, int(difficulty))
            agent.name = (
                f"Hybrid (depth={agent.depth}, heuristic — chưa train DQN)"
            )
        return agent
