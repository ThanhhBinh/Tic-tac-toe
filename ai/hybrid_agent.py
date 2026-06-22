#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tác nhân Hybrid: Minimax (Alpha-Beta) + DQN đánh giá node lá.

Ý tưởng: giữ tìm kiếm có giới hạn độ sâu (depth 2–3) của Minimax để nhìn xa
cục bộ, kết hợp heuristic pattern-based (ổn định) với Q-value DQN khi đã train.

Chiến lược (khi đã nạp DQN):
    1. Minimax + Alpha-Beta với **heuristic thuần** ở node lá — chất lượng ≥ Minimax.
    2. **Tinh chỉnh ở root**: DQN trộn với heuristic trên top-K nước ứng viên.
    → Tránh gọi DQN hàng trăm lần mỗi lượt (chậm, nhiễu) mà không đổi nước đi.

LƯU Ý: Nếu chưa có checkpoint DQN, node lá dùng THUẦN heuristic — mạng ngẫu nhiên
sẽ làm AI yếu hơn Minimax thuần nếu không blend.
"""

from __future__ import annotations

import numpy as np

from ai.board_encoder import legal_action_mask
from ai.dqn_agent import DQNAgent
from ai.heuristic import evaluate_position, find_tactical_move
from ai.minimax_agent import MinimaxAgent
from config import (
    HYBRID_CANDIDATE_RADIUS_BY_DIFFICULTY,
    HYBRID_DEPTH_BY_DIFFICULTY,
    HYBRID_LEAF_HEURISTIC_WEIGHT,
    HYBRID_MAX_BRANCH_BY_DIFFICULTY,
    HYBRID_ROOT_REFINE_CANDIDATES,
    HYBRID_ROOT_REFINE_MIN_GAIN,
    Difficulty,
    Player,
    TacticalConfig,
)
from core.caro_env import CaroEnv
from core.constants import Move


class HybridAgent(MinimaxAgent):
    """Minimax + Alpha-Beta; DQN tinh chỉnh ở root khi đã train."""

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
        root_refine_candidates: int = HYBRID_ROOT_REFINE_CANDIDATES,
        tactical_config: TacticalConfig | None = None,
    ) -> None:
        """Khởi tạo Hybrid: Minimax depth cố định + mạng DQN suy luận.

        Args:
            depth: Độ sâu Minimax (ply); nên <= 3 cho Hybrid.
            board_size: Cạnh bàn cờ (phải khớp checkpoint DQN).
            candidate_radius: Bán kính sinh nước ứng viên.
            device: Thiết bị PyTorch cho DQN ('cpu' / 'cuda' / None).
            cache_size: Dự phòng (cache node lá không còn dùng khi refine root).
            max_branch: Giới hạn nhánh mở rộng mỗi node (None = không giới hạn).
            leaf_heuristic_weight: Trọng số heuristic khi trộn với DQN ở root.
            root_refine_candidates: Số nước top-K DQN xem xét sau Minimax.
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
        self._root_refine_candidates = max(1, root_refine_candidates)
        self._heuristic_only_search = False

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

    def _blended_score(self, env: CaroEnv, ai_player: Player) -> float:
        """Trộn heuristic + DQN cho một thế cờ (dùng ở root refine)."""
        heuristic = self._heuristic_score(env, ai_player)
        if not self.dqn._model_loaded:
            return heuristic
        dqn = self._scale_dqn_to_heuristic(
            self._dqn_raw_score(env, ai_player), heuristic
        )
        w = self._leaf_heuristic_weight
        return w * heuristic + (1.0 - w) * dqn

    def _evaluate_leaf(self, env: CaroEnv, ai_player: Player) -> float:
        """Node lá: heuristic thuần khi search (đảm bảo ≥ Minimax thuần).

        DQN chỉ tham gia ở bước tinh chỉnh root sau khi Minimax chọn nước cơ sở.
        """
        if env.done:
            return self._heuristic_score(env, ai_player)

        if not self.dqn._model_loaded or self._heuristic_only_search:
            return self._heuristic_score(env, ai_player)

        # Legacy path: trộn tại lá (chỉ khi gọi trực tiếp, không qua get_move).
        heuristic = self._heuristic_score(env, ai_player)
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

    def _refine_root_with_dqn(self, env: CaroEnv, minimax_move: Move) -> Move:
        """Chọn lại trong top-K nước nếu DQN+heuristic rõ ràng tốt hơn Minimax."""
        player = env.current_player
        ordered = self._ordered_moves(env, player)
        k = self._root_refine_candidates
        candidates = ordered[:k]
        if minimax_move not in candidates:
            candidates = [minimax_move] + [m for m in candidates if m != minimax_move][: k - 1]

        best_move = minimax_move
        best_score = float("-inf")
        minimax_score: float | None = None

        for move in candidates:
            sim = env.clone()
            sim.step(move)
            blended = self._blended_score(sim, player)
            if move == minimax_move:
                minimax_score = blended
            if blended > best_score:
                best_score = blended
                best_move = move

        if (
            best_move != minimax_move
            and minimax_score is not None
            and best_score >= minimax_score * HYBRID_ROOT_REFINE_MIN_GAIN
        ):
            return best_move
        return minimax_move

    def get_move(self, env: CaroEnv) -> Move:
        """Minimax heuristic thuần, rồi tinh chỉnh bằng DQN ở root nếu có model."""
        self._eval_cache.clear()
        player = env.current_player

        tactical = find_tactical_move(
            env,
            player,
            radius=self.candidate_radius,
            config=self.tactical_config,
        )
        if tactical is not None:
            return tactical

        self._heuristic_only_search = True
        minimax_move = super().get_move(env)
        self._heuristic_only_search = False

        if not self.dqn._model_loaded:
            return minimax_move
        return self._refine_root_with_dqn(env, minimax_move)

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
        # Đã train DQN: duyệt đủ nhánh như Minimax thuần (không cắt top-8).
        if agent.dqn._model_loaded:
            agent.max_branch = None
            agent.name = f"Hybrid (depth={agent.depth}, DQN đã nạp)"
        else:
            # Chưa train DQN → node lá chỉ heuristic, tăng depth bằng Minimax thuần.
            agent.depth = max(agent.depth, int(difficulty))
            agent.name = (
                f"Hybrid (depth={agent.depth}, heuristic — chưa train DQN)"
            )
        return agent
