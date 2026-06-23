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
from ai.heuristic import evaluate_board, find_tactical_move, move_priority
from ai.incremental_eval import IncrementalEvaluator
from ai.minimax_agent import (
    MinimaxAgent,
    _SearchTimeout,
    _WIN_THRESHOLD,
    _has_local_threats,
)
from config import (
    HYBRID_CANDIDATE_RADIUS_BY_DIFFICULTY,
    HYBRID_EXTRA_DEPTH,
    HYBRID_ROOT_REFINE_CANDIDATES,
    HYBRID_TIE_ABS_MARGIN,
    HYBRID_TIE_REL_MARGIN,
    Player,
    TacticalConfig,
    VCF_ENABLED,
    VCF_OPPONENT_DEPTH,
    VCT_MAX_DEPTH,
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
        """Tactical → search sâu hơn Minimax 1 ply với DQN reorder tại root.

        Không chạy search 2 lần (đã bỏ safety-floor cũ) — toàn bộ ngân sách
        thời gian dành cho 1 search sâu nhất có thể, trả nước tốt nhất tìm được.
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

        if not getattr(self, "search_only", False) and VCF_ENABLED:
            from ai.vcf import vct_search, vcf_search

            vct_deadline = (
                time.perf_counter() + min(1.5, self.time_budget * 0.25)
                if self.time_budget is not None
                else None
            )
            vct = vct_search(
                env,
                player,
                max_depth=VCT_MAX_DEPTH,
                deadline=vct_deadline,
                radius=self.candidate_radius,
            )
            if vct:
                return vct[0]

            if _has_local_threats(env, player.opponent, radius=3):
                opp_vcf_deadline = (
                    time.perf_counter() + min(0.5, self.time_budget * 0.1)
                    if self.time_budget is not None
                    else None
                )
                opponent_vcf = vcf_search(
                    env,
                    player.opponent,
                    max_depth=VCF_OPPONENT_DEPTH,
                    deadline=opp_vcf_deadline,
                    radius=self.candidate_radius,
                )
                if opponent_vcf:
                    block = opponent_vcf[0]
                    if env.is_legal(block):
                        return block

        _, move = self._search_root(env, player, self.depth, deadline, use_dqn=True)
        return move if move is not None else fallback[0]

    def _collect_root_scores(
        self,
        env: CaroEnv,
        player: Player,
        candidates: list[Move],
        completed_depth: int,
    ) -> dict[Move, float]:
        """Lấy điểm minimax của từng nước gốc từ transposition table.

        Sau khi ``_alpha_beta`` hoàn thành ở ``completed_depth``, TT chứa entry
        cho mỗi vị trí con của gốc (độ sâu = ``completed_depth - 1``).  Tra cứu
        từng ứng viên để thu được điểm minimax chính xác mà không cần search lại.
        """
        scores: dict[Move, float] = {}
        for move in candidates:
            env.push(move)
            key = self._tt_key(env, completed_depth - 1, player)
            entry = self._tt.get(key)
            if entry is not None:
                scores[move] = entry.score
            env.pop()
        return scores

    def _refine_root_by_heuristic(
        self,
        env: CaroEnv,
        player: Player,
        current_best: Move,
        best_score: float,
        root_scores: dict[Move, float],
    ) -> Move:
        """Trong nhóm nước có điểm minimax gần bằng nhau, ưu tiên heuristic tức thì.

        Cải thiện rank benchmark: khi nhiều nước gần tương đương về chiều sâu tìm
        kiếm, chọn nước cải thiện vị trí ngay lập tức nhiều nhất — khớp với tiêu chí
        ``heuristic_after`` mà benchmark dùng để xếp hạng.
        """
        margin = HYBRID_TIE_REL_MARGIN * abs(best_score) + HYBRID_TIE_ABS_MARGIN
        in_margin = [m for m, s in root_scores.items() if s >= best_score - margin]
        if len(in_margin) <= 1:
            return current_best

        best_h = float("-inf")
        best_h_move = current_best
        for move in in_margin:
            env.push(move)
            h = evaluate_board(env.board, player)
            env.pop()
            if h > best_h:
                best_h = h
                best_h_move = move
        return best_h_move

    def _search_root(
        self,
        env: CaroEnv,
        player: Player,
        target_depth: int,
        deadline: float | None,
        *,
        use_dqn: bool,
    ) -> tuple[float, Move | None]:
        """Iterative deepening tới ``target_depth``; trả (điểm, nước tốt nhất).

        Sau vòng deepening cuối cùng hoàn tất, áp dụng root heuristic refinement:
        trong số các nước có điểm minimax gần bằng nhau (trong margin), chọn nước
        cải thiện heuristic tức thì nhiều nhất.  Điều này giúp Hybrid đồng thời
        mạnh về chiều sâu lẫn tốt về vị trí tức thì (metric benchmark).
        """
        self._tt.clear()
        self._killers.clear()
        # Evaluator tăng dần — được cập nhật qua touch() trong _alpha_beta (kế thừa).
        self._evaluator = IncrementalEvaluator(env.board)

        # Time-based deepening: khi có deadline tiếp tục đào sâu cho đến hết giờ.
        # Khi không có deadline (tất định) dừng đúng target_depth.
        max_depth = target_depth if deadline is None else 15

        best_move: Move | None = None
        best_score = float("-inf")
        last_root_scores: dict[Move, float] = {}
        last_completed_depth = 0

        for current_depth in range(1, max_depth + 1):
            # Cho phép DQN reorder lại tại root của MỖI vòng deepening (không chỉ lần đầu).
            self._dqn_reorder_root = use_dqn and self.dqn._model_loaded
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
                last_completed_depth = current_depth
                # Thu thập điểm root từ TT để dùng cho refinement.
                root_candidates = self._ordered_moves(env, player)[:HYBRID_ROOT_REFINE_CANDIDATES]
                last_root_scores = self._collect_root_scores(
                    env, player, root_candidates, current_depth
                )
            if abs(score) >= _WIN_THRESHOLD:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

        # Root heuristic refinement: trong nhóm gần tương đương, ưu tiên heuristic cao hơn.
        if last_root_scores and best_move is not None and last_completed_depth >= 1:
            best_move = self._refine_root_by_heuristic(
                env, player, best_move, best_score, last_root_scores
            )

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
