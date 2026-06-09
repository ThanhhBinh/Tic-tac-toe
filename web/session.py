#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quản lý phiên chơi cho giao diện web.

Tách logic điều phối ván đấu khỏi Pygame: dùng ``CaroEnv`` và ``Agent`` từ
``core/`` / ``ai/``, không import UI desktop.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ai.base_agent import Agent
from ai.factory import create_agent
from ai.threats import analyze_threats
from config import (
    AIType,
    DEFAULT_AI_FIRST,
    Difficulty,
    GameMode,
    Player,
    TacticalConfig,
)
from core.caro_env import CaroEnv
from core.constants import Move


@dataclass
class SessionSettings:
    """Cấu hình ván chơi từ form web."""

    mode: GameMode = GameMode.PVA
    ai_type: AIType = AIType.HYBRID
    difficulty: Difficulty = Difficulty.MEDIUM
    board_size: int = 15
    double_end_block_rule: bool = True
    threat_warnings: bool = True
    ai_aggressive: bool = True
    ai_first: bool = DEFAULT_AI_FIRST

    @property
    def tactical_config(self) -> TacticalConfig:
        """Cấu hình chiến thuật truyền sang agent."""
        return TacticalConfig(
            double_end_block_rule=self.double_end_block_rule,
            aggressive=self.ai_aggressive,
            threat_warnings=self.threat_warnings,
        )


class GameSession:
    """Một ván cờ Caro phục vụ API web (PvP / PvA / AvA)."""

    def __init__(self, settings: SessionSettings) -> None:
        """Khởi tạo môi trường, agent và lịch sử undo/redo."""
        self.id: str = str(uuid.uuid4())
        self.settings = settings
        self.env = CaroEnv(size=settings.board_size)
        self.env.reset()
        self.agents: dict[Player, Agent | None] = self._build_agents()
        self._history: list[CaroEnv] = [self.env.clone()]
        self._history_index: int = 0
        self._run_initial_ai_turn()

    def _run_initial_ai_turn(self) -> None:
        """Nếu AI đi trước (PvA), chạy nước mở đầu ngay khi tạo ván."""
        if not self._is_pva() or not self.settings.ai_first or self.env.done:
            return
        if self.agents.get(self.env.current_player) is not None:
            self._run_ai_turns()

    def _build_agents(self) -> dict[Player, Agent | None]:
        """Gán agent theo chế độ chơi."""
        ai = lambda: create_agent(  # noqa: E731
            self.settings.ai_type,
            self.settings.difficulty,
            self.settings.board_size,
            tactical_config=self.settings.tactical_config,
        )
        if self.settings.mode is GameMode.PVP:
            return {Player.X: None, Player.O: None}
        if self.settings.mode is GameMode.PVA:
            if self.settings.ai_first:
                return {Player.X: ai(), Player.O: None}
            return {Player.X: None, Player.O: ai()}
        return {Player.X: ai(), Player.O: ai()}

    def _human_player(self) -> Player | None:
        """Người chơi thật (phía không có AI) — chỉ có trong PvA."""
        for player in (Player.X, Player.O):
            if self.agents[player] is None:
                return player
        return None

    def _is_pva(self) -> bool:
        """True nếu chế độ Người vs AI."""
        return self.settings.mode is GameMode.PVA

    def _push_history(self) -> None:
        """Lưu snapshot sau mỗi nước đi."""
        self._history = self._history[: self._history_index + 1]
        self._history.append(self.env.clone())
        self._history_index += 1

    def _undo_steps(self) -> int:
        """Số bước lùi khi undo (PvA thường lùi cả cặp người + AI)."""
        if self._history_index <= 0:
            return 0
        if not self._is_pva():
            return 1
        human = self._human_player()
        if human is None:
            return 1
        snapshot = self._history[self._history_index]
        if (
            not snapshot.done
            and snapshot.current_player == human
            and snapshot.move_count >= 2
        ):
            return min(2, self._history_index)
        return 1

    def can_undo(self) -> bool:
        """Có thể quay lại không."""
        return self._undo_steps() > 0

    def can_redo(self) -> bool:
        """Có thể làm lại không."""
        return self._history_index < len(self._history) - 1

    def _restore_history(self, index: int) -> None:
        """Khôi phục trạng thái từ lịch sử."""
        index = max(0, min(index, len(self._history) - 1))
        self._history_index = index
        self.env.copy_state_from(self._history[index])

    def undo(self) -> dict[str, Any]:
        """Quay lại nước trước."""
        steps = self._undo_steps()
        if steps <= 0:
            raise ValueError("Không thể quay lại thêm.")
        self._restore_history(self._history_index - steps)
        return self.to_dict()

    def redo(self) -> dict[str, Any]:
        """Làm lại nước đã bị quay lại."""
        if not self.can_redo():
            raise ValueError("Không thể làm lại.")
        self._restore_history(self._history_index + 1)
        return self.to_dict()

    def _apply_move(self, move: Move) -> None:
        """Đặt quân và cập nhật lịch sử."""
        if not self.env.is_legal(move):
            raise ValueError(f"Nước đi không hợp lệ: {move}.")
        self.env.step(move)
        self._push_history()

    def _run_ai_turns(self) -> None:
        """Chạy lượt AI liên tiếp cho tới khi tới lượt người hoặc hết ván."""
        while not self.env.done:
            agent = self.agents[self.env.current_player]
            if agent is None:
                break
            move = agent.get_move(self.env.clone())
            self._apply_move(move)

    def step_ai_turn(self) -> dict[str, Any]:
        """Tiến đúng một lượt AI (dùng cho AvA tự chạy trên web).

        Returns:
            Trạng thái ván sau một nước AI.

        Raises:
            ValueError: Ván đã kết thúc hoặc không phải lượt AI.
        """
        if self.env.done:
            raise ValueError("Ván đã kết thúc.")
        agent = self.agents[self.env.current_player]
        if agent is None:
            raise ValueError("Hiện không phải lượt AI.")
        move = agent.get_move(self.env.clone())
        self._apply_move(move)
        return self.to_dict()

    def play_move(self, row: int, col: int) -> dict[str, Any]:
        """Người chơi đặt quân; sau đó AI phản hồi nếu cần.

        Args:
            row: Hàng trên bàn cờ.
            col: Cột trên bàn cờ.

        Returns:
            Trạng thái ván sau nước đi.
        """
        if self.env.done:
            raise ValueError("Ván đã kết thúc.")
        if self.agents[self.env.current_player] is not None:
            raise ValueError("Hiện không phải lượt người chơi.")

        self._apply_move((row, col))
        if not self.env.done:
            self._run_ai_turns()
        return self.to_dict()

    def step_ava(self) -> dict[str, Any]:
        """Tiến một lượt AI vs AI (một nước/lần — frontend gọi lặp để tự chạy)."""
        if self.settings.mode is not GameMode.AVA:
            raise ValueError("Chỉ dùng trong chế độ AI vs AI.")
        return self.step_ai_turn()

    def _win_probability(self) -> tuple[float | None, str]:
        """Xác suất thắng và nguồn ước lượng (dqn / heuristic)."""
        viewer = self._human_player()
        if viewer is None and not self.env.done:
            viewer = self.env.current_player

        from ai.win_probability import estimate_win_probability

        for agent in self.agents.values():
            if agent is not None:
                prob = agent.get_win_probability(self.env, for_player=viewer)
                if prob is not None:
                    uses_dqn = False
                    if hasattr(agent, "dqn") and agent.dqn._model_loaded:  # type: ignore[attr-defined]
                        uses_dqn = True
                    elif getattr(agent, "_model_loaded", False):
                        uses_dqn = True
                    return prob, "dqn" if uses_dqn else "heuristic"

        if viewer is not None:
            return estimate_win_probability(self.env, viewer), "heuristic"
        return None, "none"

    def _warning_player(self) -> Player | None:
        """Người chơi nhận cảnh báo đe dọa trên UI."""
        if self.env.done or self.settings.mode is GameMode.AVA:
            return None
        if self.settings.mode is GameMode.PVP:
            return self.env.current_player
        return self._human_player()

    def _threats_payload(self) -> dict[str, Any] | None:
        """Cảnh báo sắp thắng cho người chơi (PvP/PvA)."""
        if not self.settings.threat_warnings:
            return None
        viewer = self._warning_player()
        if viewer is None:
            return None
        analysis = analyze_threats(
            self.env, viewer, config=self.settings.tactical_config
        )
        if not (
            analysis.win_moves
            or analysis.block_moves
            or analysis.double_end_blocks
            or analysis.threat_stones
        ):
            return None
        return {
            "win_moves": [list(m) for m in analysis.win_moves],
            "block_moves": [list(m) for m in analysis.block_moves],
            "double_end_blocks": [list(m) for m in analysis.double_end_blocks],
            "threat_stones": [list(m) for m in analysis.threat_stones],
            "message": analysis.message,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize trạng thái ván cho JSON API."""
        human = self._human_player()
        current_agent = self.agents.get(self.env.current_player)
        is_ava = self.settings.mode is GameMode.AVA
        win_prob, win_prob_source = self._win_probability()
        return {
            "session_id": self.id,
            "board": self.env.board.tolist(),
            "board_size": self.env.size,
            "current_player": self.env.current_player.name,
            "done": self.env.done,
            "is_draw": self.env.is_draw,
            "winner": self.env.winner.name if self.env.winner else None,
            "winning_line": list(self.env.winning_line),
            "last_move": list(self.env.last_move) if self.env.last_move else None,
            "move_count": self.env.move_count,
            "can_undo": self.can_undo(),
            "can_redo": self.can_redo(),
            "win_probability": win_prob,
            "win_probability_source": win_prob_source,
            "is_human_turn": current_agent is None and not self.env.done,
            "is_ava": is_ava,
            "human_player": human.name if human else None,
            "settings": {
                "mode": self.settings.mode.value,
                "ai_type": self.settings.ai_type.value,
                "difficulty": self.settings.difficulty.name,
                "board_size": self.settings.board_size,
                "double_end_block_rule": self.settings.double_end_block_rule,
                "threat_warnings": self.settings.threat_warnings,
                "ai_aggressive": self.settings.ai_aggressive,
                "ai_first": self.settings.ai_first,
            },
            "threats": self._threats_payload(),
            "status_text": self._status_text(),
        }

    def _status_text(self) -> str:
        """Mô tả trạng thái hiện tại cho HUD."""
        if self.env.done:
            if self.env.is_draw:
                return "Hòa cờ"
            assert self.env.winner is not None
            return f"{self.env.winner.name} thắng"
        agent = self.agents[self.env.current_player]
        if agent is None:
            return f"Lượt {self.env.current_player.name} — Bạn"
        return f"Lượt {self.env.current_player.name} — {agent.name}"


@dataclass
class SessionStore:
    """Kho phiên in-memory (đủ cho dev/local)."""

    sessions: dict[str, GameSession] = field(default_factory=dict)

    def create(self, settings: SessionSettings) -> GameSession:
        """Tạo ván mới và lưu vào kho."""
        session = GameSession(settings)
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> GameSession:
        """Lấy phiên theo id.

        Raises:
            KeyError: Nếu không tìm thấy.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Phiên {session_id} không tồn tại.")
        return self.sessions[session_id]

    def delete(self, session_id: str) -> None:
        """Xoá phiên khỏi kho."""
        self.sessions.pop(session_id, None)
