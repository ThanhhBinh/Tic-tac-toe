#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Màn hình chơi: vẽ bàn cờ, xử lý lượt đi và hiển thị HUD/sidebar.

Tuân thủ nguyên tắc tách lớp: màn hình này chỉ ĐỌC trạng thái từ `CaroEnv` và
hỏi nước đi từ các `Agent`; nó không tự cài đặt luật game.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pygame

from ai.base_agent import Agent
from ai.factory import create_agent
from ai.online_learner import GameMoveRecorder, learn_from_pva_game
from ai.threats import ThreatAnalysis, analyze_threats
from config import (
    AI_MOVE_TIMEOUT_SEC,
    END_OVERLAY_ANIM_DURATION,
    HOVER_PREVIEW_ALPHA,
    LAST_MOVE_PULSE_SPEED,
    SIDEBAR_WIDTH,
    WIN_PULSE_SPEED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    AIType,
    GameMode,
    Player,
    Theme,
    create_caro_env,
)
from core.caro_env import CaroEnv
from core.constants import Move
from ui.animations import OverlayAnimation, PlaceAnimation, pulse_alpha
from ui.screens.base import BaseScreen
from ui.theme import draw_round_rect, render_text
from ui.widgets import draw_button, draw_player_badge, draw_progress_bar

if TYPE_CHECKING:  # pragma: no cover
    from ui.app import App

_BOARD_MARGIN: int = 30


class GameScreen(BaseScreen):
    """Màn hình thi đấu chính với animation và HUD nâng cao."""

    def __init__(self, app: "App") -> None:
        """Khởi tạo các thuộc tính; ván thực sự được dựng trong on_enter()."""
        super().__init__(app)
        self.env: CaroEnv = create_caro_env(
            app.settings.board_size,
            double_end_block_rule=app.settings.double_end_block_rule,
        )
        self.agents: dict[Player, Agent | None] = {Player.X: None, Player.O: None}

        self.place_anim: PlaceAnimation | None = None
        self.end_overlay_anim: OverlayAnimation = OverlayAnimation(
            duration=END_OVERLAY_ANIM_DURATION
        )
        self.anim_time: float = 0.0

        self.ai_thinking: bool = False
        self.ai_think_elapsed: float = 0.0
        self._ai_thread: threading.Thread | None = None
        self._ai_move_result: Move | None = None
        self._ai_compute_done: bool = False

        self.win_probability: float | None = None
        self.win_probability_source: str = "none"
        self.hover_cell: Move | None = None
        self._was_done: bool = False
        self._end_overlay_dismissed: bool = False
        self._end_overlay_box: pygame.Rect | None = None

        self._history: list[CaroEnv] = []
        self._history_index: int = 0

        self.board_px: int = 0
        self.cell: float = 0.0
        self.origin: tuple[int, int] = (_BOARD_MARGIN, _BOARD_MARGIN)

        self._btn_replay: pygame.Rect | None = None
        self._btn_menu: pygame.Rect | None = None
        self._btn_dismiss_overlay: pygame.Rect | None = None
        self._btn_show_overlay: pygame.Rect | None = None
        self._btn_undo: pygame.Rect | None = None
        self._btn_redo: pygame.Rect | None = None
        self._mouse_pos: tuple[int, int] = (0, 0)
        self._threat_hint: ThreatAnalysis | None = None
        self._move_recorder = GameMoveRecorder()
        self._last_learn_message: str | None = None

    def on_enter(self) -> None:
        """Dựng lại môi trường và agent theo cấu hình hiện tại mỗi khi vào màn."""
        settings = self.app.settings
        self.env = create_caro_env(
            settings.board_size,
            double_end_block_rule=settings.double_end_block_rule,
        )
        self.env.reset()

        ai = lambda: create_agent(  # noqa: E731
            settings.ai_type,
            settings.difficulty,
            settings.board_size,
            tactical_config=settings.tactical_config,
        )
        if settings.mode is GameMode.PVP:
            self.agents = {Player.X: None, Player.O: None}
        elif settings.mode is GameMode.PVA:
            if settings.ai_first:
                self.agents = {Player.X: ai(), Player.O: None}
            else:
                self.agents = {Player.X: None, Player.O: ai()}
        else:
            self.agents = {Player.X: ai(), Player.O: ai()}

        self.place_anim = None
        self.end_overlay_anim.reset()
        self.anim_time = 0.0
        self._reset_ai_worker()
        self.win_probability = None
        self.hover_cell = None
        self._was_done = False
        self._end_overlay_dismissed = False
        self._end_overlay_box = None
        self._move_recorder.reset()
        self._last_learn_message = None
        self._init_history()
        self._compute_geometry()
        self._update_win_probability()
        self._update_threat_hints()

    def _warning_player(self) -> Player | None:
        """Người chơi nhận cảnh báo trên bàn cờ."""
        settings = self.app.settings
        if self.env.done or settings.mode is GameMode.AVA:
            return None
        if settings.mode is GameMode.PVP:
            if self.agents[self.env.current_player] is not None:
                return None
            return self.env.current_player
        return self._human_player()

    def _update_threat_hints(self) -> None:
        """Cập nhật đe dọa hiển thị trên HUD/bàn cờ."""
        self._threat_hint = None
        if not self.app.settings.threat_warnings:
            return
        viewer = self._warning_player()
        if viewer is None:
            return
        self._threat_hint = analyze_threats(
            self.env, viewer, config=self.app.settings.tactical_config
        )

    def _init_history(self) -> None:
        """Khởi tạo lịch sử nước đi (snapshot clone) cho undo/redo."""
        self._history = [self.env.clone()]
        self._history_index = 0

    def _push_history(self) -> None:
        """Lưu trạng thái sau nước đi mới; xoá nhánh redo cũ."""
        self._history = self._history[: self._history_index + 1]
        self._history.append(self.env.clone())
        self._history_index += 1

    def _human_player(self) -> Player | None:
        """Người chơi thật trong PvA (phía không có agent AI)."""
        for player in (Player.X, Player.O):
            if self.agents[player] is None:
                return player
        return None

    def _record_ai_move(self, env_before: CaroEnv, move: Move, ai_player: Player) -> None:
        """Ghi nước AI để học online sau ván."""
        self._move_recorder.record_ai_move(env_before, move, ai_player, self.env)

    def _maybe_learn_from_game(self) -> None:
        """Học từ ván vừa kết thúc (chạy nền để không treo UI)."""
        if not self._is_pva() or not self.env.done:
            return

        human = self._human_player()
        if human is None:
            return

        ai_player = human.opponent
        ai_agent = self.agents.get(ai_player)
        recorder = self._move_recorder
        board_size = self.env.size
        env_snapshot = self.env.clone()

        def worker() -> None:
            result = learn_from_pva_game(
                recorder,
                board_size,
                human,
                env_snapshot.winner,
                env_snapshot.is_draw,
                ai_agent,
            )
            if result is None:
                return
            if result.outcome == "ai_loss":
                self._last_learn_message = (
                    f"AI đã học từ thất bại ({result.ai_moves} nước, "
                    f"{result.gradient_steps} bước)"
                )
            elif result.outcome == "ai_win":
                self._last_learn_message = (
                    f"AI củng cố chiến thắng ({result.gradient_steps} bước)"
                )

        threading.Thread(target=worker, daemon=True).start()

    def _is_pva(self) -> bool:
        """True nếu đang chơi Người vs AI."""
        return self.app.settings.mode is GameMode.PVA

    def _is_busy(self) -> bool:
        """True khi đang animate hoặc AI đang tính — không cho undo/redo."""
        return (
            self.place_anim is not None
            or self.ai_thinking
            or self._ai_thread is not None
        )

    def _undo_steps(self) -> int:
        """Số bước lùi khi bấm Quay lại (PvA lùi cả cặp người+AI nếu có thể)."""
        if self._history_index <= 0:
            return 0
        if not self._is_pva():
            return 1

        human = self._human_player()
        if human is None:
            return 1

        snapshot = self._history[self._history_index]
        # Lượt người chơi + đã có ít nhất 2 nước → lùi cả nước AI và nước người.
        if (
            not snapshot.done
            and snapshot.current_player == human
            and snapshot.move_count >= 2
        ):
            return min(2, self._history_index)
        return 1

    def _can_undo(self) -> bool:
        """Có thể quay lại không."""
        return not self._is_busy() and self._undo_steps() > 0

    def _can_redo(self) -> bool:
        """Có thể làm lại (redo) không."""
        return not self._is_busy() and self._history_index < len(self._history) - 1

    def _restore_history(self, index: int) -> None:
        """Khôi phục bàn cờ từ snapshot lịch sử."""
        index = max(0, min(index, len(self._history) - 1))
        self._history_index = index
        self.env.copy_state_from(self._history[index])
        self._reset_ai_worker()
        self.place_anim = None
        self.hover_cell = None

        if self.env.done:
            if not self._was_done:
                self.end_overlay_anim.reset()
                self._was_done = True
        else:
            self._was_done = False
            self._end_overlay_dismissed = False
            self.end_overlay_anim.reset()

        self._update_win_probability()
        self._update_threat_hints()

    def _undo(self) -> None:
        """Quay lại nước trước (PvA: thường lùi cả cặp người + AI)."""
        steps = self._undo_steps()
        if steps <= 0:
            return
        self._move_recorder.invalidate()
        self._restore_history(self._history_index - steps)

    def _redo(self) -> None:
        """Làm lại nước đã bị quay lại."""
        if not self._can_redo():
            return
        self._restore_history(self._history_index + 1)

    def _dismiss_end_overlay(self) -> None:
        """Ẩn pop-up kết thúc để xem bàn cờ."""
        if self.env.done:
            self._end_overlay_dismissed = True

    def _show_end_overlay(self) -> None:
        """Hiện lại pop-up kết thúc."""
        if self.env.done:
            self._end_overlay_dismissed = False

    def _reset_ai_worker(self) -> None:
        """Dừng worker AI (nếu có) và xoá trạng thái suy nghĩ."""
        self._ai_thread = None
        self._ai_move_result = None
        self._ai_compute_done = False
        self.ai_thinking = False
        self.ai_think_elapsed = 0.0

    def _start_ai_worker(self, agent: Agent) -> None:
        """Chạy ``get_move`` trong thread nền để UI vẫn render mượt."""
        env_snapshot = self.env.clone()
        self._ai_move_result = None
        self._ai_compute_done = False

        def worker() -> None:
            try:
                move = agent.get_move(env_snapshot)
            except Exception:
                legal = env_snapshot.candidate_moves(radius=2) or env_snapshot.legal_moves()
                move = legal[0] if legal else (0, 0)
            self._ai_move_result = move
            self._ai_compute_done = True

        self._ai_thread = threading.Thread(target=worker, daemon=True)
        self._ai_thread.start()

    def _poll_ai_worker(self) -> None:
        """Kiểm tra thread AI; áp dụng nước đi hoặc timeout fallback."""
        if self._ai_thread is None:
            return

        if not self._ai_compute_done:
            if self.ai_think_elapsed >= AI_MOVE_TIMEOUT_SEC:
                legal = self.env.candidate_moves(radius=2) or self.env.legal_moves()
                if legal:
                    env_before = self.env.clone()
                    ai_player = self.env.current_player
                    self._reset_ai_worker()
                    self._apply_move(legal[0], trigger_learn=False)
                    if self.agents.get(ai_player) is not None:
                        self._record_ai_move(env_before, legal[0], ai_player)
                    if self.env.done:
                        self._maybe_learn_from_game()
            return

        if self._ai_move_result is not None:
            move = self._ai_move_result
            env_before = self.env.clone()
            ai_player = self.env.current_player
            self._reset_ai_worker()
            self._apply_move(move, trigger_learn=False)
            if self.agents.get(ai_player) is not None:
                self._record_ai_move(env_before, move, ai_player)
            if self.env.done:
                self._maybe_learn_from_game()

    def _compute_geometry(self) -> None:
        """Tính kích thước ô và gốc toạ độ bàn cờ vừa với cửa sổ."""
        avail_w = WINDOW_WIDTH - SIDEBAR_WIDTH - _BOARD_MARGIN * 2
        avail_h = WINDOW_HEIGHT - _BOARD_MARGIN * 2
        self.board_px = min(avail_w, avail_h)
        self.cell = self.board_px / self.env.size
        oy = (WINDOW_HEIGHT - self.board_px) // 2
        self.origin = (_BOARD_MARGIN, oy)

    def _cell_center(self, row: int, col: int) -> tuple[float, float]:
        """Trả về toạ độ pixel tâm của ô (row, col)."""
        ox, oy = self.origin
        return (ox + (col + 0.5) * self.cell, oy + (row + 0.5) * self.cell)

    def _pixel_to_cell(self, pos: tuple[int, int]) -> Move | None:
        """Chuyển toạ độ chuột sang (row, col), hoặc None nếu ngoài bàn cờ."""
        ox, oy = self.origin
        x, y = pos
        if not (ox <= x < ox + self.board_px and oy <= y < oy + self.board_px):
            return None
        col = int((x - ox) / self.cell)
        row = int((y - oy) / self.cell)
        if self.env.in_bounds(row, col):
            return (row, col)
        return None

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Xử lý phím tắt, click đặt quân và nút trên overlay kết thúc."""
        from ui.app import SCREEN_MENU

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.env.done and not self._end_overlay_dismissed:
                        self._dismiss_end_overlay()
                    else:
                        self.app.go_to(SCREEN_MENU)
                elif event.key == pygame.K_r:
                    self.on_enter()
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if self.env.done and not self._end_overlay_dismissed:
                        self._dismiss_end_overlay()
                elif self._is_pva():
                    ctrl = bool(event.mod & pygame.KMOD_CTRL)
                    shift = bool(event.mod & pygame.KMOD_SHIFT)
                    if event.key == pygame.K_u or (
                        event.key == pygame.K_z and ctrl and not shift
                    ):
                        self._undo()
                    elif event.key == pygame.K_y or (
                        event.key == pygame.K_z and ctrl and shift
                    ):
                        self._redo()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                self._mouse_pos = event.pos
                self._update_hover(event.pos)

    def _update_hover(self, pos: tuple[int, int]) -> None:
        """Cập nhật ô preview khi chuột di chuyển (lượt người chơi)."""
        self.hover_cell = None
        if self.env.done or self.place_anim is not None:
            return
        if self.agents[self.env.current_player] is not None:
            return
        cell = self._pixel_to_cell(pos)
        if cell is not None and self.env.is_legal(cell):
            self.hover_cell = cell

    def _handle_click(self, pos: tuple[int, int]) -> None:
        """Phân luồng click: overlay kết thúc, undo/redo hay đặt quân."""
        from ui.app import SCREEN_MENU

        if self._btn_undo and self._btn_undo.collidepoint(pos):
            self._undo()
            return
        if self._btn_redo and self._btn_redo.collidepoint(pos):
            self._redo()
            return
        if self._btn_show_overlay and self._btn_show_overlay.collidepoint(pos):
            self._show_end_overlay()
            return

        if self.env.done:
            if not self._end_overlay_dismissed:
                if self._btn_replay and self._btn_replay.collidepoint(pos):
                    self.on_enter()
                    return
                if self._btn_menu and self._btn_menu.collidepoint(pos):
                    self.app.go_to(SCREEN_MENU)
                    return
                if self._btn_dismiss_overlay and self._btn_dismiss_overlay.collidepoint(pos):
                    self._dismiss_end_overlay()
                    return
                if self._end_overlay_box and not self._end_overlay_box.collidepoint(pos):
                    self._dismiss_end_overlay()
                    return
            else:
                if self._btn_replay and self._btn_replay.collidepoint(pos):
                    self.on_enter()
                elif self._btn_menu and self._btn_menu.collidepoint(pos):
                    self.app.go_to(SCREEN_MENU)
            return

        if self.place_anim is not None:
            return
        if self.agents[self.env.current_player] is not None:
            return
        cell = self._pixel_to_cell(pos)
        if cell is not None and self.env.is_legal(cell):
            self._apply_move(cell)

    def update(self, dt: float) -> None:
        """Tiến hành animation và điều phối lượt đi của AI."""
        self.anim_time += dt
        self._mouse_pos = pygame.mouse.get_pos()
        self._update_hover(self._mouse_pos)

        if self.place_anim is not None:
            self.place_anim.elapsed += dt
            if self.place_anim.finished:
                self.place_anim = None
            return

        if self.env.done:
            if not self._was_done:
                self.end_overlay_anim.reset()
                self._was_done = True
                self._end_overlay_dismissed = False
            self.end_overlay_anim.tick(dt)
            return

        agent = self.agents[self.env.current_player]
        if agent is None:
            return

        if self._ai_thread is None and not self.ai_thinking:
            self.ai_thinking = True
            self.ai_think_elapsed = 0.0
            self._start_ai_worker(agent)

        if self.ai_thinking:
            self.ai_think_elapsed += dt
            self._poll_ai_worker()

    def _apply_move(self, move: Move, *, trigger_learn: bool = True) -> None:
        """Thực hiện nước đi lên môi trường và khởi động hiệu ứng đặt quân."""
        player = self.env.current_player
        self.env.step(move)
        self._push_history()
        self.place_anim = PlaceAnimation(move=move, player=player)
        self._update_win_probability()
        self._update_threat_hints()
        if trigger_learn and self.env.done:
            self._maybe_learn_from_game()

    def _update_win_probability(self) -> None:
        """Cập nhật xác suất thắng (DQN hoặc heuristic) cho người chơi trên HUD."""
        from ai.win_probability import estimate_win_probability

        viewer = self._human_player()
        if viewer is None and not self.env.done:
            viewer = self.env.current_player

        prob: float | None = None
        source = "none"
        for agent in self.agents.values():
            if agent is not None:
                prob = agent.get_win_probability(self.env, for_player=viewer)
                if prob is not None:
                    uses_dqn = False
                    if hasattr(agent, "dqn") and agent.dqn._model_loaded:  # type: ignore[attr-defined]
                        uses_dqn = True
                    elif getattr(agent, "_model_loaded", False):
                        uses_dqn = True
                    source = "dqn" if uses_dqn else "heuristic"
                    break

        if prob is None and viewer is not None:
            prob = estimate_win_probability(self.env, viewer)
            source = "heuristic"

        self.win_probability = prob
        self.win_probability_source = source

    def draw(self, surface: pygame.Surface) -> None:
        """Vẽ toàn bộ màn chơi: bàn cờ, quân, highlight, sidebar và overlay."""
        self._draw_board(surface)
        self._draw_threat_hints(surface)
        self._draw_hover_preview(surface)
        self._draw_stones(surface)
        self._draw_highlights(surface)
        self._draw_sidebar(surface)
        if self.env.done and not self._end_overlay_dismissed:
            self._draw_end_overlay(surface)

    def _draw_board(self, surface: pygame.Surface) -> None:
        """Vẽ nền gỗ gradient, lưới và điểm sao (hoshi) trên bàn lớn."""
        ox, oy = self.origin
        board_rect = pygame.Rect(ox, oy, self.board_px, self.board_px)

        pygame.draw.rect(surface, Theme.BOARD_WOOD_DARK, board_rect, border_radius=10)
        inner = board_rect.inflate(-6, -6)
        pygame.draw.rect(surface, Theme.BOARD_WOOD, inner, border_radius=8)

        for i in range(self.env.size + 1):
            x = ox + int(i * self.cell)
            y = oy + int(i * self.cell)
            pygame.draw.line(surface, Theme.GRID_LINE, (x, oy), (x, oy + self.board_px), 1)
            pygame.draw.line(surface, Theme.GRID_LINE, (ox, y), (ox + self.board_px, y), 1)

        if self.env.size >= 15:
            stars = (3, 7, 11)
            for r in stars:
                for c in stars:
                    cx, cy = self._cell_center(r, c)
                    pygame.draw.circle(surface, Theme.GRID_LINE, (int(cx), int(cy)), 4)

    def _draw_hover_preview(self, surface: pygame.Surface) -> None:
        """Vẽ quân mờ preview tại ô chuột đang hover (lượt người)."""
        if self.hover_cell is None:
            return
        row, col = self.hover_cell
        cx, cy = self._cell_center(row, col)
        r = self._stone_radius()
        player = self.env.current_player

        preview = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        color = Theme.STONE_X if player is Player.X else Theme.STONE_O
        pygame.draw.circle(preview, (*color, HOVER_PREVIEW_ALPHA), (r + 2, r + 2), r)
        surface.blit(preview, (cx - r - 2, cy - r - 2))

    def _stone_radius(self) -> int:
        """Bán kính quân cờ theo kích thước ô."""
        return int(self.cell * 0.42)

    def _draw_stones(self, surface: pygame.Surface) -> None:
        """Vẽ tất cả các quân; quân vừa đặt có scale + fade-in."""
        anim = self.place_anim
        anim_move = anim.move if anim else None
        radius = self._stone_radius()
        for row in range(self.env.size):
            for col in range(self.env.size):
                val = int(self.env.board[row, col])
                if val == Player.EMPTY:
                    continue
                is_animating = (
                    anim is not None
                    and not anim.finished
                    and anim_move == (row, col)
                )
                if is_animating:
                    scale = anim.scale()
                    alpha = anim.alpha()
                else:
                    scale = 1.0
                    alpha = 1.0
                self._draw_stone(
                    surface,
                    row,
                    col,
                    Player(val),
                    radius,
                    scale,
                    alpha,
                    animating=is_animating,
                )

    def _draw_stone(
        self,
        surface: pygame.Surface,
        row: int,
        col: int,
        player: Player,
        radius: int,
        scale: float,
        alpha: float = 1.0,
        *,
        animating: bool = False,
    ) -> None:
        """Vẽ một quân cờ tròn tại ô (row, col)."""
        cx, cy = self._cell_center(row, col)
        cx_i, cy_i = int(cx), int(cy)
        r = max(2, int(radius * max(0.0, min(scale, 1.0))))
        color = Theme.STONE_X if player is Player.X else Theme.STONE_O
        border = Theme.STONE_O if player is Player.X else (200, 202, 210)

        if not animating:
            pygame.draw.circle(surface, Theme.STONE_SHADOW, (cx_i + 2, cy_i + 2), r)
            pygame.draw.circle(surface, color, (cx_i, cy_i), r)
            pygame.draw.circle(surface, border, (cx_i, cy_i), r, width=max(1, r // 10))
            return

        pad = 4
        stone_surf = pygame.Surface((r * 2 + pad * 2, r * 2 + pad * 2), pygame.SRCALPHA)
        center = (r + pad, r + pad)
        a = int(255 * alpha)
        shadow_a = min(a, 90)
        pygame.draw.circle(stone_surf, (*Theme.STONE_SHADOW, shadow_a), center, r)
        pygame.draw.circle(stone_surf, (*color, a), center, r)
        pygame.draw.circle(
            stone_surf, (*border, a), center, r, width=max(1, r // 10)
        )
        surface.blit(stone_surf, (cx_i - r - pad, cy_i - r - pad))

    def _draw_threat_hints(self, surface: pygame.Surface) -> None:
        """Vẽ viền cảnh báo ô thắng/chặn/chặn 2 đầu."""
        hint = self._threat_hint
        if hint is None or self.env.done:
            return

        def _ring(move: Move, color: tuple[int, int, int], width: int = 3) -> None:
            row, col = move
            cx, cy = self._cell_center(row, col)
            size = int(self.cell * 0.88)
            ring = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(ring, (*color, 200), (0, 0, size, size), width=width, border_radius=8)
            surface.blit(ring, (int(cx - size / 2), int(cy - size / 2)))

        for move in hint.win_moves:
            _ring(move, Theme.HIGHLIGHT_WIN, 3)
        for move in hint.block_moves:
            _ring(move, Theme.DANGER, 3)
        for move in hint.double_end_blocks:
            if move not in hint.win_moves and move not in hint.block_moves:
                _ring(move, Theme.DANGER, 4)
        for move in hint.threat_stones:
            row, col = move
            if self.env.board[row, col] == Player.EMPTY:
                continue
            cx, cy = self._cell_center(row, col)
            r = self._stone_radius()
            glow = pygame.Surface((r * 2 + 20, r * 2 + 20), pygame.SRCALPHA)
            alpha = pulse_alpha(self.anim_time, WIN_PULSE_SPEED, lo=140, hi=255)
            pygame.draw.circle(glow, (*Theme.DANGER, alpha), (r + 10, r + 10), r + 8)
            surface.blit(glow, (cx - r - 10, cy - r - 10))
            pygame.draw.circle(
                surface, Theme.DANGER, (int(cx), int(cy)), r + 3, width=3
            )

    def _draw_highlights(self, surface: pygame.Surface) -> None:
        """Highlight đường thắng (pulse) và nước đi gần nhất."""
        if self.env.done and self.env.winning_line:
            self._draw_winning_line(surface)
            return

        if self.env.last_move is not None and self.place_anim is None:
            row, col = self.env.last_move
            cx, cy = self._cell_center(row, col)
            alpha = pulse_alpha(self.anim_time, LAST_MOVE_PULSE_SPEED, lo=100, hi=255)
            size = int(self.cell * 0.92)
            ring = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(
                ring, (*Theme.HIGHLIGHT_LAST, alpha),
                (0, 0, size, size), width=3, border_radius=8,
            )
            surface.blit(ring, (int(cx - size / 2), int(cy - size / 2)))

    def _draw_winning_line(self, surface: pygame.Surface) -> None:
        """Vẽ đường nối 5 quân thắng với hiệu ứng pulse."""
        if len(self.env.winning_line) < 2:
            return

        pulse = pulse_alpha(self.anim_time, WIN_PULSE_SPEED, lo=120, hi=255)
        pts = [self._cell_center(r, c) for r, c in self.env.winning_line]
        int_pts = [(int(x), int(y)) for x, y in pts]

        if len(int_pts) >= 2:
            glow_surf = pygame.Surface((self.board_px + 20, self.board_px + 20), pygame.SRCALPHA)
            ox, oy = self.origin
            shifted = [(x - ox + 10, y - oy + 10) for x, y in int_pts]
            pygame.draw.lines(glow_surf, (*Theme.HIGHLIGHT_WIN, pulse), False, shifted, 8)
            surface.blit(glow_surf, (ox - 10, oy - 10))
            pygame.draw.lines(surface, Theme.HIGHLIGHT_WIN, False, int_pts, 4)

        r = self._stone_radius()
        for (row, col) in self.env.winning_line:
            cx, cy = self._cell_center(row, col)
            glow = pygame.Surface((r * 2 + 16, r * 2 + 16), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*Theme.HIGHLIGHT_WIN, pulse // 2), (r + 8, r + 8), r + 6)
            surface.blit(glow, (cx - r - 8, cy - r - 8))
            pygame.draw.circle(
                surface, Theme.HIGHLIGHT_WIN, (int(cx), int(cy)), r + 4, width=4
            )

    def _draw_sidebar(self, surface: pygame.Surface) -> None:
        """Vẽ HUD bên phải: badge người chơi, lượt, thanh AI, win prob."""
        x0 = WINDOW_WIDTH - SIDEBAR_WIDTH
        panel = pygame.Rect(x0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(surface, Theme.SURFACE, panel)
        pygame.draw.line(surface, Theme.SURFACE_LIGHT, (x0, 0), (x0, WINDOW_HEIGHT), 2)

        cx = x0 + SIDEBAR_WIDTH // 2
        render_text(surface, "TRẬN ĐẤU", 26, Theme.TEXT_PRIMARY, center=(cx, 44), bold=True)
        render_text(
            surface, self.app.settings.mode.value, 17, Theme.TEXT_MUTED, center=(cx, 74)
        )

        active = None if self.env.done else self.env.current_player
        draw_player_badge(surface, (cx - 40, 120), Player.X, active=(active is Player.X))
        draw_player_badge(surface, (cx + 40, 120), Player.O, active=(active is Player.O))

        turn_label = "Kết thúc" if self.env.done else "Đang chơi"
        render_text(surface, turn_label, 22, Theme.TEXT_MUTED, center=(cx, 168))

        who = self._current_player_kind()
        render_text(surface, who, 17, Theme.ACCENT, center=(cx, 196))

        render_text(
            surface,
            f"Nước đi: {self.env.move_count}",
            15,
            Theme.TEXT_MUTED,
            center=(cx, 222),
        )

        self._draw_undo_redo_buttons(surface, x0, 530)
        self._draw_end_result_panel(surface, x0, 590)
        self._draw_think_bar(surface, x0, 250)
        self._draw_winprob_bar(surface, x0, 340)
        self._draw_ai_info(surface, x0, 430)

        hint = "R: Chơi lại  ·  ESC: Menu"
        if self.env.done and not self._end_overlay_dismissed:
            hint = "Space/Click: Xem bàn  ·  " + hint
        elif self._is_pva() and not self.env.done:
            hint = "U: Quay lại  ·  Y: Làm lại  ·  " + hint
        render_text(
            surface, hint, 14, Theme.TEXT_MUTED,
            center=(cx, WINDOW_HEIGHT - 28),
        )

    def _draw_undo_redo_buttons(self, surface: pygame.Surface, x0: int, y: int) -> None:
        """Nút Quay lại / Làm lại — chỉ hiện trong chế độ Người vs AI."""
        self._btn_undo = None
        self._btn_redo = None
        if not self._is_pva():
            return

        margin = 24
        btn_w = (SIDEBAR_WIDTH - margin * 2 - 10) // 2
        self._btn_undo = pygame.Rect(x0 + margin, y, btn_w, 44)
        self._btn_redo = pygame.Rect(self._btn_undo.right + 10, y, btn_w, 44)

        draw_button(
            surface,
            self._btn_undo,
            "Quay lại",
            hovered=self._can_undo() and self._btn_undo.collidepoint(self._mouse_pos),
            primary=False,
        )
        if not self._can_undo():
            render_text(
                surface, "Quay lại", 16, Theme.TEXT_MUTED, center=self._btn_undo.center
            )

        draw_button(
            surface,
            self._btn_redo,
            "Làm lại",
            hovered=self._can_redo() and self._btn_redo.collidepoint(self._mouse_pos),
            primary=False,
        )
        if not self._can_redo():
            render_text(
                surface, "Làm lại", 16, Theme.TEXT_MUTED, center=self._btn_redo.center
            )

    def _draw_end_result_panel(self, surface: pygame.Surface, x0: int, y: int) -> None:
        """Banner kết quả nhỏ khi pop-up đã ẩn (xem bàn cờ)."""
        self._btn_show_overlay = None
        self._btn_replay = None
        self._btn_menu = None

        if not self.env.done or not self._end_overlay_dismissed:
            return

        margin = 24
        box = pygame.Rect(x0 + margin, y, SIDEBAR_WIDTH - margin * 2, 130)
        draw_round_rect(surface, box, Theme.SURFACE_LIGHT, radius=10)

        if self.env.is_draw:
            title, color = "HÒA", Theme.ACCENT_WARM
        else:
            assert self.env.winner is not None
            title = f"{self.env.winner.name} thắng"
            color = Theme.HIGHLIGHT_WIN

        render_text(
            surface, title, 20, color, center=(box.centerx, box.top + 28), bold=True
        )
        render_text(
            surface,
            f"{self.env.move_count} nước",
            14,
            Theme.TEXT_MUTED,
            center=(box.centerx, box.top + 52),
        )

        self._btn_show_overlay = pygame.Rect(0, 0, box.width - 20, 36)
        self._btn_show_overlay.center = (box.centerx, box.top + 82)
        draw_button(
            surface,
            self._btn_show_overlay,
            "Hiện kết quả",
            hovered=self._btn_show_overlay.collidepoint(self._mouse_pos),
            primary=True,
        )

        self._btn_replay = pygame.Rect(box.left + 10, box.bottom - 42, (box.width - 30) // 2, 36)
        self._btn_menu = pygame.Rect(self._btn_replay.right + 10, box.bottom - 42, (box.width - 30) // 2, 36)
        draw_button(
            surface, self._btn_replay, "Chơi lại",
            hovered=self._btn_replay.collidepoint(self._mouse_pos), primary=False,
        )
        draw_button(
            surface, self._btn_menu, "Menu",
            hovered=self._btn_menu.collidepoint(self._mouse_pos), primary=False,
        )

    def _current_player_kind(self) -> str:
        """Mô tả người chơi / AI đang đi."""
        if self.env.done:
            if self._end_overlay_dismissed:
                return "Đang xem bàn cờ"
            return ""
        agent = self.agents[self.env.current_player]
        if agent is None:
            return f"Lượt {self.env.current_player.name} — Người chơi"
        return f"Lượt {self.env.current_player.name} — {agent.name}"

    def _draw_think_bar(self, surface: pygame.Surface, x0: int, y: int) -> None:
        """Thanh thời gian suy nghĩ AI — chạy indeterminate khi đang tính."""
        margin = 24
        bar = pygame.Rect(x0 + margin, y, SIDEBAR_WIDTH - margin * 2, 18)

        if self.ai_thinking:
            cycle = 1.8
            phase = (self.ai_think_elapsed % cycle) / cycle
            ratio = 0.25 + 0.5 * abs(phase * 2 - 1)
            value = f"{self.ai_think_elapsed:.1f}s"
            label = "AI đang suy nghĩ..."
        else:
            ratio = 0.0
            value = ""
            label = "AI sẵn sàng"

        draw_progress_bar(
            surface, bar, ratio, Theme.ACCENT_WARM,
            label=label, value_text=value if value else None,
        )

    def _win_prob_label(self) -> str:
        """Nhãn thanh Win Probability theo người chơi hiện tại."""
        if self.env.done:
            return "Win Probability"
        return f"Win Probability ({self.env.current_player.name})"

    def _draw_winprob_bar(self, surface: pygame.Surface, x0: int, y: int) -> None:
        """Thanh xác suất thắng (DQN hoặc heuristic)."""
        margin = 24
        bar = pygame.Rect(x0 + margin, y, SIDEBAR_WIDTH - margin * 2, 22)
        label = self._win_prob_label()

        uses_dqn = self.win_probability_source == "dqn"

        if self.win_probability is None:
            draw_progress_bar(surface, bar, 0.0, Theme.HIGHLIGHT_WIN, label=label)
            render_text(
                surface,
                "Không có dữ liệu",
                14,
                Theme.TEXT_MUTED,
                center=bar.center,
            )
            return

        prob = max(0.0, min(1.0, self.win_probability))
        color = Theme.HIGHLIGHT_WIN if prob >= 0.5 else Theme.ACCENT_WARM
        suffix = "" if uses_dqn else " ~"
        draw_progress_bar(
            surface, bar, prob, color,
            label=label, value_text=f"{prob:.0%}{suffix}",
        )
        if not uses_dqn:
            render_text(
                surface,
                "(heuristic)",
                11,
                Theme.TEXT_MUTED,
                topleft=(bar.x, bar.bottom + 2),
            )
            return

    def _draw_ai_info(self, surface: pygame.Surface, x0: int, y: int) -> None:
        """Khối thông tin loại AI và độ khó."""
        margin = 24
        box = pygame.Rect(x0 + margin, y, SIDEBAR_WIDTH - margin * 2, 88)
        draw_round_rect(surface, box, Theme.SURFACE_LIGHT, radius=10)
        render_text(
            surface, "Cấu hình AI", 15, Theme.TEXT_MUTED,
            center=(box.centerx, box.top + 18),
        )
        render_text(
            surface,
            self.app.settings.ai_type.value,
            18,
            Theme.TEXT_PRIMARY,
            center=(box.centerx, box.top + 44),
        )
        render_text(
            surface,
            f"Độ khó: {self.app.settings.difficulty.name.title()}",
            15,
            Theme.TEXT_MUTED,
            center=(box.centerx, box.top + 68),
        )

    def _draw_end_overlay(self, surface: pygame.Surface) -> None:
        """Pop-up kết thúc với fade + scale animation và nút hover."""
        alpha = self.end_overlay_anim.overlay_alpha
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, alpha))
        surface.blit(overlay, (0, 0))

        cx = (WINDOW_WIDTH - SIDEBAR_WIDTH) // 2
        cy = WINDOW_HEIGHT // 2
        scale = self.end_overlay_anim.scale
        box_w, box_h = int(420 * scale), int(300 * scale)
        box = pygame.Rect(0, 0, box_w, box_h)
        box.center = (cx, cy)
        self._end_overlay_box = box
        draw_round_rect(surface, box, Theme.SURFACE_LIGHT, radius=18)

        if self.env.is_draw:
            title, color = "HÒA!", Theme.ACCENT_WARM
            subtitle = "Hai bên cân tài cân sức"
        else:
            assert self.env.winner is not None
            title = f"{self.env.winner.name} THẮNG!"
            color = Theme.HIGHLIGHT_WIN
            subtitle = f"Chiến thắng sau {self.env.move_count} nước"

        render_text(surface, title, int(44 * scale), color, center=(cx, box.top + 56), bold=True)
        render_text(surface, subtitle, 18, Theme.TEXT_MUTED, center=(cx, box.top + 100))
        if self._last_learn_message:
            render_text(
                surface,
                self._last_learn_message,
                15,
                Theme.ACCENT,
                center=(cx, box.top + 128),
            )
        render_text(
            surface,
            "Click ngoài pop-up hoặc Space để xem bàn cờ",
            14,
            Theme.TEXT_MUTED,
            center=(cx, box.top + 156 if self._last_learn_message else 132),
        )

        self._btn_dismiss_overlay = pygame.Rect(0, 0, 220, 44)
        self._btn_dismiss_overlay.center = (cx, box.bottom - 118)
        self._btn_replay = pygame.Rect(0, 0, 170, 54)
        self._btn_replay.center = (cx - 95, box.bottom - 62)
        self._btn_menu = pygame.Rect(0, 0, 170, 54)
        self._btn_menu.center = (cx + 95, box.bottom - 62)

        draw_button(
            surface,
            self._btn_dismiss_overlay,
            "Xem bàn cờ",
            hovered=self._btn_dismiss_overlay.collidepoint(self._mouse_pos),
            primary=True,
        )
        draw_button(
            surface, self._btn_replay, "Chơi lại",
            hovered=self._btn_replay.collidepoint(self._mouse_pos), primary=False,
        )
        draw_button(
            surface, self._btn_menu, "Về Menu",
            hovered=self._btn_menu.collidepoint(self._mouse_pos), primary=False,
        )
