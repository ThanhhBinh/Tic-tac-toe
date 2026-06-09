#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Màn hình Settings: chọn chế độ chơi, loại AI, độ khó và kích thước bàn cờ.

Mọi lựa chọn được ghi trực tiếp vào `app.settings` (GameSettings) để màn hình
chơi đọc khi khởi tạo ván.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pygame
import pygame_menu

from config import (
    BOARD_SIZES,
    HYBRID_DEPTH_BY_DIFFICULTY,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    AIType,
    Difficulty,
    GameMode,
)
from ui.menu_theme import build_menu_theme
from ui.screens.base import BaseScreen

if TYPE_CHECKING:  # pragma: no cover
    from ui.app import App


class SettingsScreen(BaseScreen):
    """Màn hình cấu hình ván chơi."""

    def __init__(self, app: "App") -> None:
        """Dựng các bộ chọn (selector) tương ứng với từng tuỳ chọn cấu hình."""
        super().__init__(app)
        self.menu: pygame_menu.Menu = pygame_menu.Menu(
            title="Cài đặt",
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            theme=build_menu_theme(),
        )
        self._build_widgets()

    def _build_widgets(self) -> None:
        """Thêm các selector và nút vào menu."""
        s = self.app.settings

        # Chế độ chơi.
        mode_items = [(m.value, m) for m in GameMode]
        self.menu.add.selector(
            "Chế độ:  ",
            mode_items,
            default=self._index_of([m for _, m in mode_items], s.mode),
            onchange=self._on_mode,
        )

        # Loại AI.
        ai_items = [(a.value, a) for a in AIType]
        self.menu.add.selector(
            "AI:  ",
            ai_items,
            default=self._index_of([a for _, a in ai_items], s.ai_type),
            onchange=self._on_ai,
        )

        # Độ khó — hiển thị depth Minimax và depth Hybrid (Hybrid max = 3).
        diff_items = [
            (
                f"{d.name.title()} (Minimax {int(d)} / Hybrid {HYBRID_DEPTH_BY_DIFFICULTY[d]})",
                d,
            )
            for d in Difficulty
        ]
        self.menu.add.selector(
            "Độ khó:  ",
            diff_items,
            default=self._index_of([d for _, d in diff_items], s.difficulty),
            onchange=self._on_difficulty,
        )

        # Kích thước bàn cờ.
        size_items = [(f"{n} x {n}", n) for n in BOARD_SIZES]
        self.menu.add.selector(
            "Bàn cờ:  ",
            size_items,
            default=self._index_of([n for _, n in size_items], s.board_size),
            onchange=self._on_size,
        )

        self.menu.add.toggle_switch(
            "Chặn 2 đầu",
            default=s.double_end_block_rule,
            onchange=self._on_double_end,
        )
        self.menu.add.toggle_switch(
            "Cảnh báo thắng",
            default=s.threat_warnings,
            onchange=self._on_threat_warnings,
        )
        self.menu.add.toggle_switch(
            "AI tấn công",
            default=s.ai_aggressive,
            onchange=self._on_ai_aggressive,
        )
        self.menu.add.toggle_switch(
            "AI đi trước",
            default=s.ai_first,
            onchange=self._on_ai_first,
        )

        self.menu.add.vertical_margin(24)
        self.menu.add.button("Bắt đầu chơi", self._on_start)
        self.menu.add.button("Quay lại", self._on_back)

    @staticmethod
    def _index_of(values: list[Any], target: Any) -> int:
        """Tìm chỉ số của ``target`` trong ``values`` (mặc định 0 nếu không có)."""
        try:
            return values.index(target)
        except ValueError:
            return 0

    # --- Callback cập nhật settings ---
    def _on_mode(self, item: tuple, value: GameMode) -> None:
        """Cập nhật chế độ chơi."""
        self.app.settings.mode = value

    def _on_ai(self, item: tuple, value: AIType) -> None:
        """Cập nhật loại AI."""
        self.app.settings.ai_type = value

    def _on_difficulty(self, item: tuple, value: Difficulty) -> None:
        """Cập nhật độ khó (độ sâu Minimax)."""
        self.app.settings.difficulty = value

    def _on_size(self, item: tuple, value: int) -> None:
        """Cập nhật kích thước bàn cờ."""
        self.app.settings.board_size = value

    def _on_double_end(self, value: bool) -> None:
        """Bật/tắt luật chặn 2 đầu."""
        self.app.settings.double_end_block_rule = value

    def _on_threat_warnings(self, value: bool) -> None:
        """Bật/tắt cảnh báo sắp thắng."""
        self.app.settings.threat_warnings = value

    def _on_ai_aggressive(self, value: bool) -> None:
        """Bật/tắt AI tấn công chủ động."""
        self.app.settings.ai_aggressive = value

    def _on_ai_first(self, value: bool) -> None:
        """Bật/tắt AI đi trước (chỉ áp dụng chế độ Người vs AI)."""
        self.app.settings.ai_first = value

    def _on_start(self) -> None:
        """Vào màn hình chơi với cấu hình hiện tại."""
        from ui.app import SCREEN_GAME

        self.app.go_to(SCREEN_GAME)

    def _on_back(self) -> None:
        """Quay lại Main Menu."""
        from ui.app import SCREEN_MENU

        self.app.go_to(SCREEN_MENU)

    # --- Vòng đời ---
    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Chuyển sự kiện cho pygame-menu."""
        if self.menu.is_enabled():
            self.menu.update(events)

    def update(self, dt: float) -> None:
        """Không có animation."""

    def draw(self, surface: pygame.Surface) -> None:
        """Vẽ menu cài đặt."""
        self.menu.draw(surface)
