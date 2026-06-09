#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Màn hình Main Menu: Bắt đầu, Cài đặt, Hướng dẫn, Thoát."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
import pygame_menu

from config import WINDOW_HEIGHT, WINDOW_WIDTH, Theme
from ui.menu_theme import build_menu_theme
from ui.screens.base import BaseScreen
from ui.theme import render_text

if TYPE_CHECKING:  # pragma: no cover
    from ui.app import App


class MainMenuScreen(BaseScreen):
    """Màn hình menu chính của trò chơi."""

    def __init__(self, app: "App") -> None:
        """Khởi tạo menu chính với các nút điều hướng."""
        super().__init__(app)
        self.menu: pygame_menu.Menu = pygame_menu.Menu(
            title="CỜ CARO AI",
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            theme=build_menu_theme(),
            center_content=True,
        )
        self.menu.add.vertical_margin(20)
        self.menu.add.button("Bắt đầu chơi", self._on_start)
        self.menu.add.button("Cài đặt", self._on_settings)
        self.menu.add.button("Hướng dẫn", self._on_guide)
        self.menu.add.button("Thoát", self._on_quit)

    # --- Callback các nút ---
    def _on_start(self) -> None:
        """Chuyển sang màn hình chơi."""
        from ui.app import SCREEN_GAME

        self.app.go_to(SCREEN_GAME)

    def _on_settings(self) -> None:
        """Chuyển sang màn hình cài đặt."""
        from ui.app import SCREEN_SETTINGS

        self.app.go_to(SCREEN_SETTINGS)

    def _on_guide(self) -> None:
        """Chuyển sang màn hình hướng dẫn."""
        from ui.app import SCREEN_GUIDE

        self.app.go_to(SCREEN_GUIDE)

    def _on_quit(self) -> None:
        """Thoát ứng dụng."""
        self.app.quit()

    # --- Vòng đời màn hình ---
    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Chuyển sự kiện cho pygame-menu xử lý."""
        if self.menu.is_enabled():
            self.menu.update(events)

    def update(self, dt: float) -> None:
        """Main menu không có animation theo thời gian."""

    def draw(self, surface: pygame.Surface) -> None:
        """Vẽ menu và một dòng phụ đề mô tả dự án."""
        self.menu.draw(surface)
        render_text(
            surface,
            "Minimax + Alpha-Beta  •  Deep Q-Network",
            18,
            Theme.TEXT_MUTED,
            center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 36),
        )
