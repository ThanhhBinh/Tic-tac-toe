#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vòng đời ứng dụng & máy trạng thái (state machine) chuyển màn hình.

`App` sở hữu vòng lặp game chính của pygame và điều phối việc chuyển đổi giữa
các màn hình (Main Menu, Settings, Game, End). Mỗi màn hình tự xử lý sự kiện,
cập nhật và vẽ; App chỉ gọi các hàm đó và quản lý chuyển cảnh.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from config import (
    DEFAULT_AI_AGGRESSIVE,
    DEFAULT_AI_FIRST,
    DEFAULT_BOARD_SIZE,
    DEFAULT_DOUBLE_END_BLOCK_RULE,
    DEFAULT_THREAT_WARNINGS,
    FPS,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    AIType,
    Difficulty,
    GameMode,
    TacticalConfig,
    Theme,
)


@dataclass
class GameSettings:
    """Cấu hình ván chơi do người dùng chọn ở màn hình Settings.

    Đối tượng này được truyền từ Settings sang Game để khởi tạo môi trường và
    các agent phù hợp.

    Attributes:
        mode: Chế độ chơi (PvP / PvA / AvA).
        ai_type: Loại AI sử dụng khi có AI tham gia.
        difficulty: Độ khó, ánh xạ sang độ sâu Minimax.
        board_size: Kích thước bàn cờ.
    """

    mode: GameMode = GameMode.PVA
    ai_type: AIType = AIType.MINIMAX
    difficulty: Difficulty = Difficulty.MEDIUM
    board_size: int = DEFAULT_BOARD_SIZE
    double_end_block_rule: bool = DEFAULT_DOUBLE_END_BLOCK_RULE
    threat_warnings: bool = DEFAULT_THREAT_WARNINGS
    ai_aggressive: bool = DEFAULT_AI_AGGRESSIVE
    ai_first: bool = DEFAULT_AI_FIRST

    @property
    def tactical_config(self) -> TacticalConfig:
        """Luật chiến thuật cho agent AI."""
        return TacticalConfig(
            double_end_block_rule=self.double_end_block_rule,
            aggressive=self.ai_aggressive,
            threat_warnings=self.threat_warnings,
        )


# Tên định danh các màn hình để chuyển cảnh.
SCREEN_MENU: str = "menu"
SCREEN_SETTINGS: str = "settings"
SCREEN_GAME: str = "game"
SCREEN_GUIDE: str = "guide"


class App:
    """Lớp ứng dụng chính: khởi tạo pygame và chạy vòng lặp state machine."""

    def __init__(self) -> None:
        """Khởi tạo cửa sổ, đồng hồ FPS và trạng thái dùng chung."""
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        self.surface: pygame.Surface = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.running: bool = True

        # Cấu hình ván chơi dùng chung giữa các màn hình.
        self.settings: GameSettings = GameSettings()

        # Khởi tạo (lazy) các màn hình để tránh phụ thuộc vòng khi import.
        self._screens: dict[str, object] = {}
        self.current_name: str = SCREEN_MENU
        self._build_screens()

    def _build_screens(self) -> None:
        """Tạo các instance màn hình. Import cục bộ để tránh circular import."""
        from ui.screens.game_screen import GameScreen
        from ui.screens.guide_screen import GuideScreen
        from ui.screens.main_menu import MainMenuScreen
        from ui.screens.settings_menu import SettingsScreen

        self._screens = {
            SCREEN_MENU: MainMenuScreen(self),
            SCREEN_SETTINGS: SettingsScreen(self),
            SCREEN_GAME: GameScreen(self),
            SCREEN_GUIDE: GuideScreen(self),
        }

    @property
    def current(self) -> object:
        """Màn hình đang hiển thị."""
        return self._screens[self.current_name]

    def go_to(self, name: str) -> None:
        """Chuyển sang màn hình khác và gọi hook on_enter của nó.

        Args:
            name: Tên định danh màn hình đích.
        """
        self.current_name = name
        on_enter = getattr(self.current, "on_enter", None)
        if callable(on_enter):
            on_enter()

    def quit(self) -> None:
        """Yêu cầu thoát vòng lặp chính."""
        self.running = False

    def run(self) -> None:
        """Chạy vòng lặp game chính cho tới khi người dùng thoát."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # delta-time tính bằng giây
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False

            screen = self.current
            screen.handle_events(events)  # type: ignore[attr-defined]
            screen.update(dt)  # type: ignore[attr-defined]

            self.surface.fill(Theme.BACKGROUND)
            screen.draw(self.surface)  # type: ignore[attr-defined]
            pygame.display.flip()

        pygame.quit()
