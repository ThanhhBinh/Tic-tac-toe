#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Màn hình Hướng dẫn: mô tả luật chơi và cách điều khiển."""

from __future__ import annotations

import pygame

from config import WINDOW_WIDTH, Theme
from ui.screens.base import BaseScreen
from ui.theme import render_text

# Nội dung hướng dẫn hiển thị từng dòng.
_GUIDE_LINES: tuple[str, ...] = (
    "MỤC TIÊU: Tạo 5 quân cùng màu liên tiếp theo hàng ngang,",
    "dọc hoặc chéo để giành chiến thắng.",
    "",
    "ĐIỀU KHIỂN:",
    "  •  Nhấp chuột trái vào ô trống để đặt quân.",
    "  •  Phím R: chơi lại ván mới.",
    "  •  Phím ESC: quay về Menu chính.",
    "",
    "CHẾ ĐỘ CHƠI (chọn ở phần Cài đặt):",
    "  •  Player vs Player — hai người chơi cùng máy.",
    "  •  Player vs AI — đấu với máy (Minimax / DQN / Hybrid).",
    "  •  AI vs AI — xem hai AI tự đấu.",
    "",
    "Nhấn ESC hoặc nhấp chuột để quay lại.",
)


class GuideScreen(BaseScreen):
    """Màn hình hiển thị hướng dẫn chơi."""

    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """ESC hoặc click chuột để quay về Menu chính."""
        from ui.app import SCREEN_MENU

        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.app.go_to(SCREEN_MENU)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.app.go_to(SCREEN_MENU)

    def update(self, dt: float) -> None:
        """Màn hình tĩnh, không cần cập nhật."""

    def draw(self, surface: pygame.Surface) -> None:
        """Vẽ tiêu đề và các dòng hướng dẫn."""
        render_text(
            surface,
            "HƯỚNG DẪN CHƠI",
            44,
            Theme.ACCENT,
            center=(WINDOW_WIDTH // 2, 90),
            bold=True,
        )
        y = 180
        for line in _GUIDE_LINES:
            render_text(surface, line, 24, Theme.TEXT_PRIMARY, topleft=(120, y))
            y += 38
