#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test cho khung UI: chạy headless để bắt lỗi runtime khi vẽ/cập nhật.

Dùng driver giả lập của SDL (dummy) để không cần màn hình thật. Cho hai AI tự
đánh trên bàn nhỏ tới khi kết thúc, qua đó kiểm tra toàn bộ luồng vẽ/cập nhật
của GameScreen và các overlay không phát sinh ngoại lệ.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from config import AIType, Difficulty, GameMode  # noqa: E402
from ui.app import SCREEN_GAME, App  # noqa: E402


def test_game_screen_chay_headless_den_khi_ket_thuc() -> None:
    """AI vs AI trên bàn 10x10 phải chạy tới khi có kết quả mà không lỗi."""
    app = App()
    app.settings.mode = GameMode.AVA
    app.settings.board_size = 10
    app.settings.ai_type = AIType.MINIMAX
    app.settings.difficulty = Difficulty.EASY
    app.go_to(SCREEN_GAME)
    screen = app.current

    # AI chạy thread nền — cần đủ vòng lặp để thread kịp hoàn thành.
    for _ in range(15_000):
        screen.handle_events([])  # type: ignore[attr-defined]
        screen.update(0.05)  # type: ignore[attr-defined]
        screen.draw(app.surface)  # type: ignore[attr-defined]
        if screen.env.done:  # type: ignore[attr-defined]
            break

    assert screen.env.done  # type: ignore[attr-defined]


def test_chuyen_man_hinh_menu_settings() -> None:
    """Chuyển cảnh giữa các màn hình chính không phát sinh lỗi."""
    from ui.app import SCREEN_MENU, SCREEN_SETTINGS

    app = App()
    for name in (SCREEN_SETTINGS, SCREEN_MENU, SCREEN_GAME):
        app.go_to(name)
        app.current.handle_events([])  # type: ignore[attr-defined]
        app.current.update(0.016)  # type: ignore[attr-defined]
        app.current.draw(app.surface)  # type: ignore[attr-defined]
