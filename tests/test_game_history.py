#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test undo/redo và ẩn pop-up kết thúc trên GameScreen."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from config import AIType, Difficulty, GameMode, Player  # noqa: E402
from core.caro_env import CaroEnv  # noqa: E402
from ui.app import SCREEN_GAME, App  # noqa: E402


def test_copy_state_from_khoi_phuc_ban() -> None:
    """copy_state_from phải khôi phục đúng trạng thái bàn cờ."""
    env = CaroEnv(size=10)
    env.reset()
    env.step((3, 3))
    snapshot = env.clone()
    env.step((4, 4))
    assert env.move_count == 2

    target = CaroEnv(size=10)
    target.reset()
    target.copy_state_from(snapshot)
    assert target.move_count == 1
    assert target.board[3, 3] == Player.X
    assert target.board[4, 4] == Player.EMPTY


def test_pva_undo_redo_lui_tien() -> None:
    """PvA: quay lại lùi cặp nước, làm lại khôi phục được."""
    app = App()
    app.settings.mode = GameMode.PVA
    app.settings.board_size = 10
    app.settings.ai_type = AIType.MINIMAX
    app.settings.difficulty = Difficulty.EASY
    app.go_to(SCREEN_GAME)
    screen = app.current

    screen._apply_move((5, 5))  # type: ignore[attr-defined]
    screen.place_anim = None  # type: ignore[attr-defined]
    screen._apply_move((5, 6))  # type: ignore[attr-defined]
    screen.place_anim = None  # type: ignore[attr-defined]

    assert screen.env.move_count == 2  # type: ignore[attr-defined]
    screen._undo()  # type: ignore[attr-defined]
    assert screen.env.move_count == 0  # type: ignore[attr-defined]

    screen._redo()  # type: ignore[attr-defined]
    assert screen.env.move_count == 1  # type: ignore[attr-defined]
    assert screen.env.board[5, 5] == Player.X  # type: ignore[attr-defined]


def test_an_popup_ket_thuc() -> None:
    """Pop-up kết thúc có thể ẩn để xem bàn cờ."""
    app = App()
    app.settings.mode = GameMode.PVP
    app.settings.board_size = 10
    app.go_to(SCREEN_GAME)
    screen = app.current

    moves = [
        (0, 0), (1, 0), (0, 1), (1, 1), (0, 2),
        (1, 2), (0, 3), (1, 3), (0, 4),
    ]
    for move in moves:
        screen._apply_move(move)  # type: ignore[attr-defined]
        screen.place_anim = None  # type: ignore[attr-defined]

    assert screen.env.done  # type: ignore[attr-defined]
    assert screen.env.winner is Player.X  # type: ignore[attr-defined]
    assert not screen._end_overlay_dismissed  # type: ignore[attr-defined]
    screen._dismiss_end_overlay()  # type: ignore[attr-defined]
    assert screen._end_overlay_dismissed  # type: ignore[attr-defined]
    screen._show_end_overlay()  # type: ignore[attr-defined]
    assert not screen._end_overlay_dismissed  # type: ignore[attr-defined]
