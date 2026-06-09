#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test phiên chơi web (không cần trình duyệt)."""

from __future__ import annotations

from config import AIType, Difficulty, GameMode, Player
from web.session import GameSession, SessionSettings


def test_web_session_pva_move_undo() -> None:
    """PvA: đặt quân, AI phản hồi, undo lùi cặp nước."""
    session = GameSession(
        SessionSettings(
            mode=GameMode.PVA,
            ai_type=AIType.MINIMAX,
            difficulty=Difficulty.EASY,
            board_size=10,
        )
    )
    assert session.to_dict()["move_count"] == 0

    session.play_move(5, 5)
    assert session.env.move_count >= 2

    before = session.env.move_count
    session.undo()
    assert session.env.move_count == 0
    assert session.can_redo()

    session.redo()
    assert session.env.move_count == 1
    assert session.env.board[5, 5] == Player.X


def test_web_session_to_dict_keys() -> None:
    """API serialize đủ trường cho frontend."""
    session = GameSession(SessionSettings(board_size=10))
    data = session.to_dict()
    for key in (
        "session_id", "board", "board_size", "current_player",
        "done", "can_undo", "can_redo", "is_human_turn", "is_ava", "settings",
    ):
        assert key in data


def test_web_session_ava_step_mot_nuoc() -> None:
    """AvA: mỗi ava-step chỉ tiến một nước, không treo cả ván."""
    session = GameSession(
        SessionSettings(
            mode=GameMode.AVA,
            ai_type=AIType.MINIMAX,
            difficulty=Difficulty.EASY,
            board_size=10,
        )
    )
    assert session.to_dict()["is_ava"] is True
    assert session.to_dict()["move_count"] == 0
    assert session.to_dict()["is_human_turn"] is False

    data = session.step_ava()
    assert data["move_count"] == 1
    assert not data["done"]
    assert data["is_human_turn"] is False

    data2 = session.step_ava()
    assert data2["move_count"] == 2


def test_web_session_ai_first() -> None:
    """PvA + AI đi trước: AI (X) đánh nước mở đầu, người chơi là O."""
    session = GameSession(
        SessionSettings(
            mode=GameMode.PVA,
            ai_type=AIType.MINIMAX,
            difficulty=Difficulty.EASY,
            board_size=10,
            ai_first=True,
        )
    )
    data = session.to_dict()
    assert data["human_player"] == "O"
    assert data["move_count"] == 1
    assert data["is_human_turn"] is True
    assert data["settings"]["ai_first"] is True
    assert session.env.board[data["last_move"][0]][data["last_move"][1]] == Player.X
