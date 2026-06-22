#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test learning inspector dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai.board_encoder import encode_board
from ai.learning_inspector import (
    append_learn_log,
    build_learn_log_record,
    get_learning_status,
    read_learn_log,
    state_to_board,
    transition_to_dict,
)
from ai.replay_buffer import Transition
from config import Player
from core.caro_env import CaroEnv


def test_state_to_board_roundtrip() -> None:
    """Giải mã tensor state khớp bàn cờ gốc."""
    env = CaroEnv(size=10)
    env.reset()
    env.step((5, 5))
    env.step((4, 4))
    state = encode_board(env.board, Player.O)
    board = np.array(state_to_board(state, Player.O), dtype=np.int8)
    assert np.array_equal(board, env.board)


def test_transition_to_dict_has_board_and_move() -> None:
    """Transition serialize đủ trường cho UI."""
    env = CaroEnv(size=10)
    env.reset()
    before = env.clone()
    move = (3, 3)
    env.step(move)
    t = Transition(
        state=encode_board(before.board, Player.X),
        action=move[0] * 10 + move[1],
        reward=-0.005,
        next_state=encode_board(env.board, Player.O),
        done=False,
    )
    data = transition_to_dict(t, 10, index=0, perspective=Player.X)
    assert data["move"] == [3, 3]
    assert data["reward"] == -0.005
    assert len(data["board"]) == 10


def test_learn_log_append_and_read(monkeypatch, tmp_path: Path) -> None:
    """Ghi và đọc nhật ký học JSONL."""
    log_file = tmp_path / "learn_log_10.jsonl"
    monkeypatch.setattr("ai.learning_inspector.learn_log_path", lambda _: log_file)

    append_learn_log(10, {"outcome": "ai_loss", "ai_moves": 2})
    append_learn_log(10, {"outcome": "ai_win", "ai_moves": 1})

    records = read_learn_log(10, limit=10)
    assert len(records) == 2
    assert records[0]["outcome"] == "ai_win"


def test_build_learn_log_record() -> None:
    """Bản ghi log chứa transitions serializable."""
    env = CaroEnv(size=10)
    env.reset()
    before = env.clone()
    env.step((2, 2))
    t = Transition(
        state=encode_board(before.board, Player.O),
        action=22,
        reward=-1.0,
        next_state=encode_board(env.board, Player.O),
        done=True,
    )
    record = build_learn_log_record(
        10, "ai_loss", [t], 32, 0.12, True, False, 5
    )
    assert record["ai_moves"] == 1
    assert record["transitions"][0]["reward"] == -1.0
    json.dumps(record)


def test_get_learning_status_smoke() -> None:
    """API status trả các trường cần cho dashboard."""
    status = get_learning_status(10)
    assert "buffer_size" in status
    assert "has_backup" in status
    assert status["board_size"] == 10
