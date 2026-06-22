#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test học online từ ván Người vs AI."""

from __future__ import annotations

from ai.dqn_agent import DQNAgent
from ai.factory import create_agent
from ai.online_learner import (
    GameMoveRecorder,
    OnlineLearner,
    extract_dqn_agent,
    learn_from_pva_game,
    resolve_game_outcome,
)
from config import AIType, Difficulty, GameMode, Player
from core.caro_env import CaroEnv
from web.session import GameSession, SessionSettings


def test_extract_dqn_agent() -> None:
    """DQN và Hybrid trả DQNAgent; Minimax trả None."""
    dqn = create_agent(AIType.DQN, Difficulty.EASY, board_size=10)
    hybrid = create_agent(AIType.HYBRID, Difficulty.EASY, board_size=10)
    minimax = create_agent(AIType.MINIMAX, Difficulty.EASY, board_size=10)

    assert isinstance(extract_dqn_agent(dqn), DQNAgent)
    assert extract_dqn_agent(hybrid) is hybrid.dqn  # type: ignore[attr-defined]
    assert extract_dqn_agent(minimax) is None


def test_resolve_game_outcome_human_win() -> None:
    """Người thắng → AI thua."""
    assert resolve_game_outcome(Player.O, Player.O, False) == "ai_loss"
    assert resolve_game_outcome(Player.X, Player.O, False) == "ai_win"
    assert resolve_game_outcome(Player.X, None, True) == "draw"


def test_recorder_finalize_loss_sets_negative_reward() -> None:
    """Nước AI cuối nhận reward -1 khi người thắng."""
    env = CaroEnv(size=10)
    env.reset()
    recorder = GameMoveRecorder()

    before = env.clone()
    move = (5, 5)
    env.step(move)
    recorder.record_ai_move(before, move, Player.X, env)

    transitions = recorder.build_transitions("ai_loss")
    assert len(transitions) == 1
    assert transitions[-1].reward == -1.0
    assert transitions[-1].done is True


def test_assess_game_quality_and_draw() -> None:
    """Ván hòa ngắn bị skip; hòa dài được học."""
    from ai.online_learner import assess_game_quality
    from config import ONLINE_LEARN_DRAW_MIN_AI_MOVES

    assert assess_game_quality("draw", ONLINE_LEARN_DRAW_MIN_AI_MOVES - 1) == "skip"
    assert assess_game_quality("draw", ONLINE_LEARN_DRAW_MIN_AI_MOVES) == "high"


def test_resolve_game_outcome_draw() -> None:
    """Hòa trả draw (không None) để có thể học ván dài."""
    assert resolve_game_outcome(Player.X, None, True) == "draw"


def test_recorder_invalidated_after_undo() -> None:
    """Sau invalidate không còn transition để học."""
    recorder = GameMoveRecorder()
    env = CaroEnv(size=10)
    env.reset()
    before = env.clone()
    env.step((3, 3))
    recorder.record_ai_move(before, (3, 3), Player.X, env)
    recorder.invalidate()
    assert recorder.build_transitions("ai_loss") == []


def test_online_learner_pushes_to_buffer() -> None:
    """Học online đưa transition vào replay buffer."""
    learner = OnlineLearner(board_size=10)
    recorder = GameMoveRecorder()
    env = CaroEnv(size=10)
    env.reset()
    before = env.clone()
    env.step((4, 4))
    recorder.record_ai_move(before, (4, 4), Player.O, env)

    before_len = len(learner.trainer.buffer)
    result = learner.learn_from_game(recorder, "ai_loss")
    assert result is not None
    assert result.outcome == "ai_loss"
    assert len(learner.trainer.buffer) == before_len + 1


def test_learn_from_pva_game_minimax_skips() -> None:
    """Minimax không kích hoạt học online."""
    recorder = GameMoveRecorder()
    agent = create_agent(AIType.MINIMAX, Difficulty.EASY, board_size=10)
    result = learn_from_pva_game(
        recorder,
        10,
        Player.X,
        Player.X,
        False,
        agent,
    )
    assert result is None


def test_web_session_records_ai_moves_pva() -> None:
    """PvA với DQN: sau nước người, recorder ghi nước AI."""
    session = GameSession(
        SessionSettings(
            mode=GameMode.PVA,
            ai_type=AIType.DQN,
            difficulty=Difficulty.EASY,
            board_size=10,
        )
    )
    session.play_move(5, 5)
    assert session._move_recorder.ai_move_count >= 1


def test_web_session_undo_invalidates_learning() -> None:
    """Undo huỷ bản ghi — không học từ ván đã sửa."""
    session = GameSession(
        SessionSettings(
            mode=GameMode.PVA,
            ai_type=AIType.DQN,
            difficulty=Difficulty.EASY,
            board_size=10,
        )
    )
    session.play_move(5, 5)
    assert session._move_recorder.ai_move_count >= 1
    session.undo()
    assert session._move_recorder.build_transitions("ai_loss") == []
