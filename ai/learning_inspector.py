#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Truy vết và so sánh dữ liệu / model DQN đã học (phục vụ UI dashboard)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ai.board_encoder import action_to_move, encode_board, legal_action_mask
from ai.benchmark import BENCHMARK_SCENARIOS, _make_env, _run_agent_on_scenario

try:
    from ai.dqn_agent import DQNAgent
except ModuleNotFoundError:
    DQNAgent = None  # type: ignore[assignment,misc]
from ai.replay_buffer import Transition
from config import (
    DQN_BUFFER_CAPACITY,
    ONLINE_LEARN_ENABLED,
    ONLINE_LEARN_GRADIENT_STEPS,
    ONLINE_LEARN_GRADIENT_STEPS_HIGH,
    ONLINE_LEARN_GRADIENT_STEPS_LOW,
    ONLINE_LEARN_MIN_SAMPLES,
    ONLINE_SAVE_GATE_ENABLED,
    Difficulty,
    Player,
    TacticalConfig,
    dqn_model_backup_path,
    dqn_model_path,
    learn_log_path,
)
from core.caro_env import CaroEnv
from core.constants import Move

SourceTag = Literal["online_pva", "offline_train", "unknown"]


def _file_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path.name),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def state_to_board(state: np.ndarray, perspective: Player) -> list[list[int]]:
    """Giải mã tensor (3,H,W) về ma trận bàn cờ int."""
    own = state[0] > 0.5
    opp = state[1] > 0.5
    size = state.shape[1]
    board = np.zeros((size, size), dtype=np.int8)
    board[own] = int(perspective)
    board[opp] = int(perspective.opponent)
    return board.tolist()


def transition_to_dict(
    transition: Transition,
    board_size: int,
    *,
    index: int,
    source: SourceTag = "unknown",
    perspective: Player = Player.X,
) -> dict[str, Any]:
    """Chuyển Transition sang dict hiển thị trên UI."""
    row, col = action_to_move(transition.action, board_size)
    return {
        "index": index,
        "source": source,
        "perspective": perspective.name,
        "board": state_to_board(transition.state, perspective),
        "move": [row, col],
        "action": int(transition.action),
        "reward": round(float(transition.reward), 4),
        "done": bool(transition.done),
        "board_after": state_to_board(transition.next_state, perspective),
    }


def backup_model_if_exists(board_size: int) -> bool:
    """Sao lưu checkpoint hiện tại trước khi cập nhật trọng số."""
    src = dqn_model_path(board_size)
    if not src.exists():
        return False
    shutil.copy2(src, dqn_model_backup_path(board_size))
    return True


def append_learn_log(board_size: int, record: dict[str, Any]) -> None:
    """Ghi thêm một sự kiện học vào file JSONL."""
    path = learn_log_path(board_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    record.setdefault("timestamp", datetime.now(tz=timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_learn_log(board_size: int, limit: int = 50) -> list[dict[str, Any]]:
    """Đọc các sự kiện học gần nhất (mới nhất trước)."""
    path = learn_log_path(board_size)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    records.reverse()
    return records


def _get_online_learner(board_size: int) -> Any:
    """Import lazy để tránh vòng phụ thuộc với online_learner."""
    from ai.online_learner import OnlineLearner

    return OnlineLearner.for_board(board_size)


def get_buffer_samples(board_size: int, limit: int = 24) -> list[dict[str, Any]]:
    """Lấy mẫu transition gần nhất từ buffer online."""
    learner = _get_online_learner(board_size)
    data = list(learner.trainer.buffer._data)  # noqa: SLF001 — chỉ đọc cho dashboard
    samples: list[dict[str, Any]] = []
    start = max(0, len(data) - limit)
    for idx, transition in enumerate(data[start:], start=start):
        perspective = Player.X if transition.state[0].sum() >= transition.state[1].sum() else Player.O
        samples.append(
            transition_to_dict(
                transition,
                board_size,
                index=idx,
                source="online_pva",
                perspective=perspective,
            )
        )
    samples.reverse()
    return samples


def get_learning_status(board_size: int) -> dict[str, Any]:
    """Tổng quan trạng thái học DQN cho dashboard."""
    learner = _get_online_learner(board_size)
    buffer_len = len(learner.trainer.buffer)
    model = _file_meta(dqn_model_path(board_size))
    backup = _file_meta(dqn_model_backup_path(board_size))
    log_entries = read_learn_log(board_size, limit=1)
    last_event = log_entries[0] if log_entries else None

    return {
        "board_size": board_size,
        "online_learn_enabled": ONLINE_LEARN_ENABLED,
        "buffer_size": buffer_len,
        "buffer_capacity": DQN_BUFFER_CAPACITY,
        "min_samples_to_train": ONLINE_LEARN_MIN_SAMPLES,
        "gradient_steps_per_game": ONLINE_LEARN_GRADIENT_STEPS,
        "gradient_steps_high": ONLINE_LEARN_GRADIENT_STEPS_HIGH,
        "gradient_steps_low": ONLINE_LEARN_GRADIENT_STEPS_LOW,
        "save_gate_enabled": ONLINE_SAVE_GATE_ENABLED,
        "can_train_now": buffer_len >= ONLINE_LEARN_MIN_SAMPLES,
        "model": model,
        "model_backup": backup,
        "has_backup": backup is not None,
        "learn_log_count": len(learn_log_path(board_size).read_text(encoding="utf-8").splitlines())
        if learn_log_path(board_size).exists()
        else 0,
        "last_event": last_event,
    }


def _dqn_q_at_move(agent: DQNAgent, env: CaroEnv, move: Move) -> float:
    player = env.current_player
    q = agent._predict_q_numpy(env, player)
    action = move[0] * env.size + move[1]
    if 0 <= action < q.size:
        return round(float(q[action]), 4)
    return 0.0


def _dqn_greedy_move(agent: DQNAgent, env: CaroEnv) -> Move:
    player = env.current_player
    q = agent._predict_q_numpy(env, player)
    mask = legal_action_mask(env.board)
    q[~mask] = -np.inf
    action = int(np.argmax(q))
    return action_to_move(action, env.size)


def _env_from_transition(transition: Transition, board_size: int) -> tuple[CaroEnv, Player, Move]:
    perspective = Player.X if transition.state[0].sum() >= transition.state[1].sum() else Player.O
    board = np.array(state_to_board(transition.state, perspective), dtype=np.int8)
    env = CaroEnv(size=board_size)
    env.board = board.copy()
    env.current_player = perspective
    move = action_to_move(transition.action, board_size)
    return env, perspective, move


def _compare_on_transitions(
    agent_before: DQNAgent | None,
    agent_after: DQNAgent,
    board_size: int,
    transitions: list[Transition],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, transition in enumerate(transitions):
        env, perspective, move = _env_from_transition(transition, board_size)
        after_q = _dqn_q_at_move(agent_after, env, move)
        after_greedy = _dqn_greedy_move(agent_after, env)
        row: dict[str, Any] = {
            "index": idx,
            "board": state_to_board(transition.state, perspective),
            "perspective": perspective.name,
            "learned_move": [move[0], move[1]],
            "reward": round(float(transition.reward), 4),
            "done": bool(transition.done),
            "after_q_at_move": after_q,
            "after_greedy_move": [after_greedy[0], after_greedy[1]],
            "after_would_repeat": after_greedy == move,
        }
        if agent_before is not None and agent_before._model_loaded:
            before_q = _dqn_q_at_move(agent_before, env, move)
            before_greedy = _dqn_greedy_move(agent_before, env)
            row.update(
                {
                    "before_q_at_move": before_q,
                    "before_greedy_move": [before_greedy[0], before_greedy[1]],
                    "before_would_repeat": before_greedy == move,
                    "q_delta": round(after_q - before_q, 4),
                    "move_changed": before_greedy != after_greedy,
                }
            )
        rows.append(row)
    return rows


def compare_dqn_before_after(
    board_size: int,
    difficulty: Difficulty = Difficulty.MEDIUM,
    tactical: TacticalConfig | None = None,
) -> dict[str, Any]:
    """So sánh model hiện tại vs bản backup trên benchmark + buffer."""
    cfg = tactical or TacticalConfig()
    current_path = dqn_model_path(board_size)
    backup_path = dqn_model_backup_path(board_size)

    if not current_path.exists():
        return {
            "has_current": False,
            "has_backup": backup_path.exists(),
            "message": "Chưa có model DQN — cần train trước (train.py hoặc auto_train).",
        }

    agent_after = DQNAgent(board_size=board_size, epsilon=0.0, model_path=current_path)
    agent_before: DQNAgent | None = None
    if backup_path.exists():
        agent_before = DQNAgent(board_size=board_size, epsilon=0.0, model_path=backup_path)

    scenario_rows: list[dict[str, Any]] = []
    before_tactical = 0
    after_tactical = 0
    tactical_total = 0
    improved = 0
    regressed = 0
    unchanged = 0

    for scenario in BENCHMARK_SCENARIOS:
        env, expected_moves = _make_env(scenario, board_size, cfg)
        after_result = _run_agent_on_scenario(
            agent_after, scenario, board_size, cfg, expected_moves
        )
        before_result: dict[str, Any] | None = None
        if agent_before is not None and agent_before._model_loaded:
            before_result = _run_agent_on_scenario(
                agent_before, scenario, board_size, cfg, expected_moves
            )

        if expected_moves is not None:
            tactical_total += 1
            if after_result.get("is_expected") is True:
                after_tactical += 1
            if before_result and before_result.get("is_expected") is True:
                before_tactical += 1

        verdict = "unchanged"
        if before_result is not None:
            b_ok = before_result.get("is_expected") is True
            a_ok = after_result.get("is_expected") is True
            b_move = tuple(before_result["move"])
            a_move = tuple(after_result["move"])
            if a_ok and not b_ok:
                verdict = "improved"
                improved += 1
            elif b_ok and not a_ok:
                verdict = "regressed"
                regressed += 1
            elif b_move != a_move:
                verdict = "changed"
                unchanged += 1
            else:
                unchanged += 1

        scenario_rows.append(
            {
                "id": scenario.id,
                "name": scenario.name,
                "category": scenario.category,
                "board": env.board.tolist(),
                "board_size": board_size,
                "expected_moves": [[m[0], m[1]] for m in expected_moves]
                if expected_moves
                else None,
                "before": before_result,
                "after": after_result,
                "verdict": verdict,
            }
        )

    learner = _get_online_learner(board_size)
    buffer_tail = list(learner.trainer.buffer._data)[-12:]  # noqa: SLF001
    transition_compare = _compare_on_transitions(
        agent_before if agent_before and agent_before._model_loaded else None,
        agent_after,
        board_size,
        buffer_tail,
    )

    before_score = round(before_tactical / tactical_total * 100, 1) if tactical_total else 0.0
    after_score = round(after_tactical / tactical_total * 100, 1) if tactical_total else 0.0

    return {
        "has_current": True,
        "has_backup": agent_before is not None and agent_before._model_loaded,
        "board_size": board_size,
        "difficulty": difficulty.name,
        "summary": {
            "before_tactical_correct": before_tactical,
            "after_tactical_correct": after_tactical,
            "tactical_total": tactical_total,
            "before_tactical_pct": before_score,
            "after_tactical_pct": after_score,
            "delta_tactical_pct": round(after_score - before_score, 1),
            "scenarios_improved": improved,
            "scenarios_regressed": regressed,
            "scenarios_unchanged_or_changed": unchanged,
            "is_better": after_score > before_score
            if agent_before and agent_before._model_loaded
            else None,
        },
        "scenarios": scenario_rows,
        "buffer_transitions": transition_compare,
        "headline": _build_compare_headline(
            agent_before is not None and agent_before._model_loaded,
            before_score,
            after_score,
            improved,
            regressed,
        ),
    }


def _build_compare_headline(
    has_backup: bool,
    before_pct: float,
    after_pct: float,
    improved: int,
    regressed: int,
) -> str:
    if not has_backup:
        return (
            "Chưa có bản backup — chơi ván PvA và thắng/thua AI để kích hoạt học online, "
            "lần học đầu sẽ tạo file .backup.pth để so sánh lần sau."
        )
    delta = after_pct - before_pct
    if delta > 0:
        return (
            f"Model mới tốt hơn trên benchmark chiến thuật: "
            f"{after_pct:.0f}% vs {before_pct:.0f}% (+{delta:.0f}%). "
            f"Cải thiện {improved} TH, tệ hơn {regressed} TH."
        )
    if delta < 0:
        return (
            f"Model mới yếu hơn một chút: {after_pct:.0f}% vs {before_pct:.0f}% "
            f"({delta:.0f}%). Có thể do overfit ván vừa học — cần thêm dữ liệu."
        )
    return (
        f"Điểm chiến thuật giữ nguyên ({after_pct:.0f}%). "
        f"{improved} TH cải thiện, {regressed} TH tệ hơn."
    )


def build_learn_log_record(
    board_size: int,
    outcome: str,
    transitions: list[Transition],
    gradient_steps: int,
    avg_loss: float,
    model_saved: bool,
    buffered_only: bool,
    buffer_size_after: int,
    *,
    quality: str = "medium",
    gate_reason: str = "",
    rolled_back: bool = False,
) -> dict[str, Any]:
    """Tạo bản ghi JSON cho một lần học online."""
    serialized = []
    for idx, t in enumerate(transitions):
        perspective = Player.X if t.state[0].sum() >= t.state[1].sum() else Player.O
        serialized.append(
            transition_to_dict(
                t,
                board_size,
                index=idx,
                source="online_pva",
                perspective=perspective,
            )
        )
    return {
        "board_size": board_size,
        "outcome": outcome,
        "ai_moves": len(transitions),
        "gradient_steps": gradient_steps,
        "avg_loss": round(avg_loss, 5),
        "model_saved": model_saved,
        "buffered_only": buffered_only,
        "buffer_size_after": buffer_size_after,
        "quality": quality,
        "gate_reason": gate_reason,
        "rolled_back": rolled_back,
        "transitions": serialized,
    }
