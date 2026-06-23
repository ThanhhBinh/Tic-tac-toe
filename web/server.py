#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI server phục vụ giao diện web Cờ Caro."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import AIType, Difficulty, GameMode, BOARD_SIZES, create_caro_env
from ai.benchmark import run_benchmark
from web.benchmark_store import BenchmarkCache
from web.session import GameSession, SessionSettings, SessionStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
store = SessionStore()
benchmark_cache = BenchmarkCache()

app = FastAPI(
    title="AI Cờ Caro",
    description="Giao diện web — Minimax + DQN + Hybrid",
    version="1.0.0",
)


class NewGameRequest(BaseModel):
    """Body tạo ván mới."""

    mode: str = Field(default="Player vs AI", description="Chế độ chơi")
    ai_type: str = Field(default="Hybrid (Minimax + DQN)")
    difficulty: str = Field(default="MEDIUM")
    board_size: int = Field(default=15)
    double_end_block_rule: bool = Field(default=True, description="Luật chặn 2 đầu")
    threat_warnings: bool = Field(default=True, description="Cảnh báo sắp thắng")
    ai_aggressive: bool = Field(default=True, description="AI ưu tiên tấn công")
    ai_first: bool = Field(default=False, description="AI đi trước (PvA)")


class MoveRequest(BaseModel):
    """Body đặt quân."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)


class CompareRequest(BaseModel):
    """Body chạy benchmark so sánh 3 thuật toán."""

    difficulty: str = Field(default="MEDIUM")
    board_size: int = Field(default=15)
    double_end_block_rule: bool = Field(default=True)
    ai_aggressive: bool = Field(default=True)
    scenario_set: str = Field(default="basic")
    force: bool = Field(
        default=False,
        description="True = bỏ cache DB và chạy lại benchmark (mặc định dùng bản đã lưu).",
    )


def _parse_mode(value: str) -> GameMode:
    """Chuyển chuỗi mode sang enum."""
    for mode in GameMode:
        if mode.value == value or mode.name == value:
            return mode
    raise HTTPException(status_code=400, detail=f"Chế độ không hợp lệ: {value}")


def _parse_ai_type(value: str) -> AIType:
    """Chuyển chuỗi loại AI sang enum."""
    for ai_type in AIType:
        if ai_type.value == value or ai_type.name == value:
            return ai_type
    raise HTTPException(status_code=400, detail=f"Loại AI không hợp lệ: {value}")


def _parse_difficulty(value: str) -> Difficulty:
    """Chuyển chuỗi độ khó sang enum."""
    key = value.upper()
    try:
        return Difficulty[key]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Độ khó không hợp lệ: {value}") from exc


def _parse_board_size(value: int) -> int:
    """Kiểm tra kích thước bàn cờ hợp lệ."""
    if value not in BOARD_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Kích thước bàn không hợp lệ: {value}. Chọn một trong {list(BOARD_SIZES)}.",
        )
    return value


@app.get("/")
async def index() -> FileResponse:
    """Trang chủ — giao diện game."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/compare")
async def compare_page() -> FileResponse:
    """Trang so sánh Minimax / DQN / Hybrid."""
    return FileResponse(STATIC_DIR / "compare.html")


@app.get("/learn")
async def learn_page() -> FileResponse:
    """Dashboard truy vết dữ liệu học DQN."""
    return FileResponse(STATIC_DIR / "learn.html")


@app.get("/api/options")
async def get_options() -> dict[str, Any]:
    """Danh sách tuỳ chọn cho form cài đặt."""
    return {
        "modes": [m.value for m in GameMode],
        "ai_types": [a.value for a in AIType],
        "difficulties": [d.name for d in Difficulty],
        "board_sizes": list(BOARD_SIZES),
    }


@app.post("/api/games")
async def create_game(body: NewGameRequest) -> dict[str, Any]:
    """Tạo ván chơi mới."""
    settings = SessionSettings(
        mode=_parse_mode(body.mode),
        ai_type=_parse_ai_type(body.ai_type),
        difficulty=_parse_difficulty(body.difficulty),
        board_size=_parse_board_size(body.board_size),
        double_end_block_rule=body.double_end_block_rule,
        threat_warnings=body.threat_warnings,
        ai_aggressive=body.ai_aggressive,
        ai_first=body.ai_first,
    )
    session = store.create(settings)
    return session.to_dict()


@app.get("/api/games/{session_id}")
async def get_game(session_id: str) -> dict[str, Any]:
    """Lấy trạng thái ván hiện tại."""
    try:
        return store.get(session_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/games/{session_id}/move")
async def play_move(session_id: str, body: MoveRequest) -> dict[str, Any]:
    """Người chơi đặt quân."""
    try:
        session = store.get(session_id)
        return session.play_move(body.row, body.col)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/games/{session_id}/undo")
async def undo_move(session_id: str) -> dict[str, Any]:
    """Quay lại nước trước."""
    try:
        return store.get(session_id).undo()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/games/{session_id}/redo")
async def redo_move(session_id: str) -> dict[str, Any]:
    """Làm lại nước đã quay lại."""
    try:
        return store.get(session_id).redo()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/games/{session_id}/ava-step")
async def ava_step(session_id: str) -> dict[str, Any]:
    """Tiến một lượt AI vs AI (demo)."""
    try:
        return store.get(session_id).step_ava()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/games/{session_id}")
async def delete_game(session_id: str) -> dict[str, str]:
    """Xoá phiên (giải phóng bộ nhớ)."""
    store.delete(session_id)
    return {"status": "ok"}


def _compare_cache_key(body: CompareRequest) -> str:
    return BenchmarkCache.make_key(
        body.scenario_set,
        body.difficulty,
        body.board_size,
        body.double_end_block_rule,
        body.ai_aggressive,
    )


@app.get("/api/compare/result")
async def get_compare_result(
    scenario_set: str = "basic",
    difficulty: str = "MEDIUM",
    board_size: int = 15,
    double_end_block_rule: bool = True,
    ai_aggressive: bool = True,
) -> dict[str, Any]:
    """Lấy kết quả benchmark đã lưu trong DB (404 nếu chưa chạy)."""
    key = BenchmarkCache.make_key(
        scenario_set,
        difficulty,
        _parse_board_size(board_size),
        double_end_block_rule,
        ai_aggressive,
    )
    cached = benchmark_cache.get(key)
    if cached is None:
        raise HTTPException(status_code=404, detail="Chưa có kết quả benchmark cho cấu hình này.")
    return cached


@app.post("/api/compare/run")
async def run_compare(body: CompareRequest) -> dict[str, Any]:
    """Chạy benchmark hoặc trả bản đã lưu trong DB (mặc định không chạy lại)."""
    import time

    from config import TacticalConfig

    try:
        difficulty = _parse_difficulty(body.difficulty)
    except HTTPException:
        raise

    board_size = _parse_board_size(body.board_size)
    cache_key = _compare_cache_key(body)

    if not body.force:
        cached = benchmark_cache.get(cache_key)
        if cached is not None:
            return cached

    tactical = TacticalConfig(
        double_end_block_rule=body.double_end_block_rule,
        aggressive=body.ai_aggressive,
        threat_warnings=True,
    )
    started = time.perf_counter()
    result = run_benchmark(
        difficulty=difficulty,
        board_size=board_size,
        tactical=tactical,
        scenario_set=body.scenario_set,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    benchmark_cache.save(
        cache_key,
        scenario_set=body.scenario_set,
        difficulty=body.difficulty,
        board_size=board_size,
        double_end_block_rule=body.double_end_block_rule,
        ai_aggressive=body.ai_aggressive,
        result=result,
        run_elapsed_ms=elapsed_ms,
    )
    result["from_cache"] = False
    result["cache_key"] = cache_key
    result["run_elapsed_ms"] = round(elapsed_ms, 1)
    return result


class LearnCompareRequest(BaseModel):
    """Body so sánh model DQN trước / sau học."""

    board_size: int = Field(default=15)
    difficulty: str = Field(default="MEDIUM")
    double_end_block_rule: bool = Field(default=True)
    ai_aggressive: bool = Field(default=True)


@app.get("/api/learn/status")
async def learn_status(board_size: int = 15) -> dict[str, Any]:
    """Tổng quan buffer, checkpoint và lần học gần nhất."""
    from ai.learning_inspector import get_learning_status

    return get_learning_status(_parse_board_size(board_size))


@app.get("/api/learn/buffer")
async def learn_buffer(board_size: int = 15, limit: int = 24) -> dict[str, Any]:
    """Mẫu transition gần nhất trong replay buffer online."""
    from ai.learning_inspector import get_buffer_samples

    size = _parse_board_size(board_size)
    return {
        "board_size": size,
        "samples": get_buffer_samples(size, limit=min(max(limit, 1), 100)),
    }


@app.get("/api/learn/history")
async def learn_history(board_size: int = 15, limit: int = 30) -> dict[str, Any]:
    """Nhật ký các lần học online."""
    from ai.learning_inspector import read_learn_log

    size = _parse_board_size(board_size)
    return {
        "board_size": size,
        "events": read_learn_log(size, limit=min(max(limit, 1), 200)),
    }


@app.post("/api/learn/compare")
async def learn_compare(body: LearnCompareRequest) -> dict[str, Any]:
    """So sánh model hiện tại vs bản backup trên benchmark + buffer."""
    from ai.learning_inspector import compare_dqn_before_after
    from config import TacticalConfig

    tactical = TacticalConfig(
        double_end_block_rule=body.double_end_block_rule,
        aggressive=body.ai_aggressive,
        threat_warnings=True,
    )
    return compare_dqn_before_after(
        board_size=_parse_board_size(body.board_size),
        difficulty=_parse_difficulty(body.difficulty),
        tactical=tactical,
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
