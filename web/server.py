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

from config import AIType, Difficulty, GameMode
from web.session import GameSession, SessionSettings, SessionStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
store = SessionStore()

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
    board_size: int = Field(default=15, ge=10, le=15)
    double_end_block_rule: bool = Field(default=True, description="Luật chặn 2 đầu")
    threat_warnings: bool = Field(default=True, description="Cảnh báo sắp thắng")
    ai_aggressive: bool = Field(default=True, description="AI ưu tiên tấn công")
    ai_first: bool = Field(default=False, description="AI đi trước (PvA)")


class MoveRequest(BaseModel):
    """Body đặt quân."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)


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


@app.get("/")
async def index() -> FileResponse:
    """Trang chủ — giao diện game."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
async def get_options() -> dict[str, Any]:
    """Danh sách tuỳ chọn cho form cài đặt."""
    return {
        "modes": [m.value for m in GameMode],
        "ai_types": [a.value for a in AIType],
        "difficulties": [d.name for d in Difficulty],
        "board_sizes": [10, 15],
    }


@app.post("/api/games")
async def create_game(body: NewGameRequest) -> dict[str, Any]:
    """Tạo ván chơi mới."""
    settings = SessionSettings(
        mode=_parse_mode(body.mode),
        ai_type=_parse_ai_type(body.ai_type),
        difficulty=_parse_difficulty(body.difficulty),
        board_size=body.board_size,
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


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
