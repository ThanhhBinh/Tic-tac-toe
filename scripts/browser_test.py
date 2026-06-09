#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm tra giao diện web trên Chromium (Playwright) — thay thế khi MCP timeout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parent.parent / "web" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

console_errors: list[str] = []
results: dict[str, str | bool | int] = {}


def shot(page, name: str) -> None:
    """Chụp màn hình viewport."""
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    results[f"screenshot_{name}"] = str(path)


def main() -> int:
    """Chạy luồng test E2E cơ bản."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        # 1. Trang setup
        page.goto(BASE, wait_until="networkidle", timeout=15000)
        shot(page, "01_setup")
        assert page.locator("h1").inner_text() == "AI Cờ Caro"
        results["setup_ok"] = True

        # 2. Tạo ván Minimax EASY (nhanh)
        page.select_option("#mode", "Player vs AI")
        page.select_option("#ai_type", "Minimax")
        page.select_option("#difficulty", "EASY")
        page.select_option("#board_size", "10")
        page.click('button[type="submit"]')
        page.wait_for_selector("#game-screen.active", timeout=10000)
        page.wait_for_selector(".cell", state="attached", timeout=10000)
        page.wait_for_function(
            """() => {
                const c = document.querySelector('.cell');
                return c && c.getBoundingClientRect().width > 4;
            }""",
            timeout=10000,
        )
        shot(page, "02_game_start")

        cells = page.locator(".cell.playable")
        count_playable = cells.count()
        results["playable_cells"] = count_playable
        assert count_playable > 0, "Không có ô playable"

        # 3. Đặt quân giữa bàn (10x10 → ~5,5)
        page.locator('.cell[data-row="4"][data-col="4"]').click()
        page.wait_for_function(
            "() => document.getElementById('move-count').textContent !== '0'",
            timeout=60000,
        )
        page.wait_for_function(
            "() => document.getElementById('board-loading').classList.contains('hidden')",
            timeout=60000,
        )
        move_count = int(page.locator("#move-count").inner_text())
        results["move_count_after_play"] = move_count
        assert move_count >= 2, f"AI chưa phản hồi (move_count={move_count})"
        shot(page, "03_after_move")

        # 4. Undo
        page.click("#btn-undo")
        page.wait_for_function(
            "() => document.getElementById('move-count').textContent === '0'",
            timeout=5000,
        )
        results["undo_ok"] = True
        shot(page, "04_after_undo")

        # 5. Redo
        page.click("#btn-redo")
        page.wait_for_function(
            "() => parseInt(document.getElementById('move-count').textContent) >= 1",
            timeout=5000,
        )
        results["redo_ok"] = True
        shot(page, "05_after_redo")

        # 6. Chơi thêm vài nước ngẫu nhiên tới khi xong hoặc max 8 nước
        for _ in range(8):
            if page.locator("#end-modal:not(.hidden)").count() > 0:
                break
            playable = page.locator(".cell.playable")
            if playable.count() == 0:
                break
            playable.first.click()
            page.wait_for_function(
                "() => document.getElementById('board-loading').classList.contains('hidden')",
                timeout=60000,
            )

        if page.locator("#end-modal:not(.hidden)").count() > 0:
            results["game_ended"] = True
            shot(page, "06_win_modal")
            page.click("#btn-view-board")
            page.wait_for_selector("#end-banner:not(.hidden)", timeout=3000)
            results["dismiss_modal_ok"] = True
            shot(page, "07_board_visible")
        else:
            results["game_ended"] = False

        results["console_errors"] = console_errors
        browser.close()

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if console_errors:
        print("CONSOLE ERRORS:", console_errors, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
