#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Khởi động môi trường dev — FE (static) + BE (FastAPI) trong một lệnh.

Chạy:
    python dev.py
    make dev
"""

from __future__ import annotations

import threading
import time
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def _open_browser() -> None:
    """Mở Chrome/trình duyệt mặc định sau khi server sẵn sàng."""
    time.sleep(1.0)
    webbrowser.open(URL)


def main() -> None:
    """Chạy server dev với reload tự động khi sửa code Python hoặc static."""
    print(f"Đang khởi động FE + BE tại {URL} (Ctrl+C để dừng)")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(
        "web.server:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_dirs=["web", "core", "ai"],
        reload_includes=["config.py"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
