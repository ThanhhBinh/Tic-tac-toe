#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Điểm vào giao diện web — mở trên Chrome tại http://127.0.0.1:8000

Chạy:
    python web_main.py
"""

from __future__ import annotations

import uvicorn

from web.server import app


def main() -> None:
    """Khởi động server FastAPI."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
