#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Điểm vào chính: khởi chạy giao diện game Cờ Caro.

Chạy:
    python main.py
"""

from __future__ import annotations

from ui.app import App


def main() -> None:
    """Khởi tạo ứng dụng và chạy vòng lặp chính."""
    App().run()


if __name__ == "__main__":
    main()
