#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lớp cơ sở (interface) cho mọi màn hình giao diện.

Mỗi màn hình kế thừa `BaseScreen` và hiện thực 3 phương thức vòng đời:
xử lý sự kiện, cập nhật theo thời gian, và vẽ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:  # pragma: no cover - chỉ phục vụ type hint, tránh import vòng
    from ui.app import App


class BaseScreen(ABC):
    """Khung chung cho các màn hình. Giữ tham chiếu tới App để chuyển cảnh."""

    def __init__(self, app: "App") -> None:
        """Lưu tham chiếu tới ứng dụng chủ.

        Args:
            app: Đối tượng App điều phối các màn hình.
        """
        self.app = app

    def on_enter(self) -> None:
        """Hook được gọi mỗi khi màn hình trở thành màn hình hiện tại.

        Mặc định không làm gì; màn hình con ghi đè nếu cần khởi tạo lại trạng
        thái (vd: tạo ván mới khi vào GameScreen).
        """

    @abstractmethod
    def handle_events(self, events: list[pygame.event.Event]) -> None:
        """Xử lý danh sách sự kiện pygame trong khung hình hiện tại."""

    @abstractmethod
    def update(self, dt: float) -> None:
        """Cập nhật trạng thái theo delta-time (giây)."""

    @abstractmethod
    def draw(self, surface: pygame.Surface) -> None:
        """Vẽ màn hình lên surface đích."""
