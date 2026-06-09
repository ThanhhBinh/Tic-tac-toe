#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiện ích animation cho giao diện: hàm easing và hiệu ứng đặt quân.

Tách riêng để màn hình chơi gọn gàng, dễ tái sử dụng hiệu ứng.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import PLACE_ANIM_DURATION, Player
from core.constants import Move


def ease_out_back(t: float) -> float:
    """Easing "out-back": vọt quá rồi nảy về — tạo cảm giác đặt quân nảy nhẹ.

    Args:
        t: Tiến trình trong [0, 1].

    Returns:
        Giá trị đã nội suy (có thể > 1 ở giữa do hiệu ứng nảy).
    """
    c1 = 1.70158
    c3 = c1 + 1.0
    t -= 1.0
    return 1.0 + c3 * t * t * t + c1 * t * t


def ease_out_back_clamped(t: float) -> float:
    """Biến thể out-back giới hạn trong [0, 1] — tránh quân phóng to vượt ô."""
    return max(0.0, min(ease_out_back(t), 1.0))


def ease_out_cubic(t: float) -> float:
    """Easing "out-cubic": chậm dần ở cuối, mượt cho fade.

    Args:
        t: Tiến trình trong [0, 1].

    Returns:
        Giá trị nội suy trong [0, 1].
    """
    f = 1.0 - t
    return 1.0 - f * f * f


def ease_out_elastic(t: float) -> float:
    """Easing elastic nhẹ cho pop-up kết thúc ván."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return math.pow(2.0, -10.0 * t) * math.sin((t - 0.075) * (2.0 * math.pi) / 0.3) + 1.0


def pulse_alpha(elapsed: float, speed: float, lo: int = 80, hi: int = 255) -> int:
    """Sinh alpha dao động sin cho hiệu ứng nhấp nháy.

    Args:
        elapsed: Thời gian tích luỹ (giây).
        speed: Tốc độ dao động.
        lo: Alpha tối thiểu.
        hi: Alpha tối đa.

    Returns:
        Giá trị alpha integer trong [lo, hi].
    """
    wave = (math.sin(elapsed * speed) + 1.0) * 0.5
    return int(lo + wave * (hi - lo))


@dataclass
class PlaceAnimation:
    """Trạng thái hiệu ứng khi một quân cờ được đặt xuống.

    Attributes:
        move: Vị trí quân vừa đặt.
        player: Người chơi sở hữu quân.
        elapsed: Thời gian đã trôi qua (giây).
        duration: Tổng thời lượng hiệu ứng (giây).
    """

    move: Move
    player: Player
    elapsed: float = 0.0
    duration: float = PLACE_ANIM_DURATION

    @property
    def progress(self) -> float:
        """Tiến trình chuẩn hoá trong [0, 1]."""
        if self.duration <= 0.0:
            return 1.0
        return min(self.elapsed / self.duration, 1.0)

    @property
    def finished(self) -> bool:
        """True nếu hiệu ứng đã kết thúc."""
        return self.elapsed >= self.duration

    def scale(self) -> float:
        """Hệ số phóng to của quân theo thời gian (nảy nhẹ rồi ổn định)."""
        # Bắt đầu ~60% kích thước, nảy nhẹ tới 100% — không vượt 1.0.
        return 0.6 + 0.4 * ease_out_back_clamped(self.progress)

    def alpha(self) -> float:
        """Độ mờ (0..1) tăng dần để tạo hiệu ứng fade-in."""
        return ease_out_cubic(self.progress)


@dataclass
class OverlayAnimation:
    """Hiệu ứng fade + scale cho pop-up kết thúc ván."""

    elapsed: float = 0.0
    duration: float = 0.35
    _finished: bool = field(default=False, repr=False)

    def tick(self, dt: float) -> None:
        """Cập nhật thời gian animation."""
        self.elapsed += dt
        if self.elapsed >= self.duration:
            self._finished = True

    @property
    def progress(self) -> float:
        """Tiến trình [0, 1]."""
        if self.duration <= 0:
            return 1.0
        return min(self.elapsed / self.duration, 1.0)

    @property
    def scale(self) -> float:
        """Scale pop-up (elastic nhẹ)."""
        return 0.85 + 0.15 * ease_out_elastic(self.progress)

    @property
    def overlay_alpha(self) -> int:
        """Alpha lớp phủ nền tối."""
        return int(150 * ease_out_cubic(self.progress))

    def reset(self) -> None:
        """Đặt lại animation (ván mới)."""
        self.elapsed = 0.0
        self._finished = False
