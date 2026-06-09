#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tiện ích giao diện: nạp/cache font và một số helper vẽ dùng chung.

Giữ phần "trang trí" tách khỏi logic màn hình để dễ thay đổi chủ đề.
"""

from __future__ import annotations

from pathlib import Path

import pygame

# Các đường dẫn font hệ thống hỗ trợ tiếng Việt (ưu tiên macOS, sau đó Linux).
# pygame_menu cần một file .ttf cụ thể để hiển thị đúng dấu tiếng Việt.
_VN_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def get_vietnamese_font_path() -> str | None:
    """Tìm đường dẫn tới một font .ttf hỗ trợ tiếng Việt trên máy.

    Returns:
        Đường dẫn font đầu tiên tồn tại, hoặc None nếu không tìm thấy
        (khi đó nơi gọi nên dùng font mặc định).
    """
    for path in _VN_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None

# Bộ nhớ đệm font theo (kích thước, đậm) để tránh tạo lại mỗi khung hình.
_FONT_CACHE: dict[tuple[int, bool], pygame.font.Font] = {}

# Danh sách font ưu tiên (hỗ trợ tiếng Việt có dấu). pygame.font.SysFont sẽ
# tự chọn font đầu tiên có trên hệ thống.
_PREFERRED_FONTS: str = "arial,helveticaneue,helvetica,dejavusans,sans"


def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Lấy font theo kích thước (có cache).

    Args:
        size: Cỡ chữ (pixel).
        bold: True nếu cần chữ đậm.

    Returns:
        Đối tượng pygame.font.Font sẵn sàng để render.
    """
    key = (size, bold)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = pygame.font.SysFont(_PREFERRED_FONTS, size, bold=bold)
    return _FONT_CACHE[key]


def render_text(
    surface: pygame.Surface,
    text: str,
    size: int,
    color: tuple[int, int, int],
    center: tuple[int, int] | None = None,
    topleft: tuple[int, int] | None = None,
    bold: bool = False,
) -> pygame.Rect:
    """Vẽ một dòng chữ lên surface và trả về vùng chữ.

    Args:
        surface: Bề mặt đích để vẽ.
        text: Nội dung chữ.
        size: Cỡ chữ.
        color: Màu chữ RGB.
        center: Toạ độ tâm (ưu tiên nếu được cung cấp).
        topleft: Toạ độ góc trên-trái (dùng nếu không có center).
        bold: Chữ đậm hay không.

    Returns:
        pygame.Rect bao quanh chữ vừa vẽ.
    """
    font = get_font(size, bold)
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center is not None:
        rect.center = center
    elif topleft is not None:
        rect.topleft = topleft
    surface.blit(img, rect)
    return rect


def draw_round_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    radius: int = 12,
    width: int = 0,
) -> None:
    """Vẽ hình chữ nhật bo góc (wrapper gọn cho pygame.draw.rect)."""
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)
