#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tạo theme dùng chung cho các menu xây bằng `pygame-menu`.

Đồng bộ màu sắc của menu với bảng màu `Theme` trong config để giao diện nhất
quán (modern minimal, nền tối, điểm nhấn xanh).
"""

from __future__ import annotations

import pygame_menu

from config import Theme
from ui.theme import get_vietnamese_font_path


def build_menu_theme() -> pygame_menu.themes.Theme:
    """Dựng đối tượng Theme cho pygame-menu khớp với bảng màu dự án.

    Returns:
        Theme đã cấu hình màu nền, màu chữ, font (hỗ trợ tiếng Việt nếu có).
    """
    font_path = get_vietnamese_font_path()
    title_font = font_path or pygame_menu.font.FONT_OPEN_SANS_BOLD
    widget_font = font_path or pygame_menu.font.FONT_OPEN_SANS

    theme = pygame_menu.themes.THEME_DARK.copy()
    theme.background_color = Theme.SURFACE
    theme.title_background_color = Theme.SURFACE_LIGHT
    theme.title_font = title_font
    theme.title_font_color = Theme.TEXT_PRIMARY
    theme.title_font_size = 40
    theme.widget_font = widget_font
    theme.widget_font_color = Theme.TEXT_PRIMARY
    theme.widget_font_size = 28
    theme.selection_color = Theme.ACCENT
    theme.widget_padding = 14
    return theme
