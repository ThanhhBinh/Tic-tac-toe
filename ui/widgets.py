#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Widget UI tái sử dụng: thanh tiến trình, badge người chơi, nút bấm.

Tách khỏi màn hình chơi để ``game_screen`` gọn và dễ kiểm thử riêng phần vẽ.
"""

from __future__ import annotations

import pygame

from config import Player, Theme
from ui.theme import draw_round_rect, render_text


def draw_progress_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    ratio: float,
    fill_color: tuple[int, int, int],
    *,
    bg_color: tuple[int, int, int] = Theme.SURFACE_LIGHT,
    radius: int = 8,
    label: str | None = None,
    value_text: str | None = None,
) -> None:
    """Vẽ thanh tiến trình bo góc kèm nhãn tuỳ chọn.

    Args:
        surface: Bề mặt đích.
        rect: Vùng thanh bar.
        ratio: Tiến trình trong [0, 1].
        fill_color: Màu phần đã hoàn thành.
        bg_color: Màu nền bar.
        radius: Bo góc.
        label: Nhãn phía trên bar (tuỳ chọn).
        value_text: Chữ hiển thị giữa bar (tuỳ chọn).
    """
    if label:
        render_text(surface, label, 16, Theme.TEXT_MUTED, topleft=(rect.x, rect.y - 24))

    draw_round_rect(surface, rect, bg_color, radius=radius)
    clamped = max(0.0, min(1.0, ratio))
    if clamped > 0.0:
        fill = pygame.Rect(rect.x, rect.y, max(4, int(rect.width * clamped)), rect.height)
        draw_round_rect(surface, fill, fill_color, radius=radius)

    if value_text:
        render_text(surface, value_text, 14, Theme.TEXT_PRIMARY, center=rect.center)


def draw_player_badge(
    surface: pygame.Surface,
    center: tuple[int, int],
    player: Player,
    *,
    active: bool = False,
    radius: int = 22,
) -> None:
    """Vẽ badge tròn biểu thị quân X hoặc O.

    Args:
        surface: Bề mặt đích.
        center: Tâm badge.
        player: X hoặc O.
        active: True nếu đang là lượt của người chơi này (viền sáng).
        radius: Bán kính badge.
    """
    color = Theme.STONE_X if player is Player.X else Theme.STONE_O
    cx, cy = center
    pygame.draw.circle(surface, color, center, radius)
    border_color = Theme.ACCENT if active else Theme.TEXT_MUTED
    pygame.draw.circle(surface, border_color, center, radius, width=3 if active else 2)
    text_color = Theme.STONE_O if player is Player.X else Theme.STONE_X
    render_text(surface, player.name, 20, text_color, center=center, bold=True)


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    *,
    hovered: bool = False,
    primary: bool = True,
) -> None:
    """Vẽ nút bấm bo góc với trạng thái hover.

    Args:
        surface: Bề mặt đích.
        rect: Vùng nút.
        label: Nội dung chữ.
        hovered: Chuột đang nằm trên nút.
        primary: True = nút chính (màu accent).
    """
    if primary:
        base = Theme.ACCENT if not hovered else (120, 185, 255)
    else:
        base = Theme.SURFACE if not hovered else Theme.SURFACE_LIGHT
    draw_round_rect(surface, rect, base, radius=12)
    if hovered:
        pygame.draw.rect(surface, Theme.ACCENT, rect, width=2, border_radius=12)
    render_text(surface, label, 24, Theme.TEXT_PRIMARY, center=rect.center)
