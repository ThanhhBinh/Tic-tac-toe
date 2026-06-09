#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test animation và widget UI polish."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ui.animations import OverlayAnimation, PlaceAnimation, ease_out_elastic, pulse_alpha
from config import Player


def test_place_animation_hoan_thanh() -> None:
    """PlaceAnimation kết thúc sau duration."""
    anim = PlaceAnimation(move=(3, 3), player=Player.X, duration=0.1)
    anim.elapsed = 0.2
    assert anim.finished
    assert anim.alpha() <= 1.0
    assert 0.6 <= anim.scale() <= 1.0


def test_place_animation_scale_khong_vuot_1() -> None:
    """Scale luôn trong [0.6, 1.0] — tránh quân phình to vượt ô."""
    anim = PlaceAnimation(move=(0, 0), player=Player.O, duration=0.22)
    for t in [0.0, 0.05, 0.3, 0.63, 0.9, 1.0]:
        anim.elapsed = t * anim.duration
        assert 0.6 <= anim.scale() <= 1.0


def test_overlay_animation_scale() -> None:
    """OverlayAnimation tăng scale theo thời gian."""
    ov = OverlayAnimation(duration=0.5)
    ov.tick(0.5)
    assert ov.scale >= 0.85


def test_pulse_alpha_trong_khoang() -> None:
    """pulse_alpha luôn nằm trong [lo, hi]."""
    a = pulse_alpha(1.0, speed=3.0, lo=80, hi=255)
    assert 80 <= a <= 255


def test_ease_out_elastic() -> None:
    """Elastic easing đạt ~1 tại t=1."""
    assert abs(ease_out_elastic(1.0) - 1.0) < 0.01
