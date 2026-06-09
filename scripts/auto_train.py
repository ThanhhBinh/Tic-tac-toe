#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tự huấn luyện DQN nếu chưa có model hoặc khi được yêu cầu.

Chạy:
    python scripts/auto_train.py                  # bàn 15, train nếu thiếu model
    python scripts/auto_train.py --board-size 10
    python scripts/auto_train.py --force          # train lại dù đã có file
    python scripts/auto_train.py --quick          # 800 episode (thử nhanh)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BOARD_SIZES, dqn_model_path  # noqa: E402
from train import run_training  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    """Parser CLI cho auto train."""
    parser = argparse.ArgumentParser(
        description="Tự train DQN khi thiếu checkpoint (hoặc --force)."
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=15,
        choices=BOARD_SIZES,
        help="Kích thước bàn cờ (10 hoặc 15).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train lại dù đã có models/dqn_{size}.pth.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Huấn luyện nhanh (~800 ep, Minimax depth 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / cuda / mps (mặc định tự phát hiện).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Điểm vào auto train.

    Returns:
        0 nếu thành công hoặc đã có model; 1 nếu lỗi.
    """
    args = _build_parser().parse_args(argv)
    model_path = dqn_model_path(args.board_size)

    if model_path.exists() and not args.force:
        print(f"Đã có model: {model_path} — bỏ qua (dùng --force để train lại).")
        return 0

    if args.quick:
        episodes = 800
        mode = "minimax"
        opponent_depth = 2
        save_every = 200
        log_every = 50
        eval_every = 200
    else:
        episodes = 2500
        mode = "minimax"
        opponent_depth = 3
        save_every = 500
        log_every = 50
        eval_every = 500

    print("=" * 60)
    print("  AUTO TRAIN DQN")
    print(f"  Bàn: {args.board_size}x{args.board_size} | Episodes: {episodes}")
    print(f"  Mode: {mode} (depth={opponent_depth})")
    print("=" * 60)

    ns = argparse.Namespace(
        board_size=args.board_size,
        episodes=episodes,
        mode=mode,
        opponent_depth=opponent_depth,
        save_every=save_every,
        log_every=log_every,
        eval_every=eval_every,
        eval_games=10,
        device=args.device,
        seed=42,
        resume=str(model_path) if model_path.exists() and args.force else None,
        output=str(model_path),
    )

    try:
        saved = run_training(ns)
        print(f"Xong. Model: {saved}")
    except (ValueError, FileNotFoundError, KeyboardInterrupt) as exc:
        print(f"Lỗi auto train: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
