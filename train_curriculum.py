#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curriculum huấn luyện DQN: Minimax depth 1 → 2 → 3 → self-play.

Chạy:
    python train_curriculum.py
    python train_curriculum.py --board-size 15 --device cpu
    make train-curriculum
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

from config import BOARD_SIZES, dqn_model_path
from train import run_training

# (mode, depth, episodes, max_branch)
# max_branch giới hạn nhánh Minimax đối thủ để training chạy nhanh hơn.
# Không giới hạn ở depth=1 (đủ nhanh); depth≥2 cần giới hạn tránh bùng nổ.
PHASES: tuple[tuple[str, int | None, int, int | None], ...] = (
    ("minimax", 1,  500, None),   # ~1–2h: d=1 không giới hạn
    ("minimax", 2,  600, 15),     # ~1–2h: d=2 max_branch=15
    ("minimax", 3,  400, 10),     # ~1–2h: d=3 max_branch=10
    ("selfplay", None, 500, None),# ~0.5h: self-play
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curriculum train DQN (depth 1→2→3→self-play).")
    parser.add_argument(
        "--board-size",
        type=int,
        default=15,
        choices=BOARD_SIZES,
        help="Kích thước bàn cờ.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị PyTorch (cpu / cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Tắt thanh tiến độ (dùng khi redirect log).",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=1,
        choices=range(1, len(PHASES) + 1),
        metavar="N",
        help=(
            "Bắt đầu từ phase N (1–4). Dùng khi đã xong phase trước — "
            "nạp checkpoint hiện có, không train lại từ đầu."
        ),
    )
    return parser


def _phase_label(mode: str, depth: int | None, max_branch: int | None = None) -> str:
    if mode == "selfplay":
        return "Self-play"
    branch_str = f" mb={max_branch}" if max_branch else ""
    return f"Minimax d={depth}{branch_str}"


def _build_phase_args(
    board_size: int,
    mode: str,
    depth: int | None,
    episodes: int,
    max_branch: int | None,
    resume: Path | None,
    device: str | None,
    seed: int,
    *,
    no_progress: bool,
    phase_idx: int,
    total_phases: int,
) -> Namespace:
    return Namespace(
        board_size=board_size,
        episodes=episodes,
        mode=mode,
        opponent_depth=depth if depth is not None else 2,
        opponent_max_branch=max_branch,
        save_every=0,
        log_every=0,
        eval_every=250,
        eval_games=20,
        device=device,
        seed=seed,
        resume=str(resume) if resume is not None else None,
        output=None,
        no_save_gate=False,
        no_progress=no_progress,
        phase_label=f"P{phase_idx}/{total_phases} {_phase_label(mode, depth, max_branch)}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    board_size = args.board_size
    start_phase: int = args.start_phase
    checkpoint = dqn_model_path(board_size)
    resume: Path | None = checkpoint if checkpoint.exists() else None
    if start_phase > 1 and resume is None:
        print(f"Lỗi: --start-phase {start_phase} cần checkpoint {checkpoint}", file=sys.stderr)
        return 1

    phases_to_run = PHASES[start_phase - 1 :]
    skipped_episodes = sum(ep for _, _, ep, _ in PHASES[: start_phase - 1])
    total_episodes = sum(episodes for _, _, episodes, _ in phases_to_run)
    show_progress = not args.no_progress and sys.stderr.isatty()

    print("=" * 60)
    print("  CURRICULUM DQN — AI CỜ CARO")
    print("=" * 60)
    print(f"  Bàn cờ      : {board_size}x{board_size}")
    print(f"  Tổng episode: {total_episodes}" + (f" (bỏ qua phase 1–{start_phase - 1})" if start_phase > 1 else ""))
    print(f"  Bắt đầu     : phase {start_phase}/{len(PHASES)}")
    print(f"  Checkpoint  : {checkpoint}" + (" (nạp weights đã học)" if resume else " (mới)"))
    print("=" * 60)

    curriculum_bar: tqdm | None = None
    if show_progress:
        curriculum_bar = tqdm(
            total=total_episodes,
            desc="Curriculum",
            unit="ep",
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    completed_before_phase = skipped_episodes
    if curriculum_bar is not None and skipped_episodes:
        curriculum_bar.n = skipped_episodes
        curriculum_bar.refresh()

    for phase_idx, (mode, depth, episodes, max_branch) in enumerate(phases_to_run, start=start_phase):
        label = _phase_label(mode, depth, max_branch)
        if curriculum_bar is not None:
            curriculum_bar.set_postfix_str(f"phase={phase_idx}/{len(PHASES)} {label}", refresh=True)
        else:
            print(f"\n>>> Phase {phase_idx}/{len(PHASES)}: {label} ({episodes} episodes)")

        phase_args = _build_phase_args(
            board_size,
            mode,
            depth,
            episodes,
            max_branch,
            resume,
            args.device,
            args.seed,
            no_progress=args.no_progress,
            phase_idx=phase_idx,
            total_phases=len(PHASES),
        )

        def on_episode(
            _ep: int,
            stats: object,
            *,
            _bar: tqdm | None = curriculum_bar,
            _base: int = completed_before_phase,
            _phase: int = phase_idx,
            _label: str = label,
        ) -> None:
            if _bar is None:
                return
            _bar.n = _base + _ep
            _bar.set_postfix_str(
                f"P{_phase}/{len(PHASES)} {_label} | win={getattr(stats, 'win_rate_x', 0):.0%}",
                refresh=False,  # noqa: FBT003
            )
            _bar.refresh()

        try:
            run_training(phase_args, on_episode=on_episode)
        except KeyboardInterrupt:
            if curriculum_bar is not None:
                curriculum_bar.close()
            print("\nNgắt curriculum (Ctrl+C).", flush=True)
            return 130

        completed_before_phase += episodes
        if curriculum_bar is not None:
            curriculum_bar.n = completed_before_phase
            curriculum_bar.refresh()
        resume = checkpoint

    if curriculum_bar is not None:
        curriculum_bar.close()

    print("\nCurriculum hoàn tất. Model:", checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
