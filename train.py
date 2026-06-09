#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huấn luyện DQN cho AI Cờ Caro — self-play hoặc đấu với Minimax.

Chạy:
    python train.py                          # self-play, bàn 15x15
    python train.py --mode minimax           # học bằng cách đấu Minimax
    python train.py --board-size 10 --episodes 1000
    python train.py --resume models/dqn_15.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ai.dqn_trainer import DQNTrainer, TrainStats
from ai.evaluate import play_game
from ai.dqn_agent import DQNAgent
from ai.minimax_agent import MinimaxAgent
from config import Player
from config import (
    BOARD_SIZES,
    DQN_BUFFER_CAPACITY,
    DQN_DEFAULT_EPISODES,
    DQN_EVAL_EVERY,
    DQN_EVAL_GAMES,
    DQN_LOG_EVERY,
    DQN_SAVE_EVERY,
    Difficulty,
    dqn_model_path,
)


def _build_parser() -> argparse.ArgumentParser:
    """Tạo parser tham số dòng lệnh cho script huấn luyện."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện DQN cho AI Cờ Caro (self-play hoặc vs Minimax)."
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=15,
        choices=BOARD_SIZES,
        help="Kích thước bàn cờ (10 hoặc 15).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DQN_DEFAULT_EPISODES,
        help=f"Số episode huấn luyện (mặc định {DQN_DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--mode",
        choices=("selfplay", "minimax"),
        default="selfplay",
        help="selfplay = tự đấu; minimax = DQN (X) vs Minimax (O).",
    )
    parser.add_argument(
        "--opponent-depth",
        type=int,
        default=int(Difficulty.MEDIUM),
        choices=[int(d) for d in Difficulty],
        help="Độ sâu Minimax khi --mode minimax (1–4).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DQN_SAVE_EVERY,
        help=f"Lưu checkpoint mỗi N episode (0 = tắt, mặc định {DQN_SAVE_EVERY}).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DQN_LOG_EVERY,
        help=f"In log mỗi N episode (mặc định {DQN_LOG_EVERY}).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DQN_EVAL_EVERY,
        help=f"Đánh giá vs Minimax mỗi N episode (0 = tắt, mặc định {DQN_EVAL_EVERY}).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=DQN_EVAL_GAMES,
        help=f"Số ván đấu thử mỗi lần eval (mặc định {DQN_EVAL_GAMES}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị PyTorch: cpu hoặc cuda (mặc định tự phát hiện).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên (mặc định 42).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Đường dẫn checkpoint .pth để tiếp tục huấn luyện.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn lưu model cuối (mặc định models/dqn_{size}.pth).",
    )
    return parser


def _log_progress(
    episode: int,
    stats: TrainStats,
    epsilon: float,
    last_reward_x: float,
) -> None:
    """In một dòng tiến độ huấn luyện ra console."""
    print(
        f"[Ep {episode:5d}] "
        f"ε={epsilon:.3f} | "
        f"loss={stats.avg_loss:.4f} | "
        f"buf={stats.total_steps} | "
        f"X thắng={stats.win_rate_x:.1%} | "
        f"R_X={last_reward_x:+.2f}",
        flush=True,
    )


def _evaluate_dqn(board_size: int, num_games: int, minimax_depth: int) -> float:
    """Đấu thử DQN (greedy) vs Minimax và trả tỷ lệ thắng của DQN.

    Args:
        board_size: Kích thước bàn cờ.
        num_games: Số ván đấu thử.
        minimax_depth: Độ sâu Minimax đối thủ.

    Returns:
        Tỷ lệ thắng của DQN trên tổng số ván.
    """
    dqn = DQNAgent(board_size=board_size, epsilon=0.0)
    minimax = MinimaxAgent(depth=minimax_depth)
    dqn_wins = 0

    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(dqn, minimax, board_size)
            if winner is Player.X:
                dqn_wins += 1
        else:
            winner = play_game(minimax, dqn, board_size)
            if winner is Player.O:
                dqn_wins += 1

    return dqn_wins / num_games if num_games else 0.0


def run_training(args: argparse.Namespace) -> Path:
    """Thực thi vòng lặp huấn luyện theo tham số CLI.

    Args:
        args: Namespace từ argparse.

    Returns:
        Path file model đã lưu cuối cùng.
    """
    board_size: int = args.board_size
    output = Path(args.output) if args.output else dqn_model_path(board_size)

    opponent = None
    mode_label = "Self-play"
    if args.mode == "minimax":
        opponent = MinimaxAgent(depth=args.opponent_depth)
        mode_label = f"vs Minimax (depth={args.opponent_depth})"

    print("=" * 60)
    print("  HUẤN LUYỆN DQN — AI CỜ CARO")
    print("=" * 60)
    print(f"  Bàn cờ     : {board_size}x{board_size}")
    print(f"  Chế độ     : {mode_label}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Thiết bị   : {args.device or 'tự phát hiện'}")
    print(f"  Checkpoint : {output}")
    print("=" * 60)

    trainer = DQNTrainer(
        board_size=board_size,
        device=args.device,
        buffer_capacity=DQN_BUFFER_CAPACITY,
        seed=args.seed,
    )

    if args.resume:
        resume_path = Path(args.resume)
        print(f"Tiếp tục từ checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)

    t0 = time.perf_counter()

    try:
        for ep in range(1, args.episodes + 1):
            reward_x = trainer.train_episode(opponent=opponent)

            if args.log_every > 0 and ep % args.log_every == 0:
                _log_progress(ep, trainer.stats, trainer.epsilon, reward_x)

            if args.save_every > 0 and ep % args.save_every == 0:
                saved = trainer.save_agent(output)
                print(f"  → Đã lưu checkpoint: {saved}")

            if (
                args.eval_every > 0
                and ep % args.eval_every == 0
                and trainer.stats.episodes >= 10
            ):
                trainer.save_agent(output)
                win_rate = _evaluate_dqn(board_size, args.eval_games, args.opponent_depth)
                print(
                    f"  → Eval vs Minimax ({args.eval_games} ván): "
                    f"DQN thắng {win_rate:.1%}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nNgắt huấn luyện (Ctrl+C). Đang lưu model hiện tại...", flush=True)

    # Lưu model cuối cùng (kể cả khi bị ngắt).
    final_path = trainer.save_agent(output)
    elapsed = time.perf_counter() - t0

    print("=" * 60)
    print("  HOÀN TẤT HUẤN LUYỆN")
    print(f"  Thời gian  : {elapsed:.1f}s")
    print(f"  Episodes   : {trainer.stats.episodes}")
    print(f"  Tổng bước  : {trainer.stats.total_steps}")
    print(f"  X thắng    : {trainer.stats.win_rate_x:.1%}")
    print(f"  Model      : {final_path}")
    print("=" * 60)
    return final_path


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI huấn luyện DQN.

    Args:
        argv: Tham số dòng lệnh (mặc định ``sys.argv[1:]``).

    Returns:
        Mã thoát 0 nếu thành công.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        run_training(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
