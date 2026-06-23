#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huấn luyện DQN cho AI Cờ Caro — self-play hoặc đấu với Minimax.

Chạy:
    python train.py                          # self-play, bàn 15x15
    python train.py --mode minimax           # học bằng cách đấu Minimax
    python train.py --board-size 10 --episodes 1000
    python train.py --resume models/dqn_15.pth
    python train_curriculum.py               # curriculum depth 1→2→3→self-play
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from ai.dqn_trainer import DQNTrainer, TrainStats
from ai.evaluate import play_game
from ai.dqn_agent import DQNAgent
from ai.minimax_agent import MinimaxAgent
from ai.save_gate import evaluate_train_save_gate, tactical_benchmark_score
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
    dqn_model_best_path,
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
    parser.add_argument(
        "--no-save-gate",
        action="store_true",
        help="Tắt save gate — lưu mọi checkpoint như trước.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Tắt thanh tiến độ (dùng khi chạy trong CI hoặc redirect log).",
    )
    parser.add_argument(
        "--train-tactical",
        choices=("full", "safe", "none"),
        default=None,
        help=(
            "Mức luật tactical cho learner khi train. 'safe' (mặc định config) "
            "chỉ tự ăn thắng-ngay + chặn-thua-ngay để MẠNG tự học phần còn lại — "
            "đây là yếu tố then chốt giúp DQN học được (đừng dùng 'full')."
        ),
    )
    parser.add_argument(
        "--phase-label",
        type=str,
        default=None,
        help="Nhãn hiển thị trên thanh tiến độ (curriculum).",
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
    """Đấu thử DQN (greedy) vs Minimax và trả tỷ lệ thắng của DQN."""
    dqn = DQNAgent(board_size=board_size, epsilon=0.0)
    minimax = MinimaxAgent(depth=minimax_depth)
    if minimax_depth >= 2:
        minimax.max_branch = 15  # eval nhanh, chỉ cần tương đối
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


def _maybe_save_checkpoint(
    trainer: DQNTrainer,
    output: Path,
    board_size: int,
    win_rate: float | None,
    best_win_rate: float,
    use_gate: bool,
) -> tuple[float, bool]:
    """Lưu checkpoint nếu pass save gate; trả (best_win_rate, saved)."""
    if not use_gate:
        trainer.save_agent(output)
        return best_win_rate, True

    if win_rate is None:
        trainer.save_agent(output)
        return best_win_rate, True

    if evaluate_train_save_gate(trainer, board_size, win_rate, best_win_rate):
        trainer.save_agent(output)
        best_path = dqn_model_best_path(board_size)
        shutil.copy2(output, best_path)
        new_best = max(best_win_rate, win_rate)
        return new_best, True

    print(
        f"  → Bỏ qua lưu: win rate {win_rate:.1%} < best {best_win_rate:.1%}",
        flush=True,
    )
    return best_win_rate, False


def run_training(
    args: argparse.Namespace,
    *,
    on_episode: Callable[[int, TrainStats], None] | None = None,
) -> Path:
    """Thực thi vòng lặp huấn luyện theo tham số CLI."""
    board_size: int = args.board_size
    output = Path(args.output) if args.output else dqn_model_path(board_size)
    use_gate = not getattr(args, "no_save_gate", False)
    show_progress = not getattr(args, "no_progress", False) and sys.stderr.isatty()
    bar_desc = getattr(args, "phase_label", None) or "Train"

    opponent = None
    mode_label = "Self-play"
    eval_depth = args.opponent_depth
    opponent_max_branch: int | None = getattr(args, "opponent_max_branch", None)
    if args.mode == "minimax":
        opponent = MinimaxAgent(depth=args.opponent_depth)
        # Giới hạn nhánh để training nhanh hơn (mặc định 15 nếu depth >= 2)
        if opponent_max_branch is None and args.opponent_depth >= 2:
            opponent_max_branch = 15
        if opponent_max_branch is not None:
            opponent.max_branch = opponent_max_branch
        branch_str = f", max_branch={opponent_max_branch}" if opponent_max_branch else ""
        mode_label = f"vs Minimax (depth={args.opponent_depth}{branch_str})"
    else:
        eval_depth = int(Difficulty.MEDIUM)

    from config import DQN_TRAIN_TACTICAL_LEVEL

    tactical_level = getattr(args, "train_tactical", None) or DQN_TRAIN_TACTICAL_LEVEL

    print("=" * 60)
    print("  HUẤN LUYỆN DQN — AI CỜ CARO")
    print("=" * 60)
    print(f"  Bàn cờ     : {board_size}x{board_size}")
    print(f"  Chế độ     : {mode_label}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Save gate  : {'bật' if use_gate else 'tắt'}")
    print(f"  Tactical   : {tactical_level} (luật can thiệp learner khi train)")
    print(f"  Thiết bị   : {args.device or 'tự phát hiện'}")
    print(f"  Checkpoint : {output}")
    print("=" * 60)
    trainer = DQNTrainer(
        board_size=board_size,
        device=args.device,
        buffer_capacity=DQN_BUFFER_CAPACITY,
        seed=args.seed,
        train_tactical_level=tactical_level,
    )

    if args.resume:
        resume_path = Path(args.resume)
        print(f"Tiếp tục từ checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)
    else:
        best_path = dqn_model_best_path(board_size)
        if best_path.exists():
            trainer.load_checkpoint(best_path)
            print(f"Nạp best checkpoint: {best_path}")

    best_win_rate = 0.0
    best_path = dqn_model_best_path(board_size)
    t0 = time.perf_counter()

    def _emit(message: str) -> None:
        if pbar is not None:
            pbar.write(message)
        else:
            print(message, flush=True)

    pbar: tqdm | None = None
    try:
        pbar = tqdm(
            range(1, args.episodes + 1),
            desc=bar_desc,
            unit="ep",
            disable=not show_progress,
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        for ep in pbar:
            reward_x = trainer.train_episode(opponent=opponent)

            pbar.set_postfix(
                eps=f"ε={trainer.epsilon:.2f}",
                win=f"{trainer.stats.win_rate_x:.0%}",
                best=f"{best_win_rate:.0%}",
                loss=f"{trainer.stats.avg_loss:.2f}",
                refresh=False,
            )

            if on_episode is not None:
                on_episode(ep, trainer.stats)

            if args.log_every > 0 and ep % args.log_every == 0 and not show_progress:
                _log_progress(ep, trainer.stats, trainer.epsilon, reward_x)

            if args.save_every > 0 and ep % args.save_every == 0:
                win_rate = None
                if use_gate and trainer.stats.episodes >= 10:
                    win_rate = _evaluate_dqn(board_size, min(20, args.eval_games), eval_depth)
                best_win_rate, saved = _maybe_save_checkpoint(
                    trainer, output, board_size, win_rate, best_win_rate, use_gate
                )
                if saved:
                    _emit(f"  → Đã lưu checkpoint: {output}")

            if (
                args.eval_every > 0
                and ep % args.eval_every == 0
                and trainer.stats.episodes >= 10
            ):
                win_rate = _evaluate_dqn(board_size, args.eval_games, eval_depth)
                _, saved = _maybe_save_checkpoint(
                    trainer, output, board_size, win_rate, best_win_rate, use_gate
                )
                if saved:
                    best_win_rate = max(best_win_rate, win_rate)
                agent = DQNAgent(board_size=board_size, epsilon=0.0)
                agent.network.load_state_dict(trainer.policy_net.state_dict())
                agent._model_loaded = True
                _, _, tactical_pct = tactical_benchmark_score(agent, board_size)
                _emit(
                    f"  → Eval ep {ep}: DQN thắng {win_rate:.1%} | "
                    f"benchmark TH {tactical_pct:.0f}% | best {best_win_rate:.1%}"
                )
                pbar.set_postfix(
                    eps=f"ε={trainer.epsilon:.2f}",
                    win=f"{trainer.stats.win_rate_x:.0%}",
                    best=f"{best_win_rate:.0%}",
                    loss=f"{trainer.stats.avg_loss:.2f}",
                    refresh=False,
                )
    except KeyboardInterrupt:
        _emit("\nNgắt huấn luyện (Ctrl+C). Đang lưu model tốt nhất...")
    finally:
        if pbar is not None:
            pbar.close()

    if best_path.exists():
        shutil.copy2(best_path, output)
        final_path = output
    else:
        final_path = trainer.save_agent(output)

    elapsed = time.perf_counter() - t0

    print("=" * 60)
    print("  HOÀN TẤT HUẤN LUYỆN")
    print(f"  Thời gian  : {elapsed:.1f}s")
    print(f"  Episodes   : {trainer.stats.episodes}")
    print(f"  Tổng bước  : {trainer.stats.total_steps}")
    print(f"  X thắng    : {trainer.stats.win_rate_x:.1%}")
    print(f"  Best eval  : {best_win_rate:.1%}")
    print(f"  Model      : {final_path}")
    print("=" * 60)
    return final_path


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI huấn luyện DQN."""
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
