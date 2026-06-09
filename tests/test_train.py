#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test cho ``train.py`` và ``load_checkpoint``."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from ai.dqn_trainer import DQNTrainer
from train import _build_parser, run_training


def test_train_parser_co_cac_tham_so_chinh() -> None:
    """CLI parser phải có các flag huấn luyện cốt lõi."""
    parser = _build_parser()
    args = parser.parse_args(["--episodes", "100", "--board-size", "10", "--mode", "minimax"])
    assert args.episodes == 100
    assert args.board_size == 10
    assert args.mode == "minimax"


def test_run_training_vong_ngan(tmp_path: Path) -> None:
    """Huấn luyện 3 episode trên bàn 10 phải tạo file model."""
    out = tmp_path / "dqn_10.pth"
    args = Namespace(
        board_size=10,
        episodes=3,
        mode="selfplay",
        opponent_depth=1,
        save_every=0,
        log_every=0,
        eval_every=0,
        eval_games=2,
        device="cpu",
        seed=0,
        resume=None,
        output=str(out),
    )
    saved = run_training(args)
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_load_checkpoint_tiep_tuc_huan_luyen(tmp_path: Path) -> None:
    """Nạp checkpoint và tiếp tục train không lỗi."""
    path = tmp_path / "dqn_5.pth"
    trainer = DQNTrainer(board_size=5, buffer_capacity=200, batch_size=8, seed=1)
    trainer.train(num_episodes=2)
    trainer.save_agent(path)

    trainer2 = DQNTrainer(board_size=5, buffer_capacity=200, batch_size=8, seed=2)
    trainer2.load_checkpoint(path)
    trainer2.train(num_episodes=2)
    assert trainer2.stats.episodes == 2


def test_train_stats_win_rate() -> None:
    """TrainStats tính win_rate_x sau vài episode."""
    trainer = DQNTrainer(board_size=5, buffer_capacity=200, batch_size=8, seed=3)
    trainer.train(num_episodes=5)
    assert trainer.stats.episodes == 5
    total = trainer.stats.wins_x + trainer.stats.losses_x + trainer.stats.draws
    assert total == 5
