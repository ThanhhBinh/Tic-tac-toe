#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo SỨC MẠNH THẬT của Minimax / DQN / Hybrid bằng đối kháng vòng tròn.

Khác benchmark heuristic-1-nước (dễ gây hiểu nhầm), script này cho các agent
đánh nhau nhiều ván và in tỷ lệ thắng — thước đo đúng bản chất "ai mạnh hơn".

Chạy:
    .venv/bin/python scripts/compare_strength.py
    .venv/bin/python scripts/compare_strength.py --board-size 15 --games 20 --difficulty MEDIUM
    .venv/bin/python scripts/compare_strength.py --pair hybrid minimax --games 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.evaluate import play_match_agents, round_robin  # noqa: E402
from ai.factory import create_agent  # noqa: E402
from config import AIType, Difficulty  # noqa: E402


def _print_round_robin(result: dict) -> None:
    keys = result["ranking"]
    standings = result["standings"]
    matrix = result["matrix"]

    print("=" * 64)
    print(
        f"  GIẢI ĐẤU VÒNG TRÒN — bàn {result['board_size']}x{result['board_size']}"
        f" | {result['difficulty']} | {result['num_games_per_pair']} ván/cặp"
    )
    print("=" * 64)

    print("\n  Bảng đối đầu (win-rate hàng so với cột):")
    header = "        " + "".join(f"{k:>10}" for k in keys)
    print(header)
    for a in keys:
        row = f"  {a:>6}"
        for b in keys:
            if a == b:
                row += f"{'—':>10}"
            else:
                wr = matrix[a][b]["win_rate_a"]
                row += f"{wr:>9.0%} "
        print(row)

    print("\n  Xếp hạng (tổng thắng toàn giải):")
    for k in keys:
        s = standings[k]
        print(
            f"   #{s['rank']}  {k:<8} thắng {s['total_wins']:>3}/{s['total_games']:<3}"
            f" ({s['win_rate']:.0%})   [{s['name']}]"
        )
    print(f"\n  → Mạnh nhất: {result['winner'].upper()}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo sức mạnh thật bằng đối kháng.")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--games", type=int, default=20, help="Số ván mỗi cặp.")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="MEDIUM",
        choices=[d.name for d in Difficulty],
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="Chỉ đấu 1 cặp, vd: --pair hybrid minimax.",
    )
    args = parser.parse_args(argv)
    difficulty = Difficulty[args.difficulty]

    t0 = time.perf_counter()
    if args.pair:
        type_map = {
            "minimax": AIType.MINIMAX,
            "dqn": AIType.DQN,
            "hybrid": AIType.HYBRID,
        }
        a_key, b_key = args.pair[0].lower(), args.pair[1].lower()
        a = create_agent(type_map[a_key], difficulty, args.board_size)
        b = create_agent(type_map[b_key], difficulty, args.board_size)
        res = play_match_agents(a, b, args.games, args.board_size)
        print(
            f"{a_key} vs {b_key} ({args.games} ván, bàn {args.board_size}, {difficulty.name}):\n"
            f"  {a_key}: {res['wins_a']} thắng ({res['win_rate_a']:.0%}) | "
            f"{b_key}: {res['wins_b']} thắng ({res['win_rate_b']:.0%}) | "
            f"hoà: {res['draws']}"
        )
    else:
        result = round_robin(difficulty, args.board_size, args.games)
        _print_round_robin(result)

    print(f"  (thời gian: {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
