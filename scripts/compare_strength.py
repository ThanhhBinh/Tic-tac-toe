#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║         SO SÁNH SỨC MẠNH AI CỜ CARO — TRỰC QUAN            ║
║  Minimax  |  DQN (Deep Q-Network)  |  Hybrid (MM + DQN)     ║
╚══════════════════════════════════════════════════════════════╝

Chạy:
    .venv/bin/python scripts/compare_strength.py
    .venv/bin/python scripts/compare_strength.py --quick
    .venv/bin/python scripts/compare_strength.py --games 20 --difficulty HARD
    .venv/bin/python scripts/compare_strength.py --pair hybrid minimax --games 10
    .venv/bin/python scripts/compare_strength.py --all-difficulties --games 6
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── ANSI màu ─────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# Màu cho từng thuật toán
AGENT_COLOR = {
    "hybrid":  GREEN,
    "minimax": CYAN,
    "dqn":     YELLOW,
    "random":  DIM,
}
AGENT_LABEL = {
    "hybrid":  "Hybrid (MM+DQN)",
    "minimax": "Minimax        ",
    "dqn":     "DQN            ",
}


def _strip(s: str) -> str:
    """Bỏ ANSI escape codes để đo độ dài thực."""
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _bar(pct: float, width: int = 28, color: str = GREEN) -> str:
    """Thanh ngang Unicode hiển thị tỷ lệ."""
    filled = round(pct * width)
    empty  = width - filled
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"


def _hbar(label: str, pct: float, extra: str = "", width: int = 28) -> str:
    key = label.strip().split()[0].lower()
    color = AGENT_COLOR.get(key, CYAN)
    bar   = _bar(pct, width, color)
    return f"  {color}{BOLD}{label:<16}{RESET}  {bar}  {color}{BOLD}{pct:5.0%}{RESET}  {extra}"


def _box(title: str, lines: list[str], W: int = 64) -> str:
    inner = W - 4
    top = f"  ╔{'═' * inner}╗"
    sep = f"  ╠{'═' * inner}╣"
    bot = f"  ╚{'═' * inner}╝"
    rows = [top]
    # title
    t = f"{BOLD}{title}{RESET}"
    pad = inner - len(_strip(t))
    rows.append(f"  ║ {t}{' ' * max(0, pad - 1)}║")
    rows.append(sep)
    for line in lines:
        raw_len = len(_strip(line))
        pad = inner - raw_len
        rows.append(f"  ║ {line}{' ' * max(0, pad - 1)}║")
    rows.append(bot)
    return "\n".join(rows)


# ─── Wrapper đo thời gian ─────────────────────────────────────────────────────
class TimedAgent:
    """Bọc agent gốc, ghi lại thời gian get_move() mỗi nước."""

    def __init__(self, agent, key: str) -> None:
        self._agent = agent
        self.key    = key
        self.name   = agent.name
        self.times: list[float] = []

    def get_move(self, env):
        t0 = time.perf_counter()
        mv = self._agent.get_move(env)
        self.times.append(time.perf_counter() - t0)
        return mv

    @property
    def avg_ms(self) -> float:
        return (sum(self.times) / len(self.times) * 1000) if self.times else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times) * 1000 if self.times else 0.0

    @property
    def n_moves(self) -> int:
        return len(self.times)


# ─── Chơi 1 ván ───────────────────────────────────────────────────────────────
def _play_one(a: TimedAgent, b: TimedAgent, board_size: int):
    from config import Player
    from core.caro_env import CaroEnv

    env = CaroEnv(size=board_size)
    env.reset()
    table = {Player.X: a, Player.O: b}
    limit = board_size * board_size + 1
    steps = 0
    while not env.done and steps < limit:
        ag   = table[env.current_player]
        move = ag.get_move(env)
        env.step(move)
        steps += 1
    return env.winner


# ─── Trận đấu N ván (đổi màu xen kẽ) ────────────────────────────────────────
def play_match(
    key_a: str, a: TimedAgent,
    key_b: str, b: TimedAgent,
    num_games: int,
    board_size: int,
    silent: bool = False,
) -> dict:
    from config import Player

    wins_a = wins_b = draws = 0
    col_a = AGENT_COLOR.get(key_a, CYAN)
    col_b = AGENT_COLOR.get(key_b, YELLOW)

    for i in range(num_games):
        # Đổi màu để công bằng
        if i % 2 == 0:
            winner = _play_one(a, b, board_size)
            if winner is Player.X:
                wins_a += 1
                sym = f"{col_a}WIN {key_a.upper()}{RESET}"
            elif winner is Player.O:
                wins_b += 1
                sym = f"{col_b}WIN {key_b.upper()}{RESET}"
            else:
                draws += 1
                sym = f"{DIM}HOA{RESET}"
        else:
            winner = _play_one(b, a, board_size)
            if winner is Player.X:
                wins_b += 1
                sym = f"{col_b}WIN {key_b.upper()}{RESET}"
            elif winner is Player.O:
                wins_a += 1
                sym = f"{col_a}WIN {key_a.upper()}{RESET}"
            else:
                draws += 1
                sym = f"{DIM}HOA{RESET}"

        if not silent:
            score_a = f"{col_a}{wins_a}{RESET}"
            score_b = f"{col_b}{wins_b}{RESET}"
            print(f"    Ván {i+1:>2}/{num_games}  →  {sym:35}  [{score_a} - {score_b} hoà:{draws}]")

    total = num_games
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws":  draws,
        "win_rate_a": wins_a / total if total else 0.0,
        "win_rate_b": wins_b / total if total else 0.0,
    }


# ─── Round-robin ──────────────────────────────────────────────────────────────
def round_robin(
    agents: dict[str, TimedAgent],
    num_games: int,
    board_size: int,
) -> dict:
    keys  = list(agents.keys())
    matrix: dict[str, dict] = {k: {} for k in keys}
    total_wins:  dict[str, int] = {k: 0 for k in keys}
    total_games: dict[str, int] = {k: 0 for k in keys}

    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    for key_a, key_b in pairs:
        ca = AGENT_COLOR.get(key_a, CYAN)
        cb = AGENT_COLOR.get(key_b, YELLOW)
        print(f"\n  ┌─  {ca}{BOLD}{key_a.upper()}{RESET}  vs  {cb}{BOLD}{key_b.upper()}{RESET}"
              f"  —  {num_games} ván  ─────────────────────")
        t0  = time.perf_counter()
        res = play_match(key_a, agents[key_a], key_b, agents[key_b],
                         num_games, board_size)
        dt  = time.perf_counter() - t0

        matrix[key_a][key_b] = res
        matrix[key_b][key_a] = {
            "wins_a": res["wins_b"], "wins_b": res["wins_a"],
            "draws":  res["draws"],
            "win_rate_a": res["win_rate_b"],
            "win_rate_b": res["win_rate_a"],
        }
        total_wins[key_a]  += res["wins_a"]
        total_wins[key_b]  += res["wins_b"]
        total_games[key_a] += num_games
        total_games[key_b] += num_games

        wr_a = res["win_rate_a"]
        wr_b = res["win_rate_b"]
        winner_key = key_a if wr_a > wr_b else (key_b if wr_b > wr_a else "-")
        winner_col = AGENT_COLOR.get(winner_key, DIM)
        print(f"  └─  {ca}{key_a}: {res['wins_a']}{RESET}"
              f"  —  {cb}{key_b}: {res['wins_b']}{RESET}"
              f"  —  hoà: {res['draws']}"
              f"  ({dt:.1f}s)"
              f"  →  {winner_col}{BOLD}{winner_key.upper()} dẫn{RESET}")

    ranking = sorted(keys, key=lambda k: total_wins[k], reverse=True)
    standings = {
        k: {
            "total_wins":  total_wins[k],
            "total_games": total_games[k],
            "win_rate":    total_wins[k] / total_games[k] if total_games[k] else 0.0,
            "rank":        ranking.index(k) + 1,
        }
        for k in keys
    }
    return {"matrix": matrix, "standings": standings, "ranking": ranking, "keys": keys}


# ─── In kết quả tổng hợp ──────────────────────────────────────────────────────
def print_results(result: dict, agents: dict[str, TimedAgent],
                  board_size: int, difficulty: str) -> None:
    keys      = result["ranking"]
    standings = result["standings"]
    matrix    = result["matrix"]
    W = 64

    rank_badge = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}

    # ── Header ──
    print()
    print("  " + "═" * W)
    print(f"  {BOLD}  KẾT QUẢ SO SÁNH  —  bàn {board_size}×{board_size}  |  Độ khó: {difficulty}{RESET}")
    print("  " + "═" * W)

    # ── Bar chart tỷ lệ thắng ──
    print(f"\n  {BOLD}TY LE THANG TOAN GIAI{RESET}")
    print("  " + "─" * W)
    for k in keys:
        s    = standings[k]
        wr   = s["win_rate"]
        col  = AGENT_COLOR.get(k, CYAN)
        lbl  = AGENT_LABEL.get(k, f"{k:<15}")
        badge= rank_badge.get(s["rank"], f"[#{s['rank']}]")
        extra = f"({s['total_wins']}/{s['total_games']} van)"
        print(_hbar(f"{badge} {lbl}", wr, extra))

    # ── Bảng đối đầu ──
    print(f"\n  {BOLD}BANG DOI DAU  (win-rate hang vs cot){RESET}")
    print("  " + "─" * W)
    COL = 14
    hdr = "            " + "".join(f"{k.upper():>{COL}}" for k in keys)
    print("  " + hdr)
    print("  " + "  " + "─" * (10 + COL * len(keys)))
    for a in keys:
        ca   = AGENT_COLOR.get(a, CYAN)
        row  = f"  {ca}{BOLD}{a.upper():>9}{RESET} |"
        for b in keys:
            if a == b:
                row += f"{'—':>{COL}}"
            else:
                wr = matrix[a][b]["win_rate_a"]
                if wr > 0.55:
                    col = GREEN + BOLD
                elif wr < 0.45:
                    col = RED
                else:
                    col = DIM
                row += f"{col}{wr:>12.0%}{RESET}  "
        print(row)

    # ── Tốc độ suy nghĩ ──
    print(f"\n  {BOLD}TOC DO SUY NGHI (ms/nuoc){RESET}")
    print("  " + "─" * W)
    max_avg = max((agents[k].avg_ms for k in keys if k in agents), default=1)
    for k in keys:
        if k not in agents:
            continue
        ag  = agents[k]
        col = AGENT_COLOR.get(k, CYAN)
        lbl = AGENT_LABEL.get(k, f"{k:<15}")
        # bar: nhanh hơn = dài hơn (nghịch đảo)
        speed_pct = 1.0 - min(ag.avg_ms / max(max_avg, 1), 1.0)
        spd_bar   = _bar(speed_pct, 20, col)
        print(f"  {col}{BOLD}{lbl}{RESET}   "
              f"TB: {BOLD}{ag.avg_ms:>7.1f} ms{RESET}   "
              f"Max: {ag.max_ms:>7.1f} ms   "
              f"{spd_bar}")
    print(f"  {DIM}(Thanh toc do: dai hon = nhanh hon){RESET}")

    # ── Phân tích Head-to-Head chi tiết ──
    if "hybrid" in keys and len(keys) > 1:
        print(f"\n  {BOLD}HYBRID SO VOI TUNG DOI THU{RESET}")
        print("  " + "─" * W)
        for opp in keys:
            if opp == "hybrid":
                continue
            res = matrix.get("hybrid", {}).get(opp, {})
            wr  = res.get("win_rate_a", 0.0)
            wa  = res.get("wins_a", 0)
            wb  = res.get("wins_b", 0)
            d   = res.get("draws", 0)
            col_opp = AGENT_COLOR.get(opp, CYAN)
            if wr > 0.5:
                verdict = f"{GREEN}{BOLD}HYBRID THANG{RESET}"
            elif wr < 0.5:
                verdict = f"{RED}{BOLD}HYBRID THUA{RESET}"
            else:
                verdict = f"{DIM}HOA{RESET}"
            print(f"  Hybrid  vs  {col_opp}{opp.upper()}{RESET}:  "
                  f"{GREEN}{wa}{RESET} thang  {RED}{wb}{RESET} thua  {d} hoa  "
                  f"→  {verdict}  ({GREEN}{wr:.0%}{RESET})")

    # ── Verdict box ──
    winner     = result["ranking"][0]
    winner_col = AGENT_COLOR.get(winner, GREEN)
    winner_wr  = standings[winner]["win_rate"]
    is_hybrid  = (winner == "hybrid")

    box_lines: list[str] = []
    if is_hybrid:
        box_lines += [
            f"{GREEN}{BOLD}  *** HYBRID LA THUAT TOAN MANH NHAT! ***{RESET}",
            f"  Ty le thang: {GREEN}{BOLD}{winner_wr:.0%}{RESET}  tren toan giai",
            "",
            f"  Tai sao Hybrid manh hon?",
            f"  - Tim sau hon Minimax 1 ply (depth+1)",
            f"  - DQN sap xep nuoc tai root → alpha-beta cat tinh som",
            f"  - Incremental eval: chi tinh 4 duong qua o vua di",
            f"  - Ket hop: search rong + hoc su dung → manh + nhanh",
        ]
    else:
        box_lines += [
            f"{winner_col}{BOLD}  *** {winner.upper()} MANH NHAT ({winner_wr:.0%}) ***{RESET}",
        ]

    print()
    print(_box(f"KET LUAN", box_lines, W))
    print()


# ─── In kết quả multi-difficulty ─────────────────────────────────────────────
def print_multi_difficulty_summary(
    all_results: list[dict],   # list of (difficulty_name, result_dict, agents_dict)
) -> None:
    W = 64
    print()
    print("  " + "═" * W)
    print(f"  {BOLD}  TONG HOP THEO DO KHO  {RESET}")
    print("  " + "═" * W)

    header = f"  {'Do kho':<10}" + f"{'HYBRID':>10}" + f"{'MINIMAX':>10}" + f"{'DQN':>10}" + f"{'Winner':>12}"
    print(header)
    print("  " + "─" * W)

    for diff_name, result, _ in all_results:
        standings = result["standings"]
        ranking   = result["ranking"]
        winner    = ranking[0]
        wcol      = AGENT_COLOR.get(winner, GREEN)
        h_wr  = standings.get("hybrid",  {}).get("win_rate", 0.0)
        mm_wr = standings.get("minimax", {}).get("win_rate", 0.0)
        dq_wr = standings.get("dqn",     {}).get("win_rate", 0.0)
        print(f"  {diff_name:<10}"
              f"  {GREEN}{h_wr:>6.0%}{RESET}"
              f"  {CYAN}{mm_wr:>8.0%}{RESET}"
              f"  {YELLOW}{dq_wr:>8.0%}{RESET}"
              f"  {wcol}{BOLD}{winner.upper():>10}{RESET}")
    print("  " + "─" * W)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="So sanh suc manh AI Co Caro voi output truc quan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--board-size",  type=int, default=15)
    parser.add_argument("--games",       type=int, default=10,
                        help="So van moi cap (mac dinh 10)")
    parser.add_argument("--difficulty",  default="MEDIUM",
                        choices=["EASY", "MEDIUM", "HARD", "EXPERT"])
    parser.add_argument("--quick",       action="store_true",
                        help="Demo nhanh: 4 van/cap")
    parser.add_argument("--all-difficulties", action="store_true",
                        help="Chay ca 4 do kho, in tong hop")
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                        help="Chi dau 1 cap, vd: --pair hybrid minimax")
    parser.add_argument("--no-dqn", action="store_true",
                        help="Bo qua DQN (chi so Hybrid vs Minimax)")
    args = parser.parse_args(argv)

    if args.quick:
        args.games = 4

    from config import AIType, Difficulty
    from ai.factory import create_agent

    # Xác định danh sách agent
    if args.pair:
        keys = [x.lower() for x in args.pair]
    elif args.no_dqn:
        keys = ["hybrid", "minimax"]
    else:
        keys = ["hybrid", "minimax", "dqn"]

    type_map = {
        "minimax": AIType.MINIMAX,
        "dqn":     AIType.DQN,
        "hybrid":  AIType.HYBRID,
    }

    # ── Banner mở đầu ──
    W = 64
    print()
    print("  " + "╔" + "═" * W + "╗")
    print("  " + "║" + f"{BOLD}  SO SANH SUC MANH AI CO CARO{RESET}".center(W + 8) + "║")
    print("  " + "║" + f"  Minimax  |  DQN  |  Hybrid (Minimax + DQN)".center(W) + "║")
    print("  " + "╚" + "═" * W + "╝")

    if args.all_difficulties:
        diffs = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXPERT]
        all_results = []
        for diff in diffs:
            print(f"\n  {'─'*20}  {BOLD}{diff.name}{RESET}  {'─'*20}")
            agents = {}
            for k in keys:
                ag = create_agent(type_map[k], diff, args.board_size)
                agents[k] = TimedAgent(ag, k)
                print(f"  {DIM}  + {ag.name}{RESET}")
            result = round_robin(agents, args.games, args.board_size)
            print_results(result, agents, args.board_size, diff.name)
            all_results.append((diff.name, result, agents))
        print_multi_difficulty_summary(all_results)
        return 0

    # ── Single difficulty ──
    difficulty = Difficulty[args.difficulty]
    print(f"\n  {DIM}Khoi tao agents  ({args.difficulty}, ban {args.board_size}x{args.board_size})...{RESET}")
    agents = {}
    for k in keys:
        ag = create_agent(type_map[k], difficulty, args.board_size)
        agents[k] = TimedAgent(ag, k)
        col = AGENT_COLOR.get(k, DIM)
        print(f"  {col}  + {ag.name}{RESET}")

    t0 = time.perf_counter()
    print(f"\n  {BOLD}Bat dau doi khang...  {args.games} van/cap{RESET}")

    if args.pair:
        key_a, key_b = keys
        ca = AGENT_COLOR.get(key_a, CYAN)
        cb = AGENT_COLOR.get(key_b, YELLOW)
        print(f"\n  {ca}{BOLD}{key_a.upper()}{RESET}  vs  {cb}{BOLD}{key_b.upper()}{RESET}  —  {args.games} van\n")
        res = play_match(key_a, agents[key_a], key_b, agents[key_b],
                         args.games, args.board_size)
        print()
        wa, wb, d = res["wins_a"], res["wins_b"], res["draws"]
        wr_a = res["win_rate_a"]
        winner = key_a if wa > wb else (key_b if wb > wa else "HOA")
        wcol   = AGENT_COLOR.get(winner.lower(), DIM)
        print(f"  {ca}{BOLD}{key_a.upper()}{RESET}: {GREEN}{wa}{RESET} thang ({wr_a:.0%})"
              f"   {cb}{BOLD}{key_b.upper()}{RESET}: {wb} thang ({res['win_rate_b']:.0%})"
              f"   hoa: {d}")
        print(f"\n  Ket luan: {wcol}{BOLD}{winner.upper()} MANH HON{RESET}")
        # Timing
        for k in [key_a, key_b]:
            ag  = agents[k]
            col = AGENT_COLOR.get(k, DIM)
            print(f"  {col}{k.upper()}{RESET}  TB: {ag.avg_ms:.1f} ms/nuoc  Max: {ag.max_ms:.1f} ms")
    else:
        result = round_robin(agents, args.games, args.board_size)
        print_results(result, agents, args.board_size, args.difficulty)

    print(f"  {DIM}Tong thoi gian: {time.perf_counter() - t0:.1f}s{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
