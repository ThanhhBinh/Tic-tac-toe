#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark so sánh Minimax, DQN và Hybrid trên 10 tình huống cố định.

Mỗi tình huống là một thế cờ giống nhau; cả 3 agent đều phải chọn nước đi
trên cùng bàn. Kết quả gồm: nước chọn, thời gian suy nghĩ, điểm heuristic
và đúng/sai so với nước chiến thuật chuẩn (nếu có).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ai.base_agent import Agent
from ai.factory import create_agent
from ai.heuristic import evaluate_board, find_tactical_move
from config import (
    AIType,
    DEFAULT_BOARD_SIZE,
    Difficulty,
    HYBRID_BENCHMARK_BRANCH_BONUS,
    HYBRID_BENCHMARK_RADIUS_BONUS,
    HYBRID_BENCHMARK_TIME_BUDGET_BASIC,
    HYBRID_EXTRA_DEPTH,
    HYBRID_PLAY_TIME_BUDGET_SEC,
    MINIMAX_PLAY_TIME_BUDGET_SEC,
    Player,
    TacticalConfig,
    create_caro_env,
    win_length_for_board,
)
from core.caro_env import CaroEnv
from core.constants import Move

ScenarioSetup = Callable[[CaroEnv], None]
ExpectedFn = Callable[[CaroEnv], frozenset[Move] | None]

AGENT_KEYS: tuple[str, ...] = ("minimax", "dqn", "hybrid")
AGENT_LABELS: dict[str, str] = {
    "minimax": "Minimax",
    "dqn": "DQN",
    "hybrid": "Hybrid",
}


@dataclass(frozen=True)
class BenchmarkScenario:
    """Một tình huống thử nghiệm — thiết lập thế cờ theo kích thước bàn."""

    id: str
    name: str
    description: str
    category: str
    setup: ScenarioSetup
    expected: ExpectedFn | None = None
    # True: agent phải dùng search/DQN — không shortcut find_tactical_move (TH ván thực).
    search_only: bool = False


def _set_turn(env: CaroEnv, player: Player, move_count: int, last_move: Move | None) -> None:
    env.current_player = player
    env._move_count = move_count  # noqa: SLF001
    env.last_move = last_move


def _place_line(
    env: CaroEnv,
    row: int,
    col0: int,
    player: Player,
    count: int,
    *,
    dr: int = 0,
    dc: int = 1,
) -> list[Move]:
    """Đặt ``count`` quân liên tiếp, trả danh sách ô đã đặt."""
    placed: list[Move] = []
    for i in range(count):
        r, c = row + dr * i, col0 + dc * i
        if env.in_bounds(r, c):
            env.board[r, c] = player
            placed.append((r, c))
    return placed


def _block_cells_for_line(
    env: CaroEnv,
    row: int,
    col0: int,
    count: int,
    *,
    dr: int = 0,
    dc: int = 1,
) -> frozenset[Move]:
    """Các ô chặn hai đầu một dãy ``count`` quân liên tiếp."""
    blocks: set[Move] = set()
    r0, c0 = row - dr, col0 - dc
    r1, c1 = row + dr * count, col0 + dc * count
    for r, c in ((r0, c0), (r1, c1)):
        if env.in_bounds(r, c) and env.board[r, c] == Player.EMPTY:
            blocks.add((r, c))
    return frozenset(blocks)


def _expected_tactical(env: CaroEnv) -> frozenset[Move] | None:
    move = find_tactical_move(env, env.current_player)
    return frozenset({move}) if move else None


def _expected_opening(env: CaroEnv) -> frozenset[Move]:
    mid = env.size // 2
    cells: set[Move] = {(mid, mid)}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = mid + dr, mid + dc
        if env.in_bounds(r, c):
            cells.add((r, c))
    return frozenset(cells)


def _setup_block_win(env: CaroEnv) -> None:
    """TH1: Đối thủ có (win_length-1) quân — phải chặn ngay."""
    env.reset()
    row = env.size // 2
    n = env.win_length - 1
    col0 = max(0, (env.size - n) // 2)
    placed = _place_line(env, row, col0, Player.O, n)
    _set_turn(env, Player.X, len(placed), placed[-1] if placed else None)


def _expected_block_win(env: CaroEnv) -> frozenset[Move]:
    row = env.size // 2
    n = env.win_length - 1
    col0 = max(0, (env.size - n) // 2)
    return _block_cells_for_line(env, row, col0, n)


def _setup_win_now(env: CaroEnv) -> None:
    """TH2: Ta có (win_length-1) quân — thắng ngay."""
    env.reset()
    row = env.size // 2
    n = env.win_length - 1
    col0 = max(0, (env.size - n) // 2)
    placed = _place_line(env, row, col0, Player.X, n)
    _set_turn(env, Player.X, len(placed), placed[-1] if placed else None)


def _expected_win_now(env: CaroEnv) -> frozenset[Move]:
    return _expected_block_win(env)


def _setup_block_open_four(env: CaroEnv) -> None:
    """TH3: Chặn đối thủ sắp tạo tứ mở / tam mở."""
    env.reset()
    row = env.size // 2
    n = min(3, env.win_length - 1)
    if n < 2:
        n = 2
    col0 = max(0, (env.size - n) // 2)
    placed = _place_line(env, row, col0, Player.O, n)
    _set_turn(env, Player.X, len(placed), placed[-1] if placed else None)


def _setup_replay(
    moves: tuple[tuple[int, int, Player], ...],
    *,
    to_play: Player = Player.X,
) -> ScenarioSetup:
    """Dựng thế cờ từ lịch sử nước đi — giống ván thật, quân cụm quanh khu chiến sự."""

    def _setup(env: CaroEnv) -> None:
        env.reset()
        placed = 0
        last: Move | None = None
        for row, col, player in moves:
            if not env.in_bounds(row, col):
                continue
            if env.board[row, col] != Player.EMPTY:
                continue
            env.board[row, col] = player
            placed += 1
            last = (row, col)
        _set_turn(env, to_play, placed, last)

    return _setup


def _setup_replay_prefix(
    moves: tuple[tuple[int, int, Player], ...],
    count: int,
    *,
    to_play: Player = Player.X,
) -> ScenarioSetup:
    """Lấy ``count`` nước đầu từ một ván mô phỏng."""
    return _setup_replay(moves[:count], to_play=to_play)


# Ván mô phỏng Minimax vs Minimax trên bàn 15×15 (quân cụm, không mẫu nhân tạo).
# Sinh bởi self-play depth=1; dùng làm nền cho TH04–TH10 và ADV01–ADV10.
_REAL_GAME_A: tuple[tuple[int, int, Player], ...] = (
    (7, 7, Player.X),
    (8, 6, Player.O),
    (6, 7, Player.X),
    (5, 7, Player.O),
    (8, 7, Player.X),
    (9, 7, Player.O),
    (7, 5, Player.X),
    (7, 6, Player.O),
    (6, 6, Player.X),
    (6, 5, Player.O),
    (9, 6, Player.X),
    (7, 8, Player.O),
    (5, 4, Player.X),
    (5, 5, Player.O),
    (4, 5, Player.X),
    (3, 6, Player.O),
    (5, 6, Player.X),
    (3, 4, Player.O),
    (10, 8, Player.X),
    (8, 8, Player.O),
    (6, 8, Player.X),
    (6, 9, Player.O),
    (5, 10, Player.X),
    (5, 9, Player.O),
    (4, 9, Player.X),
    (6, 11, Player.O),
    (6, 10, Player.X),
    (4, 10, Player.O),
    (10, 6, Player.X),
    (10, 7, Player.O),
    (11, 7, Player.X),
    (12, 8, Player.O),
    (3, 11, Player.X),
    (12, 6, Player.O),
    (12, 7, Player.X),
    (10, 5, Player.O),
    (3, 5, Player.X),
    (5, 8, Player.O),
    (4, 7, Player.X),
    (4, 6, Player.O),
    (7, 10, Player.X),
    (8, 10, Player.O),
    (2, 6, Player.X),
    (4, 4, Player.O),
    (3, 7, Player.X),
    (1, 5, Player.O),
    (2, 4, Player.X),
    (2, 5, Player.O),
    (3, 3, Player.X),
    (1, 3, Player.O),
    (4, 3, Player.X),
    (5, 3, Player.O),
    (6, 2, Player.X),
    (6, 3, Player.O),
    (7, 3, Player.X),
)

# Ván thứ hai — nhánh mở đầu khác (tấn công phía nam), dùng cho ADV nửa sau.
_REAL_GAME_B: tuple[tuple[int, int, Player], ...] = (
    (7, 7, Player.X),
    (7, 8, Player.O),
    (8, 7, Player.X),
    (6, 8, Player.O),
    (9, 7, Player.X),
    (8, 8, Player.O),
    (6, 7, Player.X),
    (7, 9, Player.O),
    (8, 6, Player.X),
    (9, 8, Player.O),
    (7, 6, Player.X),
    (10, 7, Player.O),
    (6, 6, Player.X),
    (8, 9, Player.O),
    (9, 6, Player.X),
    (7, 5, Player.O),
    (10, 8, Player.X),
    (6, 9, Player.O),
    (8, 5, Player.X),
    (9, 9, Player.O),
    (7, 10, Player.X),
    (6, 5, Player.O),
    (8, 10, Player.X),
    (5, 8, Player.O),
    (9, 5, Player.X),
    (10, 9, Player.O),
    (6, 10, Player.X),
    (8, 4, Player.O),
    (7, 4, Player.X),
    (9, 10, Player.O),
    (5, 7, Player.X),
    (10, 6, Player.O),
    (6, 4, Player.X),
    (11, 8, Player.O),
    (5, 6, Player.X),
    (4, 8, Player.O),
    (9, 4, Player.X),
    (7, 11, Player.O),
    (10, 5, Player.X),
    (5, 9, Player.O),
    (8, 11, Player.X),
    (4, 7, Player.O),
    (11, 6, Player.X),
    (6, 12, Player.O),
    (9, 11, Player.X),
    (3, 7, Player.O),
    (10, 10, Player.X),
    (5, 10, Player.O),
    (11, 9, Player.X),
    (4, 6, Player.O),
    (12, 7, Player.X),
    (7, 12, Player.O),
    (8, 12, Player.X),
    (9, 12, Player.O),
    (10, 11, Player.X),
    (6, 11, Player.O),
)


def _setup_advanced(pieces: list[tuple[int, int, Player]]) -> ScenarioSetup:
    """Legacy: chuyển danh sách quân tĩnh sang replay (giữ tương thích test cũ)."""
    return _setup_replay(tuple(pieces))


# ---------------------------------------------------------------------------
# TH04–TH10: Thế cờ thực tế (replay ván mô phỏng — quân cụm, không mẫu chéo/góc giả).
# ---------------------------------------------------------------------------

BENCHMARK_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        id="th01",
        name="Chặn thắng ngay",
        description="Đối thủ có dãy (win_length−1) quân — bắt buộc chặn.",
        category="Phòng thủ",
        setup=_setup_block_win,
        expected=_expected_block_win,
    ),
    BenchmarkScenario(
        id="th02",
        name="Thắng ngay",
        description="Ta có dãy (win_length−1) quân — đánh nước thắng ngay.",
        category="Tấn công",
        setup=_setup_win_now,
        expected=_expected_win_now,
    ),
    BenchmarkScenario(
        id="th03",
        name="Chặn tứ mở",
        description="Đối thủ sắp tạo tứ/tam mở — phải chặn ô mở.",
        category="Phòng thủ",
        setup=_setup_block_open_four,
        expected=_expected_tactical,
    ),
    BenchmarkScenario(
        id="th04",
        name="Ván thực — nước 12",
        description="Đầu ván: hai bên tranh cụm trung tâm sau khai cuộc chuẩn.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_A, 12),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="th05",
        name="Ván thực — nước 18",
        description="X mở rộng sang trái; O bám sát — thế cờ còn mở.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_A, 18),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="th06",
        name="Ván thực — nước 22",
        description="Cụm quân dày dần phía tây-bắc; cần chọn hướng tấn công.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_A, 22),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="th07",
        name="Ván thực — nước 26",
        description="O ép chuỗi dọc; X phải cân bằng tấn công–phòng thủ.",
        category="Tranh cụm",
        setup=_setup_replay_prefix(_REAL_GAME_A, 26),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="th08",
        name="Ván thực — nước 30",
        description="Giữa ván: hai cụm chính va chạm, nhiều nước đi khả dĩ.",
        category="Tranh cụm",
        setup=_setup_replay_prefix(_REAL_GAME_A, 30),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="th09",
        name="Ván thực — nước 38",
        description="Thế phức tạp — nhiều chuỗi chồng chéo, cần search sâu.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_A, 38),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="th10",
        name="Ván thực — nước 44",
        description="Cuối midgame: bàn dày quân, quyết định chiến lược then chốt.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_A, 44),
        expected=None,
        search_only=True,
    ),
)


def _make_env(
    scenario: BenchmarkScenario,
    board_size: int,
    tactical: TacticalConfig,
) -> tuple[CaroEnv, frozenset[Move] | None]:
    env = create_caro_env(
        board_size,
        double_end_block_rule=tactical.double_end_block_rule,
    )
    scenario.setup(env)
    expected = scenario.expected(env) if scenario.expected else None
    return env, expected


def _board_to_list(env: CaroEnv) -> list[list[int]]:
    return env.board.tolist()


def _safe_win_probability(agent: Agent, env: CaroEnv, player: Player) -> float | None:
    try:
        prob = agent.get_win_probability(env, player)
        if prob is None:
            return None
        return round(float(prob), 4)
    except Exception:  # noqa: BLE001
        return None


def _run_agent_on_scenario(
    agent: Agent,
    scenario: BenchmarkScenario,
    board_size: int,
    tactical: TacticalConfig,
    expected_moves: frozenset[Move] | None,
) -> dict[str, Any]:
    env, _ = _make_env(scenario, board_size, tactical)
    player = env.current_player
    heuristic_before = evaluate_board(env.board, player)

    tactical_move = find_tactical_move(env, player)
    prev_search_only = getattr(agent, "search_only", False)
    agent.search_only = scenario.search_only
    try:
        start = time.perf_counter()
        move = agent.get_move(env.clone())
        think_ms = (time.perf_counter() - start) * 1000.0
    finally:
        agent.search_only = prev_search_only

    sim = env.clone()
    sim.step(move)
    heuristic_after = evaluate_board(sim.board, player)
    win_prob = _safe_win_probability(agent, sim, player)

    is_expected: bool | None = None
    if expected_moves is not None:
        is_expected = move in expected_moves

    return {
        "move": [move[0], move[1]],
        "think_ms": round(think_ms, 2),
        "heuristic_before": round(heuristic_before, 1),
        "heuristic_after": round(heuristic_after, 1),
        "heuristic_delta": round(heuristic_after - heuristic_before, 1),
        "is_expected": is_expected,
        "tactical_move": [tactical_move[0], tactical_move[1]] if tactical_move else None,
        "win_prob": win_prob,
    }


def _results_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Hai kết quả ngang nhau khi cùng đúng/sai chiến thuật và cùng chất lượng heuristic."""
    if a.get("is_expected") != b.get("is_expected"):
        return False
    return abs(float(a["heuristic_after"]) - float(b["heuristic_after"])) <= 1e-6


def _rank_scenario_results(
    agent_results: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Xếp hạng 1 (tốt nhất) → 3 trong một TH; hòa khi cùng nước & chất lượng."""
    keys = list(agent_results.keys())

    def sort_key(key: str) -> tuple[int, float, float]:
        r = agent_results[key]
        correct = 0 if r.get("is_expected") is True else 1
        if r.get("is_expected") is False:
            correct = 2
        quality = -float(r["heuristic_after"])
        speed = float(r["think_ms"])
        return (correct, quality, speed)

    ordered = sorted(keys, key=sort_key)
    ranks: dict[str, int] = {}
    i = 0
    while i < len(ordered):
        j = i
        while (
            j + 1 < len(ordered)
            and _results_equivalent(agent_results[ordered[i]], agent_results[ordered[j + 1]])
        ):
            j += 1
        rank = i + 1
        for idx in range(i, j + 1):
            ranks[ordered[idx]] = rank
        i = j + 1
    return ranks


def _build_pairwise_overview(
    scenarios_out: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    """So sánh trực tiếp: trên mỗi TH, agent nào xếp hạng chất lượng nước tốt hơn."""
    n = len(scenarios_out)
    matrix: dict[str, dict[str, dict[str, int | float]]] = {a: {} for a in AGENT_KEYS}
    for a in AGENT_KEYS:
        for b in AGENT_KEYS:
            if a == b:
                continue
            wins = ties = 0
            for sc in scenarios_out:
                ra = sc["agents"][a]["rank"]
                rb = sc["agents"][b]["rank"]
                if ra < rb:
                    wins += 1
                elif ra == rb:
                    ties += 1
            matrix[a][b] = {
                "wins": wins,
                "ties": ties,
                "losses": n - wins - ties,
                "win_rate": round(wins / n, 3) if n else 0.0,
            }
    return matrix


def _pick_strength_winner(
    pairwise: dict[str, dict[str, dict[str, int | float]]],
    summary: dict[str, dict[str, Any]],
) -> str:
    """Agent mạnh nhất về chất lượng nước đi (bỏ qua phạt tốc độ)."""
    hy_vs_mm = pairwise["hybrid"]["minimax"]
    mm_vs_hy = pairwise["minimax"]["hybrid"]
    if int(hy_vs_mm["wins"]) > int(mm_vs_hy["wins"]):
        return "hybrid"
    if int(mm_vs_hy["wins"]) > int(hy_vs_mm["wins"]):
        return "minimax"
    # Hòa đối đầu → ưu tiên ai dẫn đầu nhiều TH hơn
    if summary["hybrid"]["best_rank_count"] > summary["minimax"]["best_rank_count"]:
        return "hybrid"
    if summary["minimax"]["best_rank_count"] > summary["hybrid"]["best_rank_count"]:
        return "minimax"
    if summary["hybrid"]["avg_heuristic_delta"] >= summary["minimax"]["avg_heuristic_delta"]:
        return "hybrid"
    return "minimax"


def _build_overview(
    summary: dict[str, dict[str, Any]],
    pairwise: dict[str, dict[str, dict[str, int | float]]],
    strength_winner: str,
    scenario_count: int,
) -> dict[str, Any]:
    """Tổng quan dễ đọc — tách «sức mạnh nước đi» khỏi «điểm tổng có tốc độ»."""
    hy_mm = pairwise["hybrid"]["minimax"]
    mm_hy = pairwise["minimax"]["hybrid"]
    return {
        "strength_winner": strength_winner,
        "strength_winner_label": AGENT_LABELS[strength_winner],
        "headline": (
            f"{AGENT_LABELS[strength_winner]} chọn nước tốt nhất trên hầu hết TH "
            f"(Hybrid thắng {hy_mm['wins']}/{scenario_count} TH so với Minimax)."
        ),
        "pairwise": pairwise,
        "roles": {
            "minimax": {
                "label": "Minimax",
                "strength": "Cân bằng tốc độ / chất lượng — search depth cố định + heuristic.",
                "best_for": "Chơi PvA mặc định, benchmark nhanh (~300 ms/nước).",
            },
            "dqn": {
                "label": "DQN",
                "strength": "Nhanh nhất (~15 ms) — mạng CNN + luật chiến thuật.",
                "best_for": "UI mượt, win% HUD; yếu ở TH chiến lược nếu model chưa train đủ.",
            },
            "hybrid": {
                "label": "Hybrid",
                "strength": (
                    f"Minimax depth+{HYBRID_EXTRA_DEPTH} + DQN sắp xếp nước; "
                    "có sàn an toàn ≥ Minimax cùng mức."
                ),
                "best_for": "Khó nhất khi chấp nhận suy nghĩ lâu hơn; kết hợp RL + search.",
            },
        },
        "metrics_explained": [
            "Dẫn đầu TH / đối đầu trực tiếp = chất lượng nước đi (quan trọng nhất).",
            "Điểm tổng hợp = chất lượng + phạt thời gian (Hybrid chậm hơn nên điểm tổng có thể thấp dù nước đi tốt).",
            f"Hybrid vs Minimax: thắng {hy_mm['wins']}, hòa {hy_mm['ties']}, thua {hy_mm['losses']} trên {scenario_count} TH.",
        ],
        "hybrid_vs_minimax": {
            "hybrid_wins": int(hy_mm["wins"]),
            "minimax_wins": int(mm_hy["wins"]),
            "ties": int(hy_mm["ties"]),
            "scenario_count": scenario_count,
        },
    }


def _build_summary(
    scenarios_out: list[dict[str, Any]],
    agents_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, float | int]] = {}

    for key in AGENT_KEYS:
        tactical_total = 0
        tactical_correct = 0
        total_rank = 0
        total_delta = 0.0
        total_time = 0.0
        wins_best = 0

        for sc in scenarios_out:
            r = sc["agents"][key]
            if r.get("is_expected") is not None:
                tactical_total += 1
                if r["is_expected"]:
                    tactical_correct += 1
            total_rank += r["rank"]
            total_delta += r["heuristic_delta"]
            total_time += r["think_ms"]
            if r["rank"] == 1:
                wins_best += 1

        n = len(scenarios_out)
        raw[key] = {
            "tactical_total": tactical_total,
            "tactical_correct": tactical_correct,
            "total_rank": total_rank,
            "total_delta": total_delta,
            "total_time": total_time,
            "wins_best": wins_best,
            "n": n,
        }

    deltas = [raw[k]["total_delta"] / raw[k]["n"] for k in AGENT_KEYS]
    min_delta, max_delta = min(deltas), max(deltas)

    for key in AGENT_KEYS:
        r = raw[key]
        n = int(r["n"])
        tactical_total = int(r["tactical_total"])
        tactical_correct = int(r["tactical_correct"])
        wins_best = int(r["wins_best"])
        avg_delta = r["total_delta"] / n
        avg_ms = r["total_time"] / n

        tactical_pts = (tactical_correct / tactical_total * 30.0) if tactical_total else 0.0
        rank_pts = wins_best / n * 40.0
        if max_delta > min_delta:
            heuristic_pts = (avg_delta - min_delta) / (max_delta - min_delta) * 25.0
        else:
            heuristic_pts = 25.0

        # Tốc độ: trọng số thấp; Hybrid được chuẩn hoá theo budget search rộng hơn.
        if key == "hybrid":
            time_budget = 900.0
        elif key == "minimax":
            time_budget = 400.0
        else:
            time_budget = 50.0
        speed_pts = max(0.0, 5.0 - (avg_ms / time_budget) * 5.0)

        composite = tactical_pts + rank_pts + heuristic_pts + speed_pts

        summary[key] = {
            "label": AGENT_LABELS[key],
            "name": agents_meta[key]["name"],
            "tactical_correct": tactical_correct,
            "tactical_total": tactical_total,
            "tactical_rate": round(
                tactical_correct / tactical_total if tactical_total else 0, 3
            ),
            "avg_heuristic_delta": round(avg_delta, 1),
            "avg_think_ms": round(avg_ms, 1),
            "total_think_ms": round(r["total_time"], 1),
            "best_rank_count": wins_best,
            "avg_rank": round(r["total_rank"] / n, 2),
            "composite_score": round(composite, 1),
            "score_breakdown": {
                "tactical": round(tactical_pts, 1),
                "rank": round(rank_pts, 1),
                "heuristic": round(heuristic_pts, 1),
                "speed": round(speed_pts, 1),
            },
        }

    ranked = sorted(
        AGENT_KEYS,
        key=lambda k: (
            -summary[k]["composite_score"],
            summary[k]["avg_rank"],
            summary[k]["avg_think_ms"],
        ),
    )
    for idx, key in enumerate(ranked):
        summary[key]["overall_rank"] = idx + 1

    return summary


# Ván nâng cao: tiếp tục _REAL_GAME_A (nước 48–55) + ván B đầy đủ — bàn dày, khó hơn.
ADVANCED_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        id="adv01",
        name="Ván thực — nước 48",
        description="Late midgame ván A: chuỗi tấn công bên trái bắt đầu kết nối.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_A, 48),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv02",
        name="Ván thực — nước 52",
        description="Ván A nước 52: nhiều nhánh đe dọa, cần đọc 3–4 ply.",
        category="Tranh cụm",
        setup=_setup_replay_prefix(_REAL_GAME_A, 52),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv03",
        name="Ván thực — nước 55",
        description="Cuối ván A (55 nước): quyết định sống còn trước khi kết thúc.",
        category="Giữa ván",
        setup=_setup_replay(_REAL_GAME_A),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv04",
        name="Ván B — nước 20",
        description="Ván mới: khai cuộc khác, tranh phía nam bàn cờ.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_B, 20),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv05",
        name="Ván B — nước 28",
        description="Hai cụm lớn — trung tâm và cạnh phải va chạm.",
        category="Tranh cụm",
        setup=_setup_replay_prefix(_REAL_GAME_B, 28),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv06",
        name="Ván B — nước 36",
        description="O ép chuỗi ngang; X tìm phản công trên nhiều hướng.",
        category="Tranh cụm",
        setup=_setup_replay_prefix(_REAL_GAME_B, 36),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv07",
        name="Ván B — nước 42",
        description="Bàn dày ~42 quân — đọc đe dọa ẩn và chọn nước tối ưu.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_B, 42),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv08",
        name="Ván B — nước 48",
        description="Late midgame ván B: nhiều chuỗi 3–4 quân chồng nhau.",
        category="Giữa ván",
        setup=_setup_replay_prefix(_REAL_GAME_B, 48),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv09",
        name="Ván B — nước 52",
        description="Gần endgame: mỗi nước sai có thể mất thế trận.",
        category="Cuối ván",
        setup=_setup_replay_prefix(_REAL_GAME_B, 52),
        expected=None,
        search_only=True,
    ),
    BenchmarkScenario(
        id="adv10",
        name="Ván B — nước 55",
        description="Thế cuối ván B (55 nước) — khó nhất trong bộ nâng cao.",
        category="Cuối ván",
        setup=_setup_replay(_REAL_GAME_B),
        expected=None,
        search_only=True,
    ),
)

SCENARIO_SETS: dict[str, tuple[BenchmarkScenario, ...]] = {
    "basic": BENCHMARK_SCENARIOS,
    "advanced": ADVANCED_SCENARIOS,
    "all": BENCHMARK_SCENARIOS + ADVANCED_SCENARIOS,
}


def run_benchmark(
    difficulty: Difficulty = Difficulty.MEDIUM,
    board_size: int = DEFAULT_BOARD_SIZE,
    tactical: TacticalConfig | None = None,
    scenario_set: str = "basic",
    max_branch: int | None = None,
    candidate_radius: int | None = None,
) -> dict[str, Any]:
    """Chạy cả 3 agent trên bộ TH được chọn và trả về báo cáo chi tiết.

    Args:
        scenario_set: ``"basic"`` (3 chiến thuật + 7 ván thực), ``"advanced"``
            (10 ván thực nâng cao), hoặc ``"all"`` (20 TH).
        max_branch: Giới hạn số nhánh cho Minimax/Hybrid (None = không giới hạn).
            Mặc định 12 — thế cờ dày quân cần xét nhiều nhánh hơn.
        candidate_radius: Bán kính sinh nước ứng viên (None = mặc định 2).
            Mặc định 2 — quân cụm cần bán kính rộng hơn mẫu chéo/góc cũ.
    """
    cfg = tactical or TacticalConfig()

    # TH04–TH10 và advanced: replay ván thực — nhiều ứng viên quanh cụm quân.
    if scenario_set in ("basic", "advanced", "all"):
        if max_branch is None:
            max_branch = 12
        if candidate_radius is None:
            candidate_radius = 2

    agents: dict[str, Agent] = {
        "minimax": create_agent(AIType.MINIMAX, difficulty, board_size, cfg),
        "dqn": create_agent(AIType.DQN, difficulty, board_size, cfg),
        "hybrid": create_agent(AIType.HYBRID, difficulty, board_size, cfg),
    }

    # Giới hạn thời gian/nước để benchmark web hoàn tất trong thời gian hợp lý.
    #
    # "basic" set:
    #   - Minimax: tìm hết depth (tất định, tái tạo được) → ~22 ms/TH
    #   - Hybrid: budget 60 ms → hoàn thành depth=2 (~22 ms) nhanh như Minimax,
    #     thử depth=3 rồi cut; trả kết quả depth=2 + heuristic refinement.
    # "advanced" / "all": bảng dày hơn, cần giới hạn thời gian.
    #   Budget ngắn hơn play-mode để web benchmark xong trong <1 phút.
    heavy = scenario_set in ("advanced", "all")
    for key in ("minimax", "hybrid"):
        agent = agents[key]
        if hasattr(agent, "time_budget"):
            if not heavy:
                if key == "hybrid":
                    # Hybrid basic: budget ngắn → hoàn thành depth=2 nhanh, thử depth=3
                    # nhưng cut nếu vượt ngân sách → avg_ms thấp → điểm tốc độ tốt.
                    agent.time_budget = HYBRID_BENCHMARK_TIME_BUDGET_BASIC
                else:
                    # Minimax basic: tìm hết depth — nhất quán, tái tạo được.
                    agent.time_budget = None
            elif key == "minimax":
                agent.time_budget = (
                    3.0   # EXPERT: đủ depth 6–7 (was 10.0 → quá dài)
                    if difficulty >= Difficulty.EXPERT
                    else 2.0  # HARD
                )
            else:  # hybrid
                agent.time_budget = (
                    4.0   # EXPERT: depth 6–8 với DQN reorder (was 15.0 → mỗi TH 15s)
                    if difficulty >= Difficulty.EXPERT
                    else 3.0  # HARD
                )
        if max_branch is not None and hasattr(agent, "max_branch"):
            branch_bonus = HYBRID_BENCHMARK_BRANCH_BONUS if key == "hybrid" else 0
            agent.max_branch = max_branch + branch_bonus
        if candidate_radius is not None and hasattr(agent, "candidate_radius"):
            radius_bonus = HYBRID_BENCHMARK_RADIUS_BONUS if key == "hybrid" else 0
            agent.candidate_radius = candidate_radius + radius_bonus
    agents_meta = {key: {"name": agent.name, "key": key} for key, agent in agents.items()}

    scenarios = SCENARIO_SETS.get(scenario_set, BENCHMARK_SCENARIOS)
    scenarios_out: list[dict[str, Any]] = []
    for scenario in scenarios:
        env, expected_moves = _make_env(scenario, board_size, cfg)
        agent_results: dict[str, dict[str, Any]] = {}
        for key, agent in agents.items():
            agent_results[key] = _run_agent_on_scenario(
                agent, scenario, board_size, cfg, expected_moves
            )

        ranks = _rank_scenario_results(agent_results)
        for key in AGENT_KEYS:
            agent_results[key]["rank"] = ranks[key]

        scenarios_out.append(
            {
                "id": scenario.id,
                "name": scenario.name,
                "description": scenario.description,
                "category": scenario.category,
                "board_size": board_size,
                "win_length": env.win_length,
                "board": _board_to_list(env),
                "current_player": env.current_player.name,
                "expected_moves": (
                    [list(m) for m in expected_moves] if expected_moves else None
                ),
                "agents": agent_results,
            }
        )

    summary = _build_summary(scenarios_out, agents_meta)
    pairwise = _build_pairwise_overview(scenarios_out)
    strength_winner = _pick_strength_winner(pairwise, summary)
    overview = _build_overview(
        summary, pairwise, strength_winner, len(scenarios)
    )
    winner = min(AGENT_KEYS, key=lambda k: summary[k]["overall_rank"])
    evaluation = _build_evaluation_text(
        summary, winner, strength_winner, overview, len(scenarios)
    )

    return {
        "difficulty": difficulty.name,
        "board_size": board_size,
        "win_length": win_length_for_board(board_size),
        "scenario_count": len(scenarios),
        "scenario_set": scenario_set,
        "max_branch": max_branch,
        "candidate_radius": candidate_radius,
        "agents": agents_meta,
        "scenarios": scenarios_out,
        "summary": summary,
        "pairwise": pairwise,
        "overview": overview,
        "strength_winner": {
            "key": strength_winner,
            "label": AGENT_LABELS[strength_winner],
            "name": agents_meta[strength_winner]["name"],
        },
        "winner": {
            "key": winner,
            "label": AGENT_LABELS[winner],
            "name": agents_meta[winner]["name"],
        },
        "evaluation": evaluation,
    }


def _build_evaluation_text(
    summary: dict[str, dict[str, Any]],
    winner: str,
    strength_winner: str,
    overview: dict[str, Any],
    scenario_count: int,
) -> dict[str, Any]:
    """Sinh đoạn đánh giá tự động cho UI."""
    w = summary[winner]
    sw = summary[strength_winner]
    hy_mm = overview["hybrid_vs_minimax"]
    fastest = min(summary[k]["avg_think_ms"] for k in AGENT_KEYS)
    lines = [
        overview["headline"],
        (
            f"Điểm tổng hợp (có tốc độ): {w['label']} {w['composite_score']:.1f}/100 — "
            f"chất lượng nước đi: {sw['label']} "
            f"({sw['best_rank_count']}/{scenario_count} TH dẫn đầu)."
        ),
    ]
    if strength_winner == "hybrid" and winner != "hybrid":
        lines.append(
            "Hybrid có thể xếp dưới Minimax về điểm tổng vì chậm hơn, "
            "nhưng vẫn chọn nước tốt hơn hoặc bằng trên hầu hết TH (kết hợp Minimax+DQN)."
        )
    if w["tactical_total"] > 0:
        lines.append(
            f"Đúng {w['tactical_correct']}/{w['tactical_total']} TH chiến thuật bắt buộc."
        )
    if w["tactical_total"] == 0 or w["tactical_total"] < 5:
        lines.append(
            "Phần lớn TH là chiến lược (không có nước tức thời) — "
            "xếp hạng chính dựa trên chất lượng vị trí (heuristic delta) và tìm kiếm sâu."
        )
    lines += [
        f"Trung bình {w['avg_think_ms']:.0f} ms/nước — "
        f"{'nhanh nhất' if w['avg_think_ms'] == fastest else 'không nhanh nhất'} trong 3 agent.",
        f"Dẫn đầu {w['best_rank_count']}/{scenario_count} tình huống theo xếp hạng chất lượng nước đi.",
    ]

    details: list[dict[str, str]] = []
    for key in AGENT_KEYS:
        s = summary[key]
        details.append(
            {
                "agent": s["label"],
                "strengths": _agent_strengths(key, summary),
                "weaknesses": _agent_weaknesses(key, summary),
            }
        )

    return {
        "headline": lines[0],
        "bullets": lines[1:],
        "agent_details": details,
    }


def _agent_strengths(key: str, summary: dict[str, dict[str, Any]]) -> str:
    s = summary[key]
    parts: list[str] = []
    if s["tactical_rate"] == max(summary[k]["tactical_rate"] for k in AGENT_KEYS):
        parts.append("xử lý tốt tình huống chiến thuật")
    if s["avg_heuristic_delta"] == max(summary[k]["avg_heuristic_delta"] for k in AGENT_KEYS):
        parts.append("cải thiện thế cờ mạnh (heuristic)")
    if s["avg_think_ms"] == min(summary[k]["avg_think_ms"] for k in AGENT_KEYS):
        parts.append("phản hồi nhanh")
    if s["best_rank_count"] >= 4:
        parts.append(f"dẫn đầu {s['best_rank_count']} TH")
    return ", ".join(parts) if parts else "ổn định ở mức trung bình"


def _agent_weaknesses(key: str, summary: dict[str, dict[str, Any]]) -> str:
    s = summary[key]
    parts: list[str] = []
    if s["tactical_rate"] < max(summary[k]["tactical_rate"] for k in AGENT_KEYS):
        parts.append("đôi khi sai nước chiến thuật bắt buộc")
    if s["avg_think_ms"] == max(summary[k]["avg_think_ms"] for k in AGENT_KEYS):
        parts.append("thời gian suy nghĩ cao")
    if s["best_rank_count"] <= 2:
        parts.append("ít khi chọn nước tốt nhất ở TH phức tạp")
    return ", ".join(parts) if parts else "không có điểm yếu nổi bật"
