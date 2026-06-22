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


def _setup_advanced(pieces: list[tuple[int, int, Player]]) -> ScenarioSetup:
    """Tạo hàm setup từ danh sách quân cờ tường minh (grid spacing≥3)."""

    def _setup(env: CaroEnv) -> None:
        env.reset()
        last_o: Move | None = None
        count = 0
        for r, c, p in pieces:
            if env.in_bounds(r, c) and env.board[r, c] == Player.EMPTY:
                env.board[r, c] = p
                count += 1
                if p == Player.O:
                    last_o = (r, c)
        _set_turn(env, Player.X, count, last_o)

    return _setup


# ---------------------------------------------------------------------------
# TH04–TH10: Tình huống chiến lược (không có nước chiến thuật tức thì).
# Mọi quân cùng màu cách nhau ≥3 theo mọi hướng → find_tactical_move = None.
# Agents phải dùng tìm kiếm sâu / Q-value / heuristic để phân biệt nhau.
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
        name="Bốn góc trong",
        description="X chiếm 4 góc nội (4,4)–(10,10); O giữ 4 đỉnh ngoài — tranh giành trung tâm.",
        category="Chiến lược",
        setup=_setup_advanced([
            (4, 4, Player.X), (4, 10, Player.X), (10, 4, Player.X), (10, 10, Player.X),
            (1, 7, Player.O), (7, 1, Player.O), (7, 13, Player.O), (13, 7, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="th05",
        name="Chéo chính vs chéo phụ",
        description="X chiếm đường chéo chính; O chiếm đường chéo phụ — va chạm tại tâm.",
        category="Đường chéo",
        setup=_setup_advanced([
            (1, 1, Player.X), (4, 4, Player.X), (7, 7, Player.X), (10, 10, Player.X), (13, 13, Player.X),
            (1, 13, Player.O), (4, 10, Player.O), (10, 4, Player.O), (13, 1, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="th06",
        name="Hoa tâm vs bốn góc",
        description="X giữ tâm hình hoa (7,7 và 4 cánh); O chiếm 4 góc xa và (4,4).",
        category="Kiểm soát",
        setup=_setup_advanced([
            (7, 7, Player.X), (4, 7, Player.X), (10, 7, Player.X), (7, 4, Player.X), (7, 10, Player.X),
            (1, 1, Player.O), (1, 13, Player.O), (13, 1, Player.O), (13, 13, Player.O), (4, 4, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="th07",
        name="Hàng ngang vs cột dọc",
        description="X giữ hai hàng ngang (hàng 4 và 10); O giữ hai cột dọc (cột 4 và 10).",
        category="Đối xứng",
        setup=_setup_advanced([
            (4, 1, Player.X), (4, 7, Player.X), (4, 13, Player.X), (10, 4, Player.X), (10, 10, Player.X),
            (1, 4, Player.O), (7, 4, Player.O), (13, 4, Player.O), (4, 10, Player.O), (10, 7, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="th08",
        name="Cụm trung tâm vs vành ngoài",
        description="X tập trung vùng giữa; O bao vây từ ngoài — ai kiểm soát được không gian?",
        category="Kiểm soát",
        setup=_setup_advanced([
            (4, 4, Player.X), (4, 10, Player.X), (7, 7, Player.X), (10, 4, Player.X), (10, 10, Player.X),
            (1, 7, Player.O), (7, 1, Player.O), (7, 13, Player.O), (13, 7, Player.O), (4, 7, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="th09",
        name="Phân cực Đông-Tây",
        description="X chiếm toàn bộ cột Tây (cột 1); O chiếm toàn bộ cột Đông (cột 13).",
        category="Phân chia",
        setup=_setup_advanced([
            (1, 1, Player.X), (4, 1, Player.X), (7, 1, Player.X), (10, 1, Player.X), (13, 1, Player.X),
            (1, 13, Player.O), (4, 13, Player.O), (7, 13, Player.O), (10, 13, Player.O), (13, 13, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="th10",
        name="Thế cờ phức hợp",
        description="X và O đan xen 14 quân — thế trận giữa ván đòi hỏi đánh giá chiến lược sâu.",
        category="Phức hợp",
        setup=_setup_advanced([
            (1, 4, Player.X), (4, 1, Player.X), (4, 7, Player.X), (7, 4, Player.X),
            (7, 7, Player.X), (10, 7, Player.X), (13, 4, Player.X),
            (1, 10, Player.O), (4, 13, Player.O), (7, 10, Player.O), (7, 13, Player.O),
            (10, 10, Player.O), (13, 7, Player.O), (13, 13, Player.O),
        ]),
        expected=None,
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
    start = time.perf_counter()
    move = agent.get_move(env.clone())
    think_ms = (time.perf_counter() - start) * 1000.0

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
    """Hai kết quả coi là ngang nhau khi cùng nước đi và cùng chất lượng."""
    if tuple(a["move"]) != tuple(b["move"]):
        return False
    if a.get("is_expected") != b.get("is_expected"):
        return False
    return float(a["heuristic_after"]) == float(b["heuristic_after"])


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


def _build_summary(
    scenarios_out: list[dict[str, Any]],
    agents_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
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
        composite = (
            (tactical_correct / tactical_total * 40 if tactical_total else 0)
            + (wins_best / n * 35)
            + max(0, 25 - total_time / n / 100)
        )

        summary[key] = {
            "label": AGENT_LABELS[key],
            "name": agents_meta[key]["name"],
            "tactical_correct": tactical_correct,
            "tactical_total": tactical_total,
            "tactical_rate": round(
                tactical_correct / tactical_total if tactical_total else 0, 3
            ),
            "avg_heuristic_delta": round(total_delta / n, 1),
            "avg_think_ms": round(total_time / n, 1),
            "total_think_ms": round(total_time, 1),
            "best_rank_count": wins_best,
            "avg_rank": round(total_rank / n, 2),
            "composite_score": round(composite, 1),
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


# Tình huống nâng cao: không có nước chiến thuật tức thời (aggressive=True vẫn = None).
# Mọi quân cùng màu cách nhau ≥3 ô theo mọi hướng → không thể tạo tam/tứ mở ngay.
# Agents PHẢI dùng tìm kiếm sâu / Q-value / heuristic để phân biệt nhau.
ADVANCED_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        id="adv01",
        name="Cánh cung đôi",
        description="X chiếm góc trên-trái, O chiếm góc dưới-phải — tranh giành trung tâm.",
        category="Chiến lược",
        setup=_setup_advanced([
            (1, 1, Player.X), (1, 7, Player.X), (4, 4, Player.X), (4, 10, Player.X), (7, 1, Player.X),
            (13, 13, Player.O), (13, 7, Player.O), (10, 10, Player.O), (10, 4, Player.O), (7, 13, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv02",
        name="Đối đầu chéo",
        description="X ở góc phải trên, O ở góc trái dưới — ai vươn tới trung tâm trước?",
        category="Chiến lược",
        setup=_setup_advanced([
            (1, 13, Player.X), (1, 7, Player.X), (4, 10, Player.X), (4, 7, Player.X), (7, 13, Player.X),
            (13, 1, Player.O), (13, 7, Player.O), (10, 4, Player.O), (10, 7, Player.O), (7, 1, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv03",
        name="Chữ thập vs bốn góc",
        description="X giữ đường chữ thập (hàng+cột 7), O chiếm 4 góc và trung điểm.",
        category="Kiểm soát",
        setup=_setup_advanced([
            (1, 7, Player.X), (4, 7, Player.X), (7, 1, Player.X), (7, 4, Player.X),
            (7, 10, Player.X), (7, 13, Player.X), (10, 7, Player.X), (13, 7, Player.X),
            (1, 1, Player.O), (1, 13, Player.O), (13, 1, Player.O), (13, 13, Player.O),
            (4, 4, Player.O), (4, 10, Player.O), (10, 4, Player.O), (10, 10, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv04",
        name="Đường chéo kép + tâm",
        description="X chiếm đường chéo chính và tâm; O chiếm đường chéo phụ.",
        category="Đường chéo",
        setup=_setup_advanced([
            (1, 1, Player.X), (4, 4, Player.X), (7, 7, Player.X), (10, 10, Player.X), (13, 13, Player.X),
            (1, 13, Player.O), (4, 10, Player.O), (10, 4, Player.O), (13, 1, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv05",
        name="Tứ phương vs cụm trung",
        description="X trải đều 4 phương; O tập trung ở vùng trung tâm.",
        category="Kiểm soát",
        setup=_setup_advanced([
            (1, 4, Player.X), (4, 1, Player.X), (4, 13, Player.X), (1, 10, Player.X),
            (7, 7, Player.X), (13, 4, Player.X), (10, 1, Player.X),
            (10, 13, Player.O), (13, 10, Player.O), (10, 7, Player.O),
            (7, 10, Player.O), (7, 4, Player.O), (4, 7, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv06",
        name="Hàng rào vs cột dọc",
        description="X giữ hai hàng ngang (hàng 4 và 10); O giữ hai cột dọc (cột 4 và 10).",
        category="Đối xứng",
        setup=_setup_advanced([
            (4, 1, Player.X), (4, 7, Player.X), (4, 13, Player.X),
            (10, 1, Player.X), (10, 7, Player.X), (10, 13, Player.X),
            (1, 4, Player.O), (7, 4, Player.O), (13, 4, Player.O),
            (1, 10, Player.O), (7, 10, Player.O), (13, 10, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv07",
        name="Vây hãm tâm bão",
        description="O bao vây vòng trong (khoảng cách 3–4 từ tâm); X giữ vành ngoài.",
        category="Bao vây",
        setup=_setup_advanced([
            (4, 4, Player.O), (4, 7, Player.O), (4, 10, Player.O),
            (7, 4, Player.O), (7, 10, Player.O),
            (10, 4, Player.O), (10, 7, Player.O), (10, 10, Player.O),
            (1, 1, Player.X), (1, 7, Player.X), (1, 13, Player.X),
            (7, 1, Player.X), (7, 13, Player.X),
            (13, 1, Player.X), (13, 7, Player.X), (13, 13, Player.X),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv08",
        name="Phân cực Đông-Tây",
        description="X chiếm cột phía Tây (cột 1 và 4); O chiếm cột phía Đông (cột 10 và 13).",
        category="Phân chia",
        setup=_setup_advanced([
            (1, 1, Player.X), (4, 1, Player.X), (7, 1, Player.X), (10, 1, Player.X), (13, 1, Player.X),
            (4, 4, Player.X), (10, 4, Player.X),
            (1, 13, Player.O), (4, 13, Player.O), (7, 13, Player.O), (10, 13, Player.O), (13, 13, Player.O),
            (4, 10, Player.O), (10, 10, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv09",
        name="Mê trận đan xen",
        description="X và O đan xen ở toàn bộ bàn cờ — thế trận phức tạp nhất.",
        category="Phức hợp",
        setup=_setup_advanced([
            (1, 1, Player.X), (1, 7, Player.X), (1, 13, Player.X),
            (4, 4, Player.X), (4, 10, Player.X),
            (7, 7, Player.X),
            (10, 4, Player.X), (10, 10, Player.X),
            (13, 1, Player.X), (13, 7, Player.X), (13, 13, Player.X),
            (4, 7, Player.O), (7, 4, Player.O), (7, 10, Player.O), (10, 7, Player.O),
            (1, 4, Player.O), (1, 10, Player.O), (13, 4, Player.O), (13, 10, Player.O),
        ]),
        expected=None,
    ),
    BenchmarkScenario(
        id="adv10",
        name="Tốc chiến hội tụ",
        description="Thế cờ dày đặc nhất — X và O đều đang tiến về tâm bàn.",
        category="Phức hợp",
        setup=_setup_advanced([
            (1, 4, Player.X), (4, 1, Player.X), (4, 7, Player.X), (7, 4, Player.X),
            (7, 7, Player.X), (10, 7, Player.X), (13, 4, Player.X), (7, 1, Player.X),
            (1, 10, Player.O), (4, 13, Player.O), (4, 10, Player.O), (7, 13, Player.O),
            (7, 10, Player.O), (10, 10, Player.O), (13, 13, Player.O), (10, 13, Player.O),
        ]),
        expected=None,
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
        scenario_set: ``"basic"`` (3 chiến thuật + 7 chiến lược), ``"advanced"``
            (10 TH chiến lược phức tạp hơn), hoặc ``"all"`` (20 TH).
        max_branch: Giới hạn số nhánh cho Minimax/Hybrid (None = không giới hạn).
            Mặc định 15 cho basic và advanced (giảm thời gian → ~15–35 giây).
        candidate_radius: Bán kính sinh nước ứng viên (None = mặc định 2).
            Mặc định 1 cho basic và advanced (giảm số ứng viên ~2×).
    """
    cfg = tactical or TacticalConfig()

    # TH04–TH10 (basic) và tất cả advanced có nhiều candidates phân tán;
    # max_branch=10 → MEDIUM ~10s, HARD ~25s (chấp nhận được).
    if scenario_set in ("basic", "advanced", "all"):
        if max_branch is None:
            max_branch = 10
        if candidate_radius is None:
            candidate_radius = 1

    agents: dict[str, Agent] = {
        "minimax": create_agent(AIType.MINIMAX, difficulty, board_size, cfg),
        "dqn": create_agent(AIType.DQN, difficulty, board_size, cfg),
        "hybrid": create_agent(AIType.HYBRID, difficulty, board_size, cfg),
    }

    # Apply search-width overrides so all 3 agents use the same budget.
    for key in ("minimax", "hybrid"):
        agent = agents[key]
        if max_branch is not None and hasattr(agent, "max_branch"):
            agent.max_branch = max_branch
        if candidate_radius is not None and hasattr(agent, "candidate_radius"):
            agent.candidate_radius = candidate_radius
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
    winner = min(AGENT_KEYS, key=lambda k: summary[k]["overall_rank"])
    evaluation = _build_evaluation_text(summary, winner)

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
        "winner": {
            "key": winner,
            "label": AGENT_LABELS[winner],
            "name": agents_meta[winner]["name"],
        },
        "evaluation": evaluation,
    }


def _build_evaluation_text(summary: dict[str, dict[str, Any]], winner: str) -> dict[str, Any]:
    """Sinh đoạn đánh giá tự động cho UI."""
    w = summary[winner]
    fastest = min(summary[k]["avg_think_ms"] for k in AGENT_KEYS)
    total_scenarios = sum(summary[k]["best_rank_count"] for k in AGENT_KEYS)
    lines = [
        f"{w['label']} xếp hạng tổng thể cao nhất với điểm tổng hợp {w['composite_score']:.1f}/100.",
    ]
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
        f"Dẫn đầu {w['best_rank_count']}/{total_scenarios} tình huống theo xếp hạng chất lượng nước đi.",
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
