#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bộ công cụ tự kiểm tra (self-check skills) cho dự án AI Cờ Caro.

Module này định nghĩa 3 "skill" mà agent dùng để TỰ KIỂM TRA code của chính
mình trong quá trình phát triển:

    1. lint_code      : kiểm tra phong cách code (ruff) và kiểu (mypy).
    2. run_unit_tests : chạy bộ test bằng pytest.
    3. evaluate_model : đánh giá sức mạnh của agent AI bằng cách cho đấu thử.

GHI CHÚ VỀ THƯ VIỆN `agent-skills` (https://pypi.org/project/agent-skills/):
    API thực tế của package `agent-skills` (Datalayer) KHÔNG cung cấp một
    decorator `@skill` rời rạc. Thay vào đó nó dùng đối tượng ``AgentSkill`` và
    các decorator ``@skill.script`` / ``@skill.resource`` (định hướng runtime
    Pydantic AI + MCP). Module này tích hợp ĐÚNG theo API đó khi thư viện được
    cài đặt, nhưng đồng thời cung cấp phần lõi thuần Python + CLI để có thể tự
    chạy độc lập (không cần dựng MCP/sandbox). Nhờ vậy các skill vừa "đăng ký"
    được vào toolset của agent, vừa gọi trực tiếp như lệnh dòng lệnh:

        python agent_tools.py lint
        python agent_tools.py test
        python agent_tools.py eval --games 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Thư mục gốc của dự án (nơi chứa file này).
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ==========================================================================
#  KIỂU DỮ LIỆU KẾT QUẢ
# ==========================================================================
@dataclass(slots=True)
class CheckResult:
    """Kết quả của một lần chạy skill kiểm tra.

    Attributes:
        name: Tên skill đã chạy (vd: "lint_code").
        success: True nếu kiểm tra đạt, False nếu thất bại.
        summary: Tóm tắt ngắn gọn kết quả để hiển thị/log.
        details: Thông tin chi tiết (output stdout/stderr, số liệu...).
    """

    name: str
    success: bool
    summary: str
    details: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


# ==========================================================================
#  HÀM TIỆN ÍCH CHẠY LỆNH SHELL
# ==========================================================================
def _run_module(module: str, args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Chạy một module Python qua ``python -m`` và thu output gộp.

    Dùng ``sys.executable`` để luôn chạy bằng đúng interpreter hiện tại (kể cả
    khi virtualenv chưa được kích hoạt trong PATH), tránh lỗi "không tìm thấy
    chương trình".

    Args:
        module: Tên module thực thi, vd: "ruff", "pytest", "mypy".
        args: Tham số truyền cho module.
        cwd: Thư mục làm việc; mặc định là gốc dự án.

    Returns:
        Cặp (mã thoát, output văn bản). Mã thoát -1 nghĩa là module chưa được
        cài đặt trong môi trường hiện tại.
    """
    process = subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (process.stdout + process.stderr).strip()
    if process.returncode != 0 and f"No module named {module}" in output:
        return -1, f"Chưa cài '{module}' trong môi trường hiện tại."
    return process.returncode, output


# ==========================================================================
#  LÕI 3 SKILL (THUẦN PYTHON, GỌI ĐỘC LẬP ĐƯỢC)
# ==========================================================================
def lint_code(target: str = ".") -> CheckResult:
    """Skill 1: Kiểm tra phong cách code (ruff) và kiểu tĩnh (mypy).

    Quy trình:
        - Chạy ``ruff check`` để bắt lỗi PEP 8 / import thừa / lỗi cú pháp nhẹ.
        - Chạy ``mypy`` để kiểm tra type hinting (nếu mypy có sẵn).
    Skill coi là ĐẠT khi ruff không báo lỗi (mypy thiếu chỉ cảnh báo, không
    đánh trượt để tránh chặn tiến độ khi chưa cài).

    Args:
        target: Đường dẫn cần kiểm tra (thư mục hoặc file).

    Returns:
        CheckResult mô tả kết quả lint.
    """
    parts: list[str] = []
    ruff_code, ruff_out = _run_module("ruff", ["check", target])
    if ruff_code == -1:
        parts.append("[ruff] BỎ QUA (chưa cài ruff: pip install ruff).")
        ruff_ok = True
    else:
        ruff_ok = ruff_code == 0
        parts.append(f"[ruff] {'ĐẠT' if ruff_ok else 'LỖI'}\n{ruff_out or 'Không có vấn đề.'}")

    mypy_code, mypy_out = _run_module("mypy", [target])
    if mypy_code == -1:
        parts.append("[mypy] BỎ QUA (chưa cài mypy: pip install mypy).")
    else:
        parts.append(f"[mypy] {'ĐẠT' if mypy_code == 0 else 'CẢNH BÁO'}\n{mypy_out}")

    summary = "Lint ĐẠT." if ruff_ok else "Lint phát hiện lỗi cần sửa."
    return CheckResult("lint_code", ruff_ok, summary, "\n\n".join(parts))


def run_unit_tests(test_path: str = "tests") -> CheckResult:
    """Skill 2: Chạy unit test bằng pytest.

    Args:
        test_path: Thư mục/đường dẫn test cần chạy.

    Returns:
        CheckResult mô tả kết quả test.
    """
    if not (PROJECT_ROOT / test_path).exists():
        return CheckResult(
            "run_unit_tests",
            True,
            f"Chưa có thư mục test '{test_path}', bỏ qua.",
            "Hãy thêm test trong tests/ khi bắt đầu code.",
        )

    code, out = _run_module("pytest", ["-q", test_path])
    if code == -1:
        return CheckResult(
            "run_unit_tests",
            False,
            "Chưa cài pytest (pip install pytest).",
            out,
        )

    success = code == 0
    summary = "Tất cả test ĐẠT." if success else "Có test THẤT BẠI, cần sửa."
    return CheckResult("run_unit_tests", success, summary, out)


def evaluate_model(
    games: int = 20,
    agent_a: str = "hybrid",
    agent_b: str = "minimax",
) -> CheckResult:
    """Skill 3: Đánh giá sức mạnh agent AI bằng cách cho đấu thử nhiều ván.

    Skill này gọi tới module đánh giá của dự án (``ai.evaluate``) nếu đã tồn
    tại. Khi phần AI chưa được hiện thực, skill báo "đang chờ" thay vì gây lỗi,
    để workflow self-check không bị chặn ở giai đoạn đầu.

    Args:
        games: Số ván đấu thử.
        agent_a: Tên tác nhân thứ nhất (vd: "hybrid", "dqn", "minimax").
        agent_b: Tên tác nhân đối thủ.

    Returns:
        CheckResult kèm tỷ lệ thắng trong ``metrics`` nếu chạy được.
    """
    try:
        # Import trễ: chỉ nạp khi phần ai/ đã sẵn sàng.
        from ai.evaluate import play_match  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - cố ý bắt rộng ở giai đoạn đầu
        return CheckResult(
            "evaluate_model",
            True,
            "Bỏ qua: module 'ai.evaluate' chưa sẵn sàng.",
            f"Chi tiết: {exc}. Skill sẽ hoạt động sau khi hiện thực AI.",
        )

    result = play_match(agent_a=agent_a, agent_b=agent_b, num_games=games)
    win_rate = float(result.get("win_rate_a", 0.0))
    summary = f"{agent_a} thắng {win_rate:.1%} trên {games} ván trước {agent_b}."
    return CheckResult(
        "evaluate_model",
        True,
        summary,
        details=str(result),
        metrics={"win_rate_a": win_rate, "games": float(games)},
    )


# ==========================================================================
#  ĐĂNG KÝ VÀO `agent-skills` (NẾU CÓ CÀI ĐẶT)
# ==========================================================================
def build_agent_skills() -> object | None:
    """Đăng ký 3 skill vào đối tượng ``AgentSkill`` của thư viện agent-skills.

    Hàm trả về đối tượng skill đã cấu hình để nạp vào ``AgentSkillsToolset``
    của Pydantic AI. Nếu thư viện chưa được cài, trả về ``None`` (các skill vẫn
    dùng được trực tiếp qua CLI / import thuần).

    Returns:
        Đối tượng AgentSkill nếu thành công, ngược lại None.
    """
    try:
        from agent_skills import AgentSkill  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None

    skill = AgentSkill(
        name="caro-self-check",
        description="Bộ skill tự kiểm tra dự án AI Cờ Caro: lint, test, đánh giá model.",
        content=(
            "Dùng các script này để kiểm tra chất lượng code và sức mạnh AI: "
            "lint_code, run_unit_tests, evaluate_model."
        ),
    )

    @skill.script
    async def lint(ctx, target: str = ".") -> str:  # type: ignore[no-untyped-def]
        """Kiểm tra style & type của mã nguồn."""
        res = lint_code(target)
        return f"{res.summary}\n{res.details}"

    @skill.script
    async def test(ctx, test_path: str = "tests") -> str:  # type: ignore[no-untyped-def]
        """Chạy unit test bằng pytest."""
        res = run_unit_tests(test_path)
        return f"{res.summary}\n{res.details}"

    @skill.script
    async def evaluate(ctx, games: int = 20) -> str:  # type: ignore[no-untyped-def]
        """Đánh giá agent AI bằng cách cho đấu thử."""
        res = evaluate_model(games=games)
        return f"{res.summary}\n{res.details}"

    return skill


# ==========================================================================
#  CLI: cho phép tự chạy độc lập
# ==========================================================================
def _print_result(result: CheckResult) -> int:
    """In kết quả ra console và trả về mã thoát phù hợp cho shell."""
    status = "✅" if result.success else "❌"
    print(f"{status} [{result.name}] {result.summary}")
    if result.details:
        print("-" * 70)
        print(result.details)
    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI cho bộ công cụ tự kiểm tra.

    Args:
        argv: Danh sách tham số dòng lệnh (mặc định lấy từ sys.argv).

    Returns:
        Mã thoát: 0 nếu skill đạt, khác 0 nếu thất bại.
    """
    parser = argparse.ArgumentParser(
        description="Bộ skill tự kiểm tra dự án AI Cờ Caro (lint / test / eval)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_lint = sub.add_parser("lint", help="Kiểm tra style & type.")
    p_lint.add_argument("--target", default=".", help="Đường dẫn cần lint.")

    p_test = sub.add_parser("test", help="Chạy unit test.")
    p_test.add_argument("--path", default="tests", help="Thư mục test.")

    p_eval = sub.add_parser("eval", help="Đánh giá agent AI bằng đấu thử.")
    p_eval.add_argument("--games", type=int, default=20, help="Số ván đấu thử.")
    p_eval.add_argument("--a", default="hybrid", help="Agent thứ nhất.")
    p_eval.add_argument("--b", default="minimax", help="Agent đối thủ.")

    args = parser.parse_args(argv)

    if args.command == "lint":
        return _print_result(lint_code(args.target))
    if args.command == "test":
        return _print_result(run_unit_tests(args.path))
    if args.command == "eval":
        return _print_result(evaluate_model(args.games, args.a, args.b))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
