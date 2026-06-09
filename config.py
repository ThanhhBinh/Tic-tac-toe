#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cấu hình toàn cục cho dự án AI Cờ Caro.

Tập trung mọi hằng số & tham số có thể tinh chỉnh tại một nơi duy nhất để
tránh rải "magic number" khắp code (theo .cursorrules mục 10).
"""

from __future__ import annotations

from enum import Enum, IntEnum
from dataclasses import dataclass
from pathlib import Path

# ==========================================================================
#  ĐƯỜNG DẪN
# ==========================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = PROJECT_ROOT / "ui" / "assets"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ==========================================================================
#  LUẬT GAME
# ==========================================================================
# Số quân liên tiếp để thắng (cờ caro chuẩn: 5).
WIN_LENGTH: int = 5

# Luật & cảnh báo mặc định (Settings / web form).
DEFAULT_DOUBLE_END_BLOCK_RULE: bool = True
DEFAULT_THREAT_WARNINGS: bool = True
DEFAULT_AI_AGGRESSIVE: bool = True
DEFAULT_AI_FIRST: bool = False


@dataclass
class TacticalConfig:
    """Cấu hình luật chiến thuật cho AI và cảnh báo UI.

    Attributes:
        double_end_block_rule: Bật chặn tam/tứ mở hai đầu (luật «chặn 2 đầu»).
        aggressive: AI ưu tiên tấn công (tam/tứ mở) trước chặn tam mở nhẹ.
        threat_warnings: Hiển thị cảnh báo sắp thắng trên UI.
    """

    double_end_block_rule: bool = DEFAULT_DOUBLE_END_BLOCK_RULE
    aggressive: bool = DEFAULT_AI_AGGRESSIVE
    threat_warnings: bool = DEFAULT_THREAT_WARNINGS


# Phần thưởng hình dạng khi huấn luyện DQN (khuyến khích tấn công).
DQN_REWARD_OPEN_FOUR: float = 0.35
DQN_REWARD_OPEN_THREE: float = 0.12
DQN_REWARD_STEP: float = -0.005

# Các kích thước bàn cờ cho phép người chơi chọn.
BOARD_SIZES: tuple[int, ...] = (10, 15)
DEFAULT_BOARD_SIZE: int = 15


class Player(IntEnum):
    """Định danh người chơi trên bàn cờ (trùng giá trị lưu trong NumPy)."""

    EMPTY = 0  # Ô trống
    X = 1      # Người đi trước (quân X)
    O = 2      # noqa: E741 - "O" là tên quân cờ theo quy ước, giữ nguyên

    @property
    def opponent(self) -> "Player":
        """Trả về đối thủ của người chơi hiện tại (X <-> O)."""
        if self is Player.X:
            return Player.O
        if self is Player.O:
            return Player.X
        return Player.EMPTY


# ==========================================================================
#  CHẾ ĐỘ CHƠI & LOẠI AI
# ==========================================================================
class GameMode(str, Enum):
    """Chế độ chơi được chọn trong Settings."""

    PVP = "Player vs Player"
    PVA = "Player vs AI"
    AVA = "AI vs AI"


class AIType(str, Enum):
    """Loại tác nhân AI."""

    MINIMAX = "Minimax"
    DQN = "DQN"
    HYBRID = "Hybrid (Minimax + DQN)"


class Difficulty(IntEnum):
    """Độ khó, ánh xạ trực tiếp sang độ sâu tìm kiếm của Minimax."""

    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


# ==========================================================================
#  DQN (Deep Q-Network)
# ==========================================================================
# Kích thước bàn cờ lớn nhất (dùng khi cần tham chiếu chung).
MAX_BOARD_SIZE: int = max(BOARD_SIZES)

# Siêu tham số huấn luyện DQN.
DQN_LEARNING_RATE: float = 1e-4
DQN_GAMMA: float = 0.99
DQN_BATCH_SIZE: int = 64
DQN_BUFFER_CAPACITY: int = 50_000
DQN_EPSILON_START: float = 1.0
DQN_EPSILON_END: float = 0.05
DQN_EPSILON_DECAY: float = 0.9995
DQN_TARGET_SYNC_EVERY: int = 200
DQN_TRAIN_EVERY: int = 4
DQN_MIN_BUFFER: int = 256

# Epsilon khi chơi (suy luận) theo độ khó — khó hơn = ít ngẫu nhiên hơn.
DQN_PLAY_EPSILON: dict[Difficulty, float] = {
    Difficulty.EASY: 0.20,
    Difficulty.MEDIUM: 0.08,
    Difficulty.HARD: 0.02,
    Difficulty.EXPERT: 0.0,
}

# Mặc định cho script ``train.py``.
DQN_DEFAULT_EPISODES: int = 3_000
DQN_SAVE_EVERY: int = 500
DQN_LOG_EVERY: int = 50
DQN_EVAL_EVERY: int = 500
DQN_EVAL_GAMES: int = 10

# Hybrid: depth GIỚI HẠN thấp hơn Minimax thuần vì mỗi node lá chạy forward DQN.
# Expert=3 (KHÔNG dùng 4) — depth 4 + DQN khiến UI treo hàng chục giây/phút.
HYBRID_DEPTH_BY_DIFFICULTY: dict[Difficulty, int] = {
    Difficulty.EASY: 1,
    Difficulty.MEDIUM: 2,
    Difficulty.HARD: 2,
    Difficulty.EXPERT: 3,
}

# Giới hạn số nhánh mở rộng mỗi node (giảm mũ số node lá → DQN nhanh hơn).
HYBRID_MAX_BRANCH_BY_DIFFICULTY: dict[Difficulty, int] = {
    Difficulty.EASY: 6,
    Difficulty.MEDIUM: 8,
    Difficulty.HARD: 12,
    Difficulty.EXPERT: 14,
}

# Bán kính ứng viên Hybrid theo độ khó (Expert nhìn xa hơn quanh cụm quân).
HYBRID_CANDIDATE_RADIUS_BY_DIFFICULTY: dict[Difficulty, int] = {
    Difficulty.EASY: 2,
    Difficulty.MEDIUM: 2,
    Difficulty.HARD: 2,
    Difficulty.EXPERT: 3,
}

# Trộn heuristic + DQN tại node lá (heuristic ổn định, DQN bổ sung khi đã train).
HYBRID_LEAF_HEURISTIC_WEIGHT: float = 0.55

# Thời gian tối đa chờ AI (giây) trước khi fallback nước đi an toàn.
AI_MOVE_TIMEOUT_SEC: float = 45.0

def dqn_model_path(board_size: int) -> Path:
    """Đường dẫn file checkpoint DQN theo kích thước bàn cờ.

    Args:
        board_size: Cạnh bàn cờ (10 hoặc 15).

    Returns:
        Path tới file ``.pth`` tương ứng.
    """
    return MODELS_DIR / f"dqn_{board_size}.pth"


# ==========================================================================
#  GIAO DIỆN (UI)
# ==========================================================================
WINDOW_WIDTH: int = 1100
WINDOW_HEIGHT: int = 760
FPS: int = 60
WINDOW_TITLE: str = "AI Cờ Caro — Minimax + DQN"

# Chiều rộng dành cho sidebar/HUD bên phải màn hình chơi.
SIDEBAR_WIDTH: int = 320

# Tốc độ animation (giây) cho hiệu ứng đặt quân.
PLACE_ANIM_DURATION: float = 0.22
END_OVERLAY_ANIM_DURATION: float = 0.35
AI_THINK_MIN_DISPLAY: float = 0.25
WIN_PULSE_SPEED: float = 4.0
LAST_MOVE_PULSE_SPEED: float = 5.0
HOVER_PREVIEW_ALPHA: int = 110


class Theme:
    """Bảng màu chủ đạo — modern minimal kết hợp bàn cờ gỗ classic.

    Mọi giá trị là tuple RGB (0-255). Tập trung tại đây để dễ đổi chủ đề.
    """

    # Nền tổng thể
    BACKGROUND: tuple[int, int, int] = (24, 26, 32)
    SURFACE: tuple[int, int, int] = (33, 36, 44)
    SURFACE_LIGHT: tuple[int, int, int] = (45, 49, 60)

    # Bàn cờ gỗ
    BOARD_WOOD: tuple[int, int, int] = (222, 184, 135)
    BOARD_WOOD_DARK: tuple[int, int, int] = (193, 154, 107)
    GRID_LINE: tuple[int, int, int] = (120, 90, 60)

    # Quân cờ
    STONE_X: tuple[int, int, int] = (38, 42, 52)       # X: đen ngả xám
    STONE_O: tuple[int, int, int] = (245, 246, 250)    # O: trắng ngà
    STONE_SHADOW: tuple[int, int, int] = (0, 0, 0)

    # Nhấn mạnh
    ACCENT: tuple[int, int, int] = (94, 169, 255)      # xanh dương hiện đại
    ACCENT_WARM: tuple[int, int, int] = (255, 176, 59)
    HIGHLIGHT_LAST: tuple[int, int, int] = (94, 169, 255)
    HIGHLIGHT_WIN: tuple[int, int, int] = (76, 217, 138)
    DANGER: tuple[int, int, int] = (255, 96, 96)

    # Văn bản
    TEXT_PRIMARY: tuple[int, int, int] = (236, 239, 244)
    TEXT_MUTED: tuple[int, int, int] = (150, 158, 172)
