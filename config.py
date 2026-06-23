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
        double_end_block_rule: Luật «chặn 2 đầu» — đúng 5 quân bị kẹp hai
            đầu không thắng; AI/UI cũng ưu tiên chặn tam/tứ mở hai đầu.
        aggressive: AI ưu tiên tấn công (tam/tứ mở) trước chặn tam mở nhẹ.
        threat_warnings: Hiển thị cảnh báo sắp thắng trên UI.
    """

    double_end_block_rule: bool = DEFAULT_DOUBLE_END_BLOCK_RULE
    aggressive: bool = DEFAULT_AI_AGGRESSIVE
    threat_warnings: bool = DEFAULT_THREAT_WARNINGS


# Phần thưởng hình dạng khi huấn luyện DQN (khuyến khích tấn công).
DQN_REWARD_OPEN_FOUR: float = 0.5
DQN_REWARD_OPEN_THREE: float = 0.2
DQN_REWARD_STEP: float = -0.005

# Mức can thiệp của luật tactical cho LEARNER (quân DQN học) khi HUẤN LUYỆN.
# Đây là tham số then chốt: trước đây learner luôn dùng full tactical nên các
# nước quyết định (thắng/chặn/tấn công) đều do heuristic xử → mạng gần như
# KHÔNG có gì để học, self-play hoà 100% → win-rate ~0%.
#   "full" = dùng toàn bộ luật như khi chơi (cũ) — mạng học rất kém.
#   "safe" = chỉ tự ăn nước thắng-ngay + chặn-thua-ngay; còn lại mạng tự quyết
#            → buộc mạng học tạo/đỡ đe doạ tầm xa, self-play có thắng/thua thật
#            → có tín hiệu để học (KHUYẾN NGHỊ).
#   "none" = không can thiệp, mạng học từ con số 0 (cần rất nhiều ván).
# Lưu ý: chỉ ảnh hưởng lúc TRAIN. Khi CHƠI, DQNAgent vẫn dùng full tactical.
DQN_TRAIN_TACTICAL_LEVEL: str = "safe"

# Các kích thước bàn cờ cho phép người chơi chọn.
BOARD_SIZES: tuple[int, ...] = (3, 5, 7, 10, 15)
DEFAULT_BOARD_SIZE: int = 15


def win_length_for_board(board_size: int) -> int:
    """Số quân liên tiếp để thắng phù hợp với kích thước bàn.

    Bàn nhỏ (3×3, 5×5) dùng luật ngắn hơn để ván vẫn chơi được.
    """
    if board_size <= 3:
        return 3
    if board_size <= 5:
        return board_size
    return WIN_LENGTH


def create_caro_env(
    board_size: int,
    *,
    double_end_block_rule: bool = False,
) -> "CaroEnv":
    """Tạo ``CaroEnv`` với ``win_length`` tự động theo kích thước bàn."""
    from core.caro_env import CaroEnv

    return CaroEnv(
        size=board_size,
        win_length=win_length_for_board(board_size),
        double_end_block_rule=double_end_block_rule,
    )


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
DQN_BUFFER_CAPACITY: int = 100_000
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
DQN_DEFAULT_EPISODES: int = 10_000
DQN_SAVE_EVERY: int = 500
DQN_LOG_EVERY: int = 50
DQN_EVAL_EVERY: int = 500
DQN_EVAL_GAMES: int = 50
# Win rate tối thiểu để ghi checkpoint khi eval (train.py save gate).
DQN_SAVE_MIN_WIN_RATE_DELTA: float = 0.0

# Học online khi chơi Người vs AI (DQN / Hybrid).
ONLINE_LEARN_ENABLED: bool = True
ONLINE_LEARN_GRADIENT_STEPS: int = 12
ONLINE_LEARN_GRADIENT_STEPS_HIGH: int = 16
ONLINE_LEARN_GRADIENT_STEPS_LOW: int = 4
# Số mẫu tối thiểu trong buffer để chạy gradient sau một ván.
ONLINE_LEARN_MIN_SAMPLES: int = 64
# Ván AI thắng quá ngắn (human blunder) — chỉ buffer, không train.
ONLINE_LEARN_MIN_AI_MOVES: int = 6
# Ván hòa dài mới đáng học (reward cuối = 0).
ONLINE_LEARN_DRAW_MIN_AI_MOVES: int = 20
# Credit assignment khi AI thua: nước cuối → kế cuối → thứ 3 từ cuối.
ONLINE_LOSS_CREDIT_REWARDS: tuple[float, ...] = (-1.0, -0.5, -0.25)
# Chỉ ghi checkpoint online nếu pass save gate (benchmark / transition / loss).
ONLINE_SAVE_GATE_ENABLED: bool = True
# Từ chối lưu nếu loss vượt ratio × trung bình N lần học gần nhất.
ONLINE_LOSS_SPIKE_RATIO: float = 2.0
ONLINE_LOSS_HISTORY_SIZE: int = 10
# Double DQN khi tính target Q (giảm overestimate).
DQN_USE_DOUBLE_DQN: bool = True

# Hybrid MẠNH HƠN Minimax thật sự nhờ TÌM SÂU HƠN 1 ply (depth+1), trong đó
# DQN + heuristic sắp xếp nước (move ordering) giúp alpha-beta cắt tỉa sớm để
# bù chi phí của ply phụ. Đo bằng đối kháng thật (head-to-head), KHÔNG phải điểm
# heuristic-1-nước của benchmark (thước đo đó thiên vị nước có heuristic tức thời
# cao, nên search sâu hơn đôi khi bị chấm thấp dù mạnh hơn khi đánh thật).
HYBRID_EXTRA_DEPTH: int = 1
HYBRID_MAX_DEPTH: int = 5

# Iterative deepening: ngân sách thời gian (giây) cho MỖI nước khi chơi tương tác.
# None = tìm hết độ sâu (tất định, dùng trong test trực tiếp & benchmark). Agent
# tạo qua ``from_difficulty`` (UI/web) nhận ngân sách này để không vượt
# ``AI_MOVE_TIMEOUT_SEC`` ở độ sâu lớn — luôn trả nước tốt nhất tới thời điểm cắt.
MINIMAX_PLAY_TIME_BUDGET_SEC: float = 5.0
HYBRID_PLAY_TIME_BUDGET_SEC: float = 6.0

# Giới hạn số nhánh mở rộng mỗi node cho Minimax khi CHƠI tương tác (selective
# search — giống engine online). Sau move-ordering chỉ giữ top-K nước; nhờ lớp
# tactical đã lọc nước thắng/chặn nên cắt nhánh này gần như không mất nước hay.
# Càng sâu thì K càng nhỏ để khống chế bùng nổ cấp số mũ.
#
# MẶC ĐỊNH = None (KHÔNG cắt nhánh) để GIỮ NGUYÊN SỨC MẠNH: nhờ evaluator tăng
# dần, tìm kiếm vét cạn giờ đã đủ nhanh (depth 2 ~0,9s, depth 3 ~10s; iterative
# deepening luôn trả nước tốt nhất tới mốc thời gian). Thực nghiệm đối kháng cho
# thấy cắt nhánh top-K (kể cả K=24) làm YẾU đi đôi chút vì move-ordering cục bộ
# chưa đủ tinh để chắc chắn giữ lại nước hay — nên mặc định tắt.
#
# Nếu chạy trên máy yếu và cần nhanh hơn nữa (chấp nhận giảm sức một chút), đặt
# số nguyên (vd MEDIUM: 24) để chỉ mở rộng top-K nước mỗi node.
MINIMAX_MAX_BRANCH_BY_DIFFICULTY: dict[Difficulty, int | None] = {
    Difficulty.EASY: None,
    Difficulty.MEDIUM: None,
    Difficulty.HARD: None,
    Difficulty.EXPERT: None,
}
# Budget thêm cho Hybrid trong benchmark (search rộng hơn Minimax cùng depth).
HYBRID_BENCHMARK_BRANCH_BONUS: int = 6
HYBRID_BENCHMARK_RADIUS_BONUS: int = 1
def hybrid_depth_for(difficulty: Difficulty) -> int:
    """Độ sâu Hybrid = Minimax cùng mức + EXTRA, giới hạn MAX."""
    return min(int(difficulty) + HYBRID_EXTRA_DEPTH, HYBRID_MAX_DEPTH)


# Legacy map (tham chiếu UI cũ); depth thực tế lấy từ ``hybrid_depth_for``.
HYBRID_DEPTH_BY_DIFFICULTY: dict[Difficulty, int] = {
    Difficulty.EASY: hybrid_depth_for(Difficulty.EASY),
    Difficulty.MEDIUM: hybrid_depth_for(Difficulty.MEDIUM),
    Difficulty.HARD: hybrid_depth_for(Difficulty.HARD),
    Difficulty.EXPERT: hybrid_depth_for(Difficulty.EXPERT),
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

# Trộn heuristic + DQN khi tinh chỉnh nước đi ở root (sau Minimax).
HYBRID_LEAF_HEURISTIC_WEIGHT: float = 0.55
# Số ứng viên top-K ở root để DQN chọn lại (chỉ vài forward, không chạy ở mọi node lá).
HYBRID_ROOT_REFINE_CANDIDATES: int = 12
# Chỉ đổi nước minimax nếu điểm trộn cao hơn ít nhất tỷ lệ này (tránh nhiễu DQN).
HYBRID_ROOT_REFINE_MIN_GAIN: float = 1.02

# Hybrid: trong refinement sau search, nếu hai nước chênh lệch minimax score nhỏ
# hơn margin thì ưu tiên nước có heuristic tức thì cao hơn (cải thiện rank benchmark).
# Margin tuyệt đối 5000 ≈ 5% khoảng điển hình (95000+) — đủ để kích hoạt trên
# phần lớn tình huống giữa ván mà không đánh đổi nước chiến lược rõ ràng tốt hơn.
HYBRID_TIE_REL_MARGIN: float = 0.0
HYBRID_TIE_ABS_MARGIN: float = 5000.0

# Ngân sách thời gian (giây) cho Hybrid trong basic benchmark — cho phép hoàn thành
# depth=2 (~22 ms) rồi thử depth=3; nếu depth=3 chưa xong thì trả kết quả depth=2.
# Giữ nhỏ để điểm tốc độ (speed_pts) sát Minimax trong bảng xếp hạng.
HYBRID_BENCHMARK_TIME_BUDGET_BASIC: float = 0.06  # 60 ms

# Thời gian tối đa chờ AI (giây) trước khi fallback nước đi an toàn.
AI_MOVE_TIMEOUT_SEC: float = 45.0

# VCF/VCT — tìm chuỗi thắng ép buộc (threat-space search).
VCF_MAX_DEPTH: int = 8           # Số ply xen kẽ tối đa khi tấn công VCF
VCT_MAX_DEPTH: int = 10          # Hybrid dùng (sâu hơn, gồm tam mở)
VCF_OPPONENT_DEPTH: int = 8      # Độ sâu khi quét VCF phòng thủ (đối thủ)
VCF_ENABLED: bool = True         # Tắt để debug / so sánh không VCF

def dqn_model_path(board_size: int) -> Path:
    """Đường dẫn file checkpoint DQN theo kích thước bàn cờ.

    Args:
        board_size: Cạnh bàn cờ (3, 5, 7, 10 hoặc 15).

    Returns:
        Path tới file ``.pth`` tương ứng.
    """
    return MODELS_DIR / f"dqn_{board_size}.pth"


def dqn_model_backup_path(board_size: int) -> Path:
    """Bản sao checkpoint trước lần học online gần nhất (để so sánh)."""
    return MODELS_DIR / f"dqn_{board_size}.backup.pth"


def dqn_model_best_path(board_size: int) -> Path:
    """Checkpoint tốt nhất theo eval win rate (train.py curriculum)."""
    return MODELS_DIR / f"dqn_{board_size}.best.pth"


def learn_log_path(board_size: int) -> Path:
    """Nhật ký JSONL các lần học online theo kích thước bàn."""
    return MODELS_DIR / f"learn_log_{board_size}.jsonl"


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
