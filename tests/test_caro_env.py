#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit test cho môi trường game `CaroEnv`.

Kiểm tra: đặt quân hợp lệ/không hợp lệ, phát hiện thắng theo cả 4 hướng,
hòa cờ, đổi lượt và bản sao (clone) độc lập.
"""

from __future__ import annotations

import pytest

from config import Player
from core.caro_env import CaroEnv


def _play(env: CaroEnv, moves: list[tuple[int, int]]) -> None:
    """Tiện ích: lần lượt thực hiện danh sách nước đi cho hai bên."""
    for move in moves:
        env.step(move)


def test_khoi_tao_ban_co_rong() -> None:
    """Bàn cờ mới phải rỗng, X đi trước, chưa kết thúc."""
    env = CaroEnv(size=15)
    assert env.board.shape == (15, 15)
    assert int(env.board.sum()) == 0
    assert env.current_player is Player.X
    assert not env.done
    assert env.winner is None


def test_size_nho_hon_win_length_bao_loi() -> None:
    """Khởi tạo bàn cờ nhỏ hơn số quân thắng phải báo ValueError."""
    with pytest.raises(ValueError):
        CaroEnv(size=4, win_length=5)


def test_dat_quan_va_doi_luot() -> None:
    """Sau khi X đi, ô được set đúng và đến lượt O."""
    env = CaroEnv(size=10)
    env.step((5, 5))
    assert env.board[5, 5] == Player.X
    assert env.current_player is Player.O


def test_nuoc_di_khong_hop_le_o_da_co_quan() -> None:
    """Đánh vào ô đã có quân phải báo ValueError."""
    env = CaroEnv(size=10)
    env.step((0, 0))
    with pytest.raises(ValueError):
        env.step((0, 0))


def test_nuoc_di_ngoai_ban_co() -> None:
    """Nước đi ngoài biên là không hợp lệ."""
    env = CaroEnv(size=10)
    assert not env.is_legal((-1, 0))
    assert not env.is_legal((10, 10))
    with pytest.raises(ValueError):
        env.step((10, 0))


def test_thang_hang_ngang() -> None:
    """X tạo 5 quân ngang liên tiếp -> X thắng."""
    env = CaroEnv(size=15)
    # X: (7,0..4), O đệm ở hàng khác để không thắng.
    _play(env, [(7, 0), (0, 0), (7, 1), (0, 1), (7, 2), (0, 2), (7, 3), (0, 3), (7, 4)])
    assert env.done
    assert env.winner is Player.X
    assert len(env.winning_line) == 5


def test_thang_hang_doc() -> None:
    """X tạo 5 quân dọc liên tiếp -> X thắng."""
    env = CaroEnv(size=15)
    _play(env, [(0, 7), (0, 0), (1, 7), (0, 1), (2, 7), (0, 2), (3, 7), (0, 3), (4, 7)])
    assert env.winner is Player.X


def test_thang_cheo_xuong_phai() -> None:
    """X tạo 5 quân chéo xuống-phải -> X thắng."""
    env = CaroEnv(size=15)
    _play(env, [(0, 0), (5, 0), (1, 1), (5, 1), (2, 2), (5, 2), (3, 3), (5, 3), (4, 4)])
    assert env.winner is Player.X


def test_thang_cheo_len_phai() -> None:
    """X tạo 5 quân chéo lên-phải -> X thắng."""
    env = CaroEnv(size=15)
    _play(env, [(4, 0), (9, 0), (3, 1), (9, 1), (2, 2), (9, 2), (1, 3), (9, 3), (0, 4)])
    assert env.winner is Player.X


def test_hoa_co_ban_nho() -> None:
    """Lấp đầy bàn 5x5 (win_length=5) mà không ai đủ 5 quân -> hòa.

    Mẫu dưới đây được thiết kế để: không hàng/cột/đường chéo dài 5 nào toàn
    cùng màu, và có đúng 13 quân X + 12 quân O (khớp với lượt đi X trước).
    """
    env = CaroEnv(size=5, win_length=5)
    pattern = [
        [1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2],
        [1, 2, 2, 2, 1],
        [1, 1, 2, 1, 2],
        [1, 2, 1, 2, 1],
    ]
    cells_x = [(r, c) for r in range(5) for c in range(5) if pattern[r][c] == 1]
    cells_o = [(r, c) for r in range(5) for c in range(5) if pattern[r][c] == 2]
    assert len(cells_x) == 13 and len(cells_o) == 12

    # Xen kẽ X, O, X, O, ... (X đi trước) để khớp luật đổi lượt của môi trường.
    order: list[tuple[int, int]] = []
    for i in range(25):
        order.append(cells_x.pop() if i % 2 == 0 else cells_o.pop())
    _play(env, order)

    assert env.done
    assert env.is_draw
    assert env.winner is None


def test_clone_doc_lap() -> None:
    """Bản sao phải độc lập: sửa bản sao không ảnh hưởng bản gốc."""
    env = CaroEnv(size=10)
    env.step((1, 1))
    clone = env.clone()
    clone.step((2, 2))
    assert env.board[2, 2] == Player.EMPTY
    assert clone.board[2, 2] == Player.O
    assert env.move_count == 1
    assert clone.move_count == 2


def test_push_pop_hoan_tac() -> None:
    """push/pop phải khôi phục đúng trạng thái sau khi thử nước đi tìm kiếm."""
    env = CaroEnv(size=10)
    env.step((5, 5))
    before = env.board.copy()
    turn_before = env.current_player
    count_before = env.move_count

    env.push((5, 6))
    assert env.move_count == count_before + 1
    env.pop()

    assert env.board.tobytes() == before.tobytes()
    assert env.current_player is turn_before
    assert env.move_count == count_before
    assert env.done is False
    assert env.winner is None


def test_candidate_moves_o_trung_tam_khi_trong() -> None:
    """Bàn trống -> nước ứng viên là ô trung tâm."""
    env = CaroEnv(size=15)
    assert env.candidate_moves() == [(7, 7)]


def test_candidate_moves_quanh_quan_da_dat() -> None:
    """Sau khi đặt 1 quân, ứng viên là các ô trống lân cận."""
    env = CaroEnv(size=15)
    env.step((7, 7))
    candidates = env.candidate_moves(radius=1)
    assert (7, 7) not in candidates  # ô đã có quân không phải ứng viên
    assert (6, 6) in candidates and (8, 8) in candidates
    assert len(candidates) == 8


def test_legal_moves_giam_dan() -> None:
    """Số nước hợp lệ giảm đúng 1 sau mỗi lượt."""
    env = CaroEnv(size=10)
    assert len(env.legal_moves()) == 100
    env.step((0, 0))
    assert len(env.legal_moves()) == 99


def test_reset_ve_trang_thai_dau() -> None:
    """reset() đưa môi trường về trạng thái ban đầu."""
    env = CaroEnv(size=10)
    _play(env, [(0, 0), (1, 1), (0, 1)])
    env.reset()
    assert int(env.board.sum()) == 0
    assert env.current_player is Player.X
    assert env.move_count == 0
    assert not env.done
    assert env.winner is None


def _place_stones(env: CaroEnv, stones: dict[tuple[int, int], Player]) -> None:
    """Đặt sẵn quân trên bàn (bỏ qua luật lượt — chỉ dùng trong test)."""
    for (row, col), player in stones.items():
        env.board[row, col] = player
    env._move_count = len(stones)


def test_ban_3x3_hang_3_van_thang() -> None:
    """Bàn 3×3 — đủ 3 quân liên tiếp vẫn thắng dù luật chặn 2 đầu bật."""
    env = CaroEnv(size=3, double_end_block_rule=True)
    _play(env, [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)])
    assert env.done
    assert env.winner is Player.O
    assert len(env.winning_line) == 3


def test_chan_2_dau_khong_thang() -> None:
    """OXXXXXO — đúng 5 quân bị chặn hai đầu, luật bật → không thắng."""
    env = CaroEnv(size=10, double_end_block_rule=True)
    row = 5
    _place_stones(
        env,
        {
            (row, 0): Player.O,
            (row, 1): Player.X,
            (row, 2): Player.X,
            (row, 3): Player.X,
            (row, 4): Player.X,
            (row, 6): Player.O,
        },
    )
    env.current_player = Player.X
    env.step((row, 5))
    assert not env.done
    assert env.winner is None


def test_chan_2_dau_mot_dau_mo_van_thang() -> None:
    """_XXXXXO — một đầu trống, luật bật → vẫn thắng."""
    env = CaroEnv(size=10, double_end_block_rule=True)
    row = 5
    _place_stones(
        env,
        {
            (row, 1): Player.X,
            (row, 2): Player.X,
            (row, 3): Player.X,
            (row, 4): Player.X,
        },
    )
    env.current_player = Player.X
    env.step((row, 5))
    assert env.done
    assert env.winner is Player.X


def test_chan_2_dau_tat_van_thang() -> None:
    """OXXXXXO — luật tắt → vẫn tính thắng như cờ caro chuẩn."""
    env = CaroEnv(size=10, double_end_block_rule=False)
    row = 5
    _place_stones(
        env,
        {
            (row, 0): Player.O,
            (row, 1): Player.X,
            (row, 2): Player.X,
            (row, 3): Player.X,
            (row, 4): Player.X,
        },
    )
    env.current_player = Player.X
    env.step((row, 5))
    assert env.done
    assert env.winner is Player.X


def test_hon_6_quan_bi_chan_van_thang() -> None:
    """OXXXXXXO — 6 quân liên tiếp, luật bật → vẫn thắng."""
    env = CaroEnv(size=10, double_end_block_rule=True)
    row = 5
    _place_stones(
        env,
        {
            (row, 0): Player.O,
            (row, 1): Player.X,
            (row, 2): Player.X,
            (row, 3): Player.X,
            (row, 4): Player.X,
            (row, 5): Player.X,
        },
    )
    env.current_player = Player.X
    env.step((row, 6))
    assert env.done
    assert env.winner is Player.X
