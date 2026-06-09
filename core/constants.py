#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hằng số và kiểu dữ liệu dùng chung cho gói `core`.

Tách riêng để các module trong core/ và ai/ tham chiếu mà không tạo phụ thuộc
vòng (circular import) với config.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- Type alias dùng lại nhiều nơi ---
# Bàn cờ: ma trận 2 chiều dtype int8 (0=trống, 1=X, 2=O).
Board = NDArray[np.int8]

# Một nước đi: (hàng, cột).
Move = tuple[int, int]

# 4 hướng quét để kiểm tra 5 quân liên tiếp:
#   ngang, dọc, chéo xuống-phải, chéo lên-phải.
# Chỉ cần 4 hướng (mỗi hướng kiểm tra cả 2 phía) thay vì 8.
DIRECTIONS: tuple[Move, ...] = ((0, 1), (1, 0), (1, 1), (1, -1))
