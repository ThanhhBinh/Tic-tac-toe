# AI Cờ Caro (Gomoku) — Minimax + Alpha-Beta + DQN

Dự án xây dựng AI chơi Cờ Caro kết hợp **Minimax + Alpha-Beta Pruning** và
**Deep Q-Network (DQN)**, tích hợp **giao diện đồ họa** bằng `pygame`.

Kiến trúc tách lớp rõ ràng: `core/` (logic game) ← `ai/` (thuật toán) ← `ui/`
(giao diện). Các tầng `core/` và `ai/` chạy độc lập, không phụ thuộc UI.

## Cài đặt

> Dự án dùng **Python 3.12** (PyTorch chưa hỗ trợ Python 3.14). Đã tạo sẵn môi
> trường ảo tại `.venv`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy game

### Giao diện web (Chrome) — khuyến nghị

```bash
source .venv/bin/activate
pip install -r requirements.txt
python web_main.py
```

Mở trình duyệt: **http://127.0.0.1:8000**

### Giao diện desktop (Pygame)

```bash
python main.py
```

Điều khiển desktop: nhấp chuột để đặt quân · `R` chơi lại · `ESC` về menu.

## Huấn luyện DQN

```bash
# Self-play (mặc định, bàn 15x15, 3000 episode)
python train.py

# Học qua đấu với Minimax
python train.py --mode minimax --opponent-depth 2

# Tuỳ chỉnh nhanh
python train.py --board-size 10 --episodes 1000 --save-every 200

# Tiếp tục từ checkpoint
python train.py --resume models/dqn_15.pth --episodes 2000
```

Model lưu tại `models/dqn_{size}.pth` — DQN / Hybrid tự nạp khi chơi.

## Tự kiểm tra (self-check skills)

```bash
python agent_tools.py lint      # ruff + mypy
python agent_tools.py test      # pytest
python agent_tools.py eval --games 20
```

## Cấu trúc

| Thư mục       | Vai trò                                                 |
| ------------- | ------------------------------------------------------- |
| `core/`       | Môi trường game `CaroEnv` (NumPy), không biết UI        |
| `ai/`         | `Agent`, Minimax, DQN, Hybrid (Minimax + DQN lá) |
| `ui/`         | Màn hình desktop (pygame + pygame-menu)                 |
| `web/`        | Giao diện web FastAPI + static (chạy trên Chrome)       |
| `tests/`      | Unit test (pytest)                                      |
| `config.py`   | Tham số toàn cục (kích thước bàn, màu, FPS...)          |
| `train.py`    | CLI huấn luyện DQN (self-play / vs Minimax)             |

## Tiến độ

- [x] **Bước 1** — `CaroEnv` + cấu hình + unit test
- [x] **Khung UI** — menu, settings, game screen (PvP chạy được), end overlay
- [x] **Bước 2** — `MinimaxAgent` (Alpha-Beta + heuristic pattern-based)
- [x] **Bước 3** — `DQNAgent` (PyTorch CNN + replay buffer + trainer)
- [x] **Bước 4** — `HybridAgent` (Minimax depth 2–4 + DQN ở node lá)
- [x] **Bước 5** — `train.py` (self-play / vs Minimax, lưu checkpoint định kỳ)
- [x] **Bước 6** — UI polish: hover preview, HUD nâng cao, animation thắng/end

> Chọn **AI: DQN** hoặc **Hybrid** trong Settings. Model lưu tại
> ``models/dqn_{size}.pth`` sau khi chạy ``python train.py``.
