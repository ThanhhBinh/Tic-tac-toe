<div align="center">

# BÁO CÁO BÀI TẬP LỚN

## MÔN: TRÍ TUỆ NHÂN TẠO

---

## ĐỀ TÀI

### XÂY DỰNG HỆ THỐNG AI CHƠI CỜ CARO  
### KẾT HỢP THUẬT TOÁN MINIMAX + ALPHA-BETA PRUNING  
### VÀ DEEP Q-NETWORK (DQN)

---

**Nhóm thực hiện:**

| STT | Họ và tên | MSSV | Vai trò |
|-----|-----------|------|---------|
| 1 | **Nguyễn Minh Quân** | 21521001 | Trưởng nhóm — Minimax, Hybrid |
| 2 | **Phạm Thị Lan Anh** | 21521015 | DQN, huấn luyện mô hình |
| 3 | **Trần Đức Huy** | 21521027 | Giao diện Web, FastAPI |
| 4 | **Lê Hoàng Phúc** | 21521033 | Giao diện Desktop, kiểm thử |

**Lớp:** K215IT — Khóa 2021  
**Khoa:** Công nghệ Thông tin  
**Trường:** Đại học Công nghệ Thông tin — ĐHQG-HCM  

**Giảng viên hướng dẫn:** TS. Trần Văn Bình  

**TP. Hồ Chí Minh, tháng 06 năm 2026**

</div>

---

## LỜI CẢM ƠN

Trước hết, nhóm chúng em xin gửi lời cảm ơn chân thành tới **TS. Trần Văn Bình** — giảng viên phụ trách môn Trí tuệ nhân tạo, đã tận tình giảng dạy, hướng dẫn và góp ý trong suốt quá trình thực hiện đồ án này.

Em cũng xin cảm ơn các bạn trong lớp K215IT đã chia sẻ tài liệu, góp ý khi em demo sản phẩm, cũng như các nguồn báo cáo mẫu trên mạng giúp em hình dung được cấu trúc một bài báo cáo đồ án AI chuẩn.

Cuối cùng, em xin cảm ơn gia đình và bạn bè đã tạo điều kiện để em có thời gian hoàn thành dự án — đặc biệt là những đêm ngồi train DQN mà máy quạt ầm ầm suốt vì CPU nóng.

Do thời gian và kinh nghiệm còn hạn chế, báo cáo không tránh khỏi thiếu sót. Em rất mong nhận được ý kiến đóng góp từ thầy và các bạn.

**Nhóm thực hiện**

---

## LỜI CAM ĐOAN

Chúng em xin cam đoan đây là công trình nghiên cứu của nhóm, được thực hiện dưới sự hướng dẫn của **TS. Trần Văn Bình**. Các nội dung trình bày trong báo cáo là kết quả quá trình tìm hiểu, phân tích source code và triển khai thực tế. Mọi tài liệu tham khảo đều được trích dẫn rõ ràng ở cuối báo cáo.

Nếu có sai sót, chúng em xin hoàn toàn chịu trách nhiệm.

<div align="right">

**TP. Hồ Chí Minh, ngày 15 tháng 06 năm 2026**

**Nhóm trưởng**  
*(Nguyễn Minh Quân)*

</div>

---

## MỤC LỤC

| Chương | Nội dung | Trang |
|--------|----------|-------|
| | Lời cảm ơn | |
| | Lời cam đoan | |
| **1** | **Tổng quan đề tài** | |
| 1.1 | Lý do chọn đề tài | |
| 1.2 | Mục tiêu nghiên cứu | |
| 1.3 | Phạm vi đồ án | |
| 1.4 | Phương pháp thực hiện | |
| 1.5 | **Chiến lược kết hợp Minimax và DQN** | |
| **2** | **Cơ sở lý thuyết** | |
| 2.1 | Trò chơi hai người và cây trò chơi | |
| 2.2 | Thuật toán Minimax | |
| 2.3 | Cắt tỉa Alpha-Beta | |
| 2.4 | Hàm lượng giá (Heuristic) | |
| 2.5 | Học tăng cường và Deep Q-Network | |
| **3** | **Phân tích và thiết kế hệ thống** | |
| 3.1 | Yêu cầu chức năng và phi chức năng | |
| 3.2 | Kiến trúc phần mềm | |
| 3.3 | Thiết kế môi trường game `CaroEnv` | |
| 3.4 | Thiết kế các Agent AI | |
| **4** | **Cài đặt và đọc hiểu source code** | |
| 4.1 | Cấu trúc thư mục dự án | |
| 4.2 | Module `core/` — logic game | |
| 4.3 | Module `ai/` — Minimax Agent | |
| 4.4 | Module `ai/` — DQN Agent | |
| 4.5 | Module `ai/` — huấn luyện DQN | |
| 4.5.5 | Ba nguồn dữ liệu huấn luyện | |
| 4.5.6 | Cơ chế học: DQN học cái gì? | |
| 4.5.7 | Học online từ ván Người vs AI | |
| 4.5.8 | Áp dụng kiến thức đã học khi chơi | |
| 4.6 | Giao diện người dùng | |
| **5** | **Kết hợp hai thuật toán: Hybrid Agent** | |
| 5.1 | Động lực và ý tưởng | |
| 5.2 | Kiến trúc lai ghép | |
| 5.3 | Công thức đánh giá node lá | |
| 5.4 | Ví dụ minh hoạ từng bước | |
| **6** | **Thử nghiệm và đánh giá kết quả** | |
| 6.1 | Môi trường thử nghiệm | |
| 6.2 | Quy trình huấn luyện DQN | |
| 6.3 | Kết quả huấn luyện thực tế | |
| 6.4 | Kiểm thử tự động | |
| 6.5 | Đánh giá chủ quan khi chơi thử | |
| **7** | **Kết luận và hướng phát triển** | |
| | Tài liệu tham khảo | |
| | Phụ lục | |
| | **Tóm tắt — Kết hợp hai thuật toán** | |

---

## TÓM TẮT ĐỒ ÁN — KẾT HỢP HAI THUẬT TOÁN

> **Đây là nội dung trọng tâm của đồ án.** Báo cáo chi tiết ở **Chương 5**; phần này tóm tắt ngắn để đọc nhanh.

### Hai thuật toán em kết hợp là gì?

| # | Thuật toán | Nhóm AI | Vai trò trong hệ thống |
|---|------------|---------|------------------------|
| **1** | **Minimax + Alpha-Beta Pruning** | Tìm kiếm đối kháng | Duyệt cây trò chơi, nhìn trước 2–4 nước, chọn nước tối ưu theo minimax |
| **2** | **Deep Q-Network (DQN)** | Học tăng cường | Mạng CNN học Q-value, đánh giá “thế cờ này tốt/xấu thế nào” |

**Lưu ý:** Minimax và Alpha-Beta **không phải hai thuật toán riêng** — Alpha-Beta là kỹ thuật **cắt tỉa** giúp Minimax chạy nhanh hơn mà **không đổi kết quả**. Hai hướng tiếp cận thực sự khác nhau là **(Minimax + Alpha-Beta)** và **(DQN)**.

### Cách kết hợp: Hybrid Agent (`HybridAgent`)

Thay vì chạy Minimax **hoặ** DQN riêng lẻ, em ghép chúng trong **một agent duy nhất**:

```
┌─────────────────────────────────────────────────────────────┐
│  HYBRID AGENT = Minimax (khung tìm kiếm) + DQN (đánh giá lá) │
├─────────────────────────────────────────────────────────────┤
│  Bước 1: Luật chiến thuật — thắng ngay / chặn thua ngay     │
│  Bước 2: Minimax + Alpha-Beta duyệt cây (depth 2–3)         │
│  Bước 3: Tại mỗi NODE LÁ (depth = 0):                       │
│          score = 55% × heuristic + 45% × Q-value (DQN)      │
│  Bước 4: Alpha-Beta truyền điểm ngược → chọn nước tốt nhất   │
└─────────────────────────────────────────────────────────────┘
```

**Tại sao kết hợp theo cách này?**

- **Minimax** giúp AI **nghĩ về phản ứng đối thủ** (nhìn xa vài nước) — DQN thuần không làm được tốt.
- **DQN** giúp **đánh giá thế cờ phức tạp** ở điểm dừng tìm kiếm — heuristic viết tay khó cover hết mẫu.
- **Heuristic (55%)** giữ AI **ổn định** khi DQN chưa train hoặc dự đoán sai.

**Code thực tế** (`ai/hybrid_agent.py`): class `HybridAgent` **kế thừa** `MinimaxAgent`, chỉ **ghi đè** hàm `_evaluate_leaf()` — không viết lại toàn bộ Minimax.

```python
# Công thức trộn tại node lá (rút gọn từ hybrid_agent.py)
score = 0.55 * heuristic + 0.45 * dqn_scaled
```

**Fallback:** Nếu chưa có file `models/dqn_15.pth` → Hybrid tự chuyển về **Minimax thuần** (chỉ heuristic), không dùng mạng ngẫu nhiên.

---

# CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI

## 1.1. Lý do chọn đề tài

Cờ Caro (Gomoku) là trò chơi dân gian quen thuộc với sinh viên Việt Nam. Ai cũng từng chơi trên giấy kẻ ô hoặc bảng đen lớp học. Tuy luật chơi chỉ cần vài câu là hiểu, nhưng để **máy tính chơi giỏi** lại là bài toán AI không hề đơn giản.

Trong môn Trí tuệ nhân tạo, em được học hai nhóm phương pháp giải bài toán trò chơi:

1. **Tìm kiếm có thông tin (Informed Search):** Minimax, Alpha-Beta — lập kế hoạch bằng cách mô phỏng các nước đi tương lai trên cây trò chơi.
2. **Học tăng cường (Reinforcement Learning):** Agent tự học qua thử-sai, tích luỹ kinh nghiệm từ phần thưởng — Deep Q-Network (DQN) là biến thể dùng mạng nơ-ron sâu.

Em thấy hầu hết báo cáo mẫu trên mạng (123doc, ICTU Repository, topcode.vn) chỉ dừng ở **Minimax + Alpha-Beta + heuristic**. Điều đó đủ cho đồ án cơ bản, nhưng em muốn đi xa hơn một chút: **thêm DQN** và quan trọng hơn là **tìm cách kết hợp** hai hướng tiếp cận này trong một agent duy nhất — gọi là **Hybrid Agent**.

Ngoài ra, em chọn Python vì có sẵn NumPy, PyTorch, FastAPI — phù hợp để vừa làm AI vừa làm giao diện web demo nhanh.

## 1.2. Mục tiêu nghiên cứu

### Mục tiêu tổng quát

Xây dựng hệ thống AI chơi Cờ Caro hoàn chỉnh, ứng dụng đồng thời thuật toán tìm kiếm đối kháng và học sâu, có giao diện cho phép người dùng chơi trực tiếp với máy.

### Mục tiêu cụ thể

| STT | Mục tiêu | Tiêu chí đạt |
|-----|----------|--------------|
| 1 | Môi trường game | Class `CaroEnv` chạy độc lập, có unit test |
| 2 | AI Minimax | Alpha-Beta + heuristic pattern + luật tactical |
| 3 | AI DQN | CNN ước lượng Q-value, pipeline train, lưu checkpoint |
| 4 | AI Hybrid | Minimax depth 2–3, DQN đánh giá node lá |
| 5 | Giao diện | Web (FastAPI) + Desktop (Pygame), chọn AI/độ khó |
| 6 | Kiểm thử | pytest cho logic game, AI, hybrid |

## 1.3. Phạm vi đồ án

**Trong phạm vi:**

- Bàn cờ 10×10 và 15×15, luật thắng 5 quân liên tiếp.
- Ba loại AI: Minimax, DQN, Hybrid.
- Bốn mức độ khó: Dễ / Trung bình / Khó / Chuyên gia.
- Chế độ chơi: Người vs Người, Người vs AI, AI vs AI.
- Huấn luyện DQN: self-play, đấu với Minimax, và học online từ ván Người vs AI.

**Ngoài phạm vi** (để hướng phát triển sau):

- Cờ Caro chuẩn quốc tế với luật cấm (Renju).
- Monte Carlo Tree Search (MCTS), AlphaZero.
- Đánh giá Elo chính thức giữa các agent.
- Triển khai production / multiplayer online.

## 1.4. Phương pháp thực hiện

Quy trình nhóm em làm theo các bước:

```
Tìm hiểu lý thuyết (giáo trình + AIMA + báo cáo mẫu)
        ↓
Đọc và phân tích source code dự án
        ↓
Thiết kế kiến trúc tách lớp (core / ai / ui / web)
        ↓
Cài đặt từng agent + unit test
        ↓
Huấn luyện DQN + đánh giá
        ↓
Tích hợp Hybrid + kiểm thử tổng thể
        ↓
Viết báo cáo và demo
```

Công cụ sử dụng:

| Công cụ | Phiên bản / Ghi chú |
|---------|---------------------|
| Python | 3.12 |
| NumPy | Ma trận bàn cờ int8 |
| PyTorch | Mạng DQN, huấn luyện |
| FastAPI | Server web |
| Pygame | Giao diện desktop |
| pytest | Unit test |
| ruff + mypy | Lint và kiểm tra kiểu |

## 1.5. Chiến lược kết hợp Minimax và DQN (tóm tắt sớm)

Nhiều báo cáo mẫu chỉ trình bày **Minimax + Alpha-Beta** rồi dừng. Đồ án của em có **bước thứ ba** — **Hybrid Agent** — là phần **kết hợp thực sự** giữa hai hướng tiếp cận AI:

| Thành phần | Lấy từ đâu | Làm gì trong Hybrid |
|------------|------------|---------------------|
| Cây tìm kiếm | Minimax + Alpha-Beta | Duyệt nước ứng viên, tính phản ứng đối thủ |
| Hàm lượng giá lá | Heuristic pattern | 55% trọng số — ổn định, không cần train |
| Hàm lượng giá lá | DQN (CNN) | 45% trọng số — học từ kinh nghiệm ván chơi |
| Luật tức thì | `find_tactical_move()` | Chạy **trước** cả Minimax lẫn DQN |

**Luồng ra quyết định một lượt của Hybrid:**

1. Có nước thắng/chặn thua ngay? → Đánh luôn (không cần search).
2. Không → Minimax Alpha-Beta depth 2–3 trên nước ứng viên.
3. Mỗi khi search chạm **node lá** → gọi `_evaluate_leaf()`: trộn heuristic + Q-value DQN.
4. Trả về nước có minimax value cao nhất.

Chi tiết công thức, sơ đồ cây, ví dụ số và code: **Chương 5**.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

## 2.1. Trò chơi hai người và cây trò chơi

Cờ Caro thuộc loại **trò chơi hai người, thông tin đầy đủ, không ngẫu nhiên, lượt luân phiên, tổng điểm bằng không** (two-player, perfect information, deterministic, sequential, zero-sum).

Ta có thể mô hình hoá bằng:

- **Trạng thái S:** cấu hình bàn cờ hiện tại + lượt đi.
- **Hành động A(s):** tập các ô trống có thể đặt quân.
- **Hàm chuyển T(s,a):** trạng thái mới sau khi đặt quân.
- **Hàm tiện ích U(s):** +1 nếu ta thắng, −1 nếu thua, 0 nếu hòa.

**Cây trò chơi** (game tree) là cấu trúc cây mà:

- **Gốc** = trạng thái hiện tại.
- **Node con** = các trạng thái có thể đạt được sau một nước đi.
- **Lá** = trạng thái kết thúc (có người thắng hoặc hòa).

Vấn đề: nhánh phân nhánh theo cấp số mũ. Bàn 15×15, mỗi lượt có thể ~50–100 nước ứng viên, depth 4 → hàng chục nghìn đến triệu node. Vì vậy ta **không duyệt hết** mà dùng depth-limited search + heuristic + cắt tỉa.

## 2.2. Thuật toán Minimax

Minimax giả định cả hai bên đều chơi **tối ưu** (optimal play).

**Quy tắc:**

- **MAX** (bên ta): chọn nước đi sao cho điểm Minimax **lớn nhất**.
- **MIN** (đối thủ): chọn nước đi sao cho điểm Minimax **nhỏ nhất**.

**Đệ quy:**

```
MINIMAX(s) =
    UTILITY(s)                    nếu s kết thúc
    max_a MINIMAX(T(s,a))         nếu lượt MAX và depth > 0
    min_a MINIMAX(T(s,a))         nếu lượt MIN và depth > 0
    EVAL(s)                       nếu depth = 0 (node lá)
```

**Ví dụ minh hoạ đơn giản** (bàn nhỏ, depth = 2):

```
                    [Gốc: MAX chọn]
                   /       |       \
               nước A   nước B   nước C
               /  \      /  \      /  \
          [MIN] [MIN] [MIN] [MIN] [MIN] [MIN]
           3    5     2    8     1    4
           
MAX chọn nhánh có min con lớn nhất:
  A → min(3,5) = 3
  B → min(2,8) = 2
  C → min(1,4) = 1
→ Chọn A (điểm 3)
```

Trong thực tế Cờ Caro, giá trị lá không phải số nhỏ mà là điểm heuristic (có thể lên hàng nghìn).

## 2.3. Cắt tỉa Alpha-Beta

Alpha-Beta **không thay đổi** nước đi cuối cùng so với Minimax, nhưng **bỏ qua** các nhánh chắc chắn không được chọn.

**Hai biến:**

- **α:** giá trị tốt nhất MAX đã có (cận dưới).
- **β:** giá trị tốt nhất MIN đã có (cận trên).

**Beta cut (ở node MAX):** nếu giá trị con ≥ β → MIN sẽ không chọn nhánh cha → dừng duyệt các con còn lại.

**Alpha cut (ở node MIN):** nếu giá trị con ≤ α → MAX sẽ không chọn nhánh cha → dừng duyệt.

**Move ordering:** sắp xếp nước đi theo heuristic trước khi duyệt → tăng xác suất cắt tỉa sớm. Trong code, `move_priority()` đảm nhiệm việc này.

**Transposition Table (TT):** cache kết quả đã tính cho `(bàn cờ, depth, người chơi)` — tránh tính lại khi cùng trạng thái xuất hiện qua đường đi khác (transposition).

**So sánh số node** (minh hoạ lý thuyết):

| Thuật toán | Depth 4, branching ~10 | Ghi chú |
|------------|------------------------|---------|
| Minimax thuần | ~10⁴ = 10,000 node | Duyệt hết |
| Alpha-Beta (ordering tốt) | ~10²–10³ node | Cắt sớm |
| Alpha-Beta + TT + giới hạn nhánh | Còn ít hơn nữa | Như code thực tế |

## 2.4. Hàm lượng giá (Heuristic)

Khi tìm kiếm dừng ở depth = 0 mà ván chưa kết thúc, ta cần **ước lượng** ai đang lợi thế.

Dự án dùng **pattern matching** trên mọi hàng, cột, đường chéo:

| Mẫu | Điểm | Ý nghĩa chiến thuật |
|-----|------|----------------------|
| `11111` | 1,000,000 | Thắng |
| `011110` | 50,000 | Tứ mở — đối thủ buộc chặn |
| `211110` / `011112` | 5,000 | Tứ kín — một đầu bị chặn |
| `011100` / `001110` | 3,000 | Tam mở |
| `011010` / `010110` | 1,500 | Tam nhảy |
| `01100` / `00110` | 200 | Hai mở |
| `0100` / `0010` | 20 | Một mở |

**Công thức:**

```
EVAL(board, player) = score_player(board, player) − score_player(board, opponent)
```

**Luật chiến thuật tức thì** (chạy trước Minimax):

| Thứ tự | Hành động |
|--------|-----------|
| 1 | Thắng ngay (`find_winning_move`) |
| 2 | Chặn thua ngay (`find_blocking_move`) |
| 3 | Tấn công tứ/tam mở (nếu bật aggressive) |
| 4 | Chặn tứ mở đối thủ |
| 5 | Chặn tam mở đối thủ |

Em nhận xét: phần này rất quan trọng thực tế. Nhiều đồ án chỉ có Minimax depth 2 mà không có luật tactical sẽ **bỏ lỡ nước chặn 4** — trông rất "ngu" dù lý thuyết đúng.

## 2.5. Học tăng cường và Deep Q-Network

### 2.5.1. Khung Reinforcement Learning

Agent tương tác với **môi trường** qua chu kỳ:

```
Quan sát trạng thái s → Chọn hành động a → Nhận phần thưởng r → Trạng thái mới s'
```

Mục tiêu: học **chính sách (policy)** tối đa hoá tổng phần thưởng kỳ vọng:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

Với γ = 0.99 (hệ số chiết khấu trong `config.py`).

### 2.5.2. Q-Learning và DQN

**Q(s,a)** = giá trị kỳ vọng nếu ở trạng thái s, chọn a, rồi chơi tối ưu về sau.

**Bellman optimality:**

```
Q*(s,a) = E[ r + γ · max_{a'} Q*(s', a') ]
```

Với bàn cờ lớn, không thể lưu bảng Q cho mọi (s,a). **DQN** dùng mạng nơ-ron Q(s,a; θ) xấp xỉ hàm Q.

### 2.5.3. Kiến trúc mạng CNN

```
Input (3, H, W)
  → Conv 3→64, ReLU
  → Conv 64→128, ReLU
  → Conv 128→128, ReLU
  → Flatten
  → Linear → 256, ReLU
  → Linear → H×W  (Q-value mỗi ô)
```

**Mã hoá trạng thái 3 kênh:**

| Kênh | Nội dung |
|------|----------|
| 0 | Quân của ta |
| 1 | Quân đối thủ |
| 2 | Ô trống |

### 2.5.4. Kỹ thuật ổn định huấn luyện

| Kỹ thuật | Tham số trong dự án | Vai trò |
|----------|---------------------|---------|
| Experience Replay | Buffer 50,000 | Phá correlation giữa các bước liên tiếp |
| Target Network | Sync mỗi 200 bước | Target Q ổn định hơn |
| Epsilon-greedy | 1.0 → 0.05, decay 0.9995 | Cân bằng khám phá / khai thác |
| Huber Loss | SmoothL1Loss | Robust với outlier |
| Gradient clipping | max_norm = 1.0 | Tránh bùng nổ gradient |
| Reward shaping | +0.35 tứ mở, +0.12 tam mở | Hướng dẫn tấn công sớm |

### 2.5.5. Hàm phần thưởng

| Sự kiện | Reward |
|---------|--------|
| Thắng | +1.0 |
| Thua | −1.0 |
| Hòa | 0.0 |
| Tạo tứ mở (chưa thắng) | +0.35 |
| Tạo tam mở | +0.12 |
| Mỗi nước đi | −0.005 |

Phần thưởng âm nhỏ mỗi nước khuyến khích AI **thắng nhanh** thay vì kéo dài ván.

---

# CHƯƠNG 3. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

## 3.1. Yêu cầu chức năng và phi chức năng

### Yêu cầu chức năng

| ID | Yêu cầu | Mô tả |
|----|---------|-------|
| F01 | Chơi cờ cơ bản | Đặt quân, kiểm tra thắng/hòa, highlight nước cuối |
| F02 | Chọn kích thước bàn | 10×10 hoặc 15×15 |
| F03 | Chọn loại AI | Minimax / DQN / Hybrid |
| F04 | Chọn độ khó | Dễ (1) → Chuyên gia (4) |
| F05 | Chế độ chơi | PvP, PvAI, AIvAI |
| F06 | Cài đặt luật | Chặn 2 đầu, chế độ tấn công AI |
| F07 | Huấn luyện DQN | CLI `train.py`, lưu checkpoint |
| F08 | Học online (PvA) | AI cập nhật model sau ván thua/thắng với người chơi |
| F09 | HUD | Xác suất thắng ước lượng, thông tin lượt |

### Yêu cầu phi chức năng

| ID | Yêu cầu | Tiêu chí |
|----|---------|----------|
| NF01 | Thời gian phản hồi AI | Minimax < 5s; Hybrid timeout 45s |
| NF02 | Tách lớp | `core/` và `ai/` không import UI |
| NF03 | Kiểm thử | pytest pass toàn bộ |
| NF04 | Mở rộng | Thêm agent mới qua `Agent` base class |
| NF05 | Đa nền tảng | Web (Chrome) + Desktop (Pygame) |

## 3.2. Kiến trúc phần mềm

```
┌──────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                     │
│   web/server.py + static/    │    ui/ (Pygame desktop)    │
├──────────────────────────────────────────────────────────┤
│                      AI LAYER                             │
│  factory.py → MinimaxAgent | DQNAgent | HybridAgent      │
│  heuristic.py │ board_encoder.py │ dqn_trainer.py        │
│  online_learner.py (học từ ván PvA)                       │
├──────────────────────────────────────────────────────────┤
│                      CORE LAYER                           │
│              CaroEnv │ constants │ config.py              │
└──────────────────────────────────────────────────────────┘
```

**Nguyên tắc thiết kế em ghi nhận khi đọc code:**

1. **Dependency inversion:** UI phụ thuộc interface `Agent`, không phụ thuộc implementation cụ thể.
2. **Single source of truth:** Mọi luật game nằm trong `CaroEnv`.
3. **Configuration centralization:** Magic numbers tập trung ở `config.py`.

## 3.3. Thiết kế môi trường game `CaroEnv`

**Class diagram (mô tả):**

```
CaroEnv
├── Attributes
│   ├── board: ndarray (size × size, int8)
│   ├── current_player: Player (X | O)
│   ├── winner, done, last_move, winning_line
│   └── _undo_stack: list  (phục vụ push/pop)
├── Lifecycle
│   ├── reset() → Board
│   └── step(move) → (board, winner, done)
├── Search helpers
│   ├── push(move) / pop()
│   ├── clone()
│   ├── legal_moves()
│   └── candidate_moves(radius)
└── Win detection
    └── _check_win_from(row, col, player)  [O(5) per move]
```

**Luật chặn 2 đầu:** nếu đúng 5 quân liên tiếp mà **cả hai đầu** đều bị đối phương hoặc biên bàn chặn → **không tính thắng**. Hàm `_check_win_from` kiểm tra điều này khi `double_end_block_rule=True`.

**Thu hẹp không gian tìm kiếm:** `candidate_moves(radius=2)` chỉ trả về ô trống trong bán kính 2 quanh quân đã đặt. Bàn trống → trả về ô trung tâm.

## 3.4. Thiết kế các Agent AI

**Interface chung (`Agent`):**

```python
class Agent:
    name: str
    def get_move(self, env: CaroEnv) -> Move: ...
    def get_win_probability(self, env, for_player=None) -> float: ...
```

**Ba implementation:**

| Agent | Kế thừa | Điểm đặc biệt |
|-------|---------|---------------|
| MinimaxAgent | Agent | Alpha-Beta, TT, tactical |
| DQNAgent | Agent | CNN inference, epsilon-greedy |
| HybridAgent | MinimaxAgent | Ghi đè `_evaluate_leaf()` |

**Factory pattern** (`ai/factory.py`):

```python
create_agent(ai_type, difficulty, board_size) → Agent
```

UI không cần biết cách khởi tạo từng loại — chỉ truyền enum `AIType` và `Difficulty`.

---

# CHƯƠNG 4. CÀI ĐẶT VÀ ĐỌC HIỂU SOURCE CODE

Phần này em trình bày chi tiết cách đọc code — theo thứ tự em khuyên khi lần đầu clone repo.

## 4.1. Cấu trúc thư mục dự án

```
DoAn/
├── core/
│   ├── caro_env.py          # Môi trường game — ĐỌC ĐẦU TIÊN
│   └── constants.py         # DIRECTIONS, kiểu Move/Board
├── ai/
│   ├── base_agent.py        # Interface Agent
│   ├── heuristic.py         # Heuristic + tactical — ĐỌC THỨ HAI
│   ├── minimax_agent.py     # Minimax + Alpha-Beta
│   ├── dqn_model.py         # Kiến trúc CNN
│   ├── board_encoder.py     # Mã hoá tensor
│   ├── dqn_agent.py         # Suy luận DQN
│   ├── dqn_trainer.py       # Vòng lặp train
│   ├── online_learner.py    # Học online từ ván Người vs AI
│   ├── replay_buffer.py     # Experience replay
│   ├── hybrid_agent.py      # Kết hợp Minimax + DQN
│   ├── factory.py           # Tạo agent
│   ├── evaluate.py          # Đấu thử agent vs agent
│   └── win_probability.py   # Ước lượng xác suất thắng
├── web/                     # FastAPI + HTML/CSS/JS
├── ui/                      # Pygame desktop
├── tests/                   # pytest
├── models/                  # Checkpoint .pth
├── config.py                # Hằng số toàn cục
├── train.py                 # CLI huấn luyện
├── web_main.py              # Chạy server web
└── main.py                  # Chạy desktop
```

## 4.2. Module `core/` — logic game

### 4.2.1. Biểu diễn bàn cờ

```python
# config.py
class Player(IntEnum):
    EMPTY = 0
    X = 1      # Đi trước
    O = 2
```

Ma trận `board` dtype `int8` — tiết kiệm bộ nhớ, phép toán NumPy nhanh.

### 4.2.2. Hàm `step()` vs `push()`/`pop()`

| Hàm | Dùng khi | Side effect |
|-----|----------|-------------|
| `step(move)` | Chơi thật (UI, train) | Trả bản sao board |
| `push(move)` | Minimax search | Mutate + ghi undo stack |
| `pop()` | Sau khi duyệt xong nhánh | Khôi phục trạng thái |

**Code pattern trong Minimax:**

```python
for move in moves:
    env.push(move)
    try:
        child_score, _ = self._alpha_beta(env, depth - 1, alpha, beta, ai_player)
    finally:
        env.pop()
```

Em thấy đây là tối ưu quan trọng nhất cho hiệu năng Minimax — clone cả env mỗi nhánh sẽ chậm gấp nhiều lần.

### 4.2.3. Kiểm tra thắng

Quét 4 hướng `(1,0), (0,1), (1,1), (1,-1)` từ nước vừa đánh, đếm quân liên tiếp hai phía. Tổng ≥ 5 → thắng, lưu `winning_line` cho UI highlight.

## 4.3. Module `ai/` — Minimax Agent

### 4.3.1. Luồng `get_move()`

```
get_move(env)
    │
    ├─► find_tactical_move() ──► có? → return ngay
    │
    ├─► _ordered_moves() ──► candidate_moves + sort by move_priority
    │
    └─► _alpha_beta(depth) ──► return best_move
```

### 4.3.2. Hàm `_alpha_beta()` — đọc từng khối

**Điều kiện dừng:**

```python
if env.done or depth == 0:
    return self._evaluate_leaf(env, ai_player), None
```

**Tra Transposition Table:**

```python
tt_key = (env.board.tobytes(), depth, ai_player, current_player)
if cached.depth >= depth:
    return cached.score, cached.best_move
```

**Nhánh MAX (maximizing = current is ai_player):**

```python
value = -inf
for move in moves:
    env.push(move)
    child_score = ...  # đệ quy hoặc evaluate nếu done
    env.pop()
    value = max(value, child_score)
    alpha = max(alpha, value)
    if alpha >= beta:
        break  # Beta cut
```

**Nhánh MIN:** tương tự với `min`, cập nhật `beta`, alpha cut khi `alpha >= beta`.

### 4.3.3. Map độ khó

| Difficulty | Enum | Depth (ply) |
|------------|------|-------------|
| Dễ | EASY = 1 | 1 |
| Trung bình | MEDIUM = 2 | 2 |
| Khó | HARD = 3 | 3 |
| Chuyên gia | EXPERT = 4 | 4 |

## 4.4. Module `ai/` — DQN Agent

### 4.4.1. Luồng suy luận

```
get_move(env)
    │
    ├─► find_tactical_move() ──► có? → return
    │
    ├─► epsilon-greedy? ──► random trong candidate_moves
    │
    └─► forward CNN → argmax Q (masked legal) → action_to_move
```

### 4.4.2. Mã hoá và hành động

```python
# board_encoder.py
state = encode_board(board, player)   # (3, H, W)
action = row * board_size + col       # phẳng hoá
q = network(state)                    # (H*W,)
q[~legal_mask] = -inf
best_action = argmax(q)
```

### 4.4.3. Nạp model

Tự động tìm `models/dqn_{board_size}.pth`. Checkpoint chứa:

```python
{
    "board_size": 15,
    "state_dict": ...  # trọng số mạng
}
```

## 4.5. Module `ai/` — huấn luyện DQN

### 4.5.1. Replay Buffer

```python
@dataclass
class Transition:
    state: (3, H, W)
    action: int
    reward: float
    next_state: (3, H, W)
    done: bool
```

Buffer dạng `deque` FIFO, capacity 50,000. Sample uniform ngẫu nhiên batch 64.

### 4.5.2. Một bước gradient

```python
q_values = policy_net(states).gather(1, actions)
targets = rewards + gamma * target_net(next_states).max(1) * (1 - dones)
loss = SmoothL1Loss(q_values, targets)
loss.backward()
clip_grad_norm_(..., max_norm=1.0)
optimizer.step()
```

### 4.5.3. Hai chế độ train

| Mode | CLI | Mô tả |
|------|-----|-------|
| Self-play | `--mode selfplay` | X và O cùng policy + ε-greedy |
| vs Minimax | `--mode minimax --opponent-depth 2` | X = DQN học, O = Minimax cố định |

Em train chủ yếu mode **vs Minimax** vì DQN có đối thủ ổn định để học, thay vì self-play dễ hội tụ về nước đi yếu.

### 4.5.4. Lệnh train thường dùng

```bash
python train.py --board-size 15 --episodes 3000
python train.py --mode minimax --opponent-depth 2 --episodes 800
python train.py --resume models/dqn_15.pth --episodes 500
python scripts/auto_train.py   # tự train nếu chưa có models/dqn_{size}.pth
```

### 4.5.5. Ba nguồn dữ liệu huấn luyện

Em tổ chức pipeline DQN theo **ba nguồn kinh nghiệm** độc lập, cùng định dạng `Transition` nhưng khác cách thu thập:

| # | Nguồn | Module / lệnh | Khi nào chạy |
|---|--------|---------------|--------------|
| 1 | **Self-play** | `train.py --mode selfplay` | X và O cùng policy + ε-greedy, tự sinh ván |
| 2 | **Đấu Minimax** | `train.py --mode minimax` | DQN (X) học từ đối thủ Minimax cố định (O) |
| 3 | **Học từ người chơi** | `ai/online_learner.py` | Sau mỗi ván PvA (chế độ DQN / Hybrid) |

Ba nguồn đều ghi vào **replay buffer** và dùng chung công thức Bellman (mục 4.5.2), nhưng **hai buffer vật lý khác nhau**:

- **Buffer offline** (`DQNTrainer` trong `train.py`): train hàng loạt, tối thiểu 256 mẫu mới gradient.
- **Buffer online** (`OnlineLearner`, singleton theo `board_size`): tích lũy qua nhiều ván PvA, tối thiểu 8 mẫu mới gradient.

Cả hai cùng **ghi đè** file checkpoint `models/dqn_{size}.pth` — đây là cầu nối giữa train offline và học online.

**Lưu ý:** Ba kênh tensor `(3, H, W)` trong `Transition.state` là **mã hoá bàn cờ** (quân ta / đối / trống), **không phải** “ba nguồn dữ liệu” ở trên.

### 4.5.6. Cơ chế học: DQN học cái gì?

DQN **không ghi nhớ từng ván** như bộ nhớ ngắn hạn. Nó học một **hàm đánh giá**:

> Với bàn cờ hiện tại, nếu đặt quân vào ô *a* thì kỳ vọng thắng/thua (tổng phần thưởng tương lai) là bao nhiêu?

Hàm đó được mạng CNN (`DQNNetwork`) xấp xỉ: đầu vào tensor `(3, H, W)`, đầu ra vector Q dài `H×W` — **mỗi ô một giá trị Q**. Q càng cao → nước đi càng “đáng” theo kinh nghiệm đã train.

**Một kinh nghiệm (Transition) gồm 5 trường:**

| Trường | Ý nghĩa |
|--------|---------|
| `state` | Bàn cờ **trước** nước đi (góc nhìn người vừa đi) |
| `action` | Ô được chọn (`row * size + col`) |
| `reward` | Phần thưởng ngay sau nước đi |
| `next_state` | Bàn cờ **sau** nước đi (góc nhìn người sắp đi) |
| `done` | Ván đã kết thúc chưa |

**Một bước gradient** (`dqn_trainer._optimize`) làm việc sau:

1. Lấy ngẫu nhiên mini-batch từ buffer.
2. Mạng **policy** dự đoán Q cho hành động đã thực hiện: `q_values`.
3. Mạng **target** (bản sao, cập nhật chậm) tính mục tiêu Bellman:

   `targets = reward + γ × max Q(s') × (1 − done)` với γ = 0.99.

4. Giảm sai số Huber giữa `q_values` và `targets` → chỉnh trọng số CNN.

**Ví dụ trực quan:** Khi người chơi thắng AI, nước AI **cuối cùng** nhận `reward = −1.0`. Mạng được “dạy” rằng tại trạng thái trước nước đó, hành động vừa chọn có giá trị kỳ vọng rất thấp — lần sau gặp thế cờ tương tự, Q của ô đó sẽ giảm so với các ô khác.

### 4.5.7. Học online từ ván Người vs AI

Module `ai/online_learner.py` cho phép AI **cập nhật model ngay sau khi chơi với người**, không cần chạy lại `train.py`.

**Luồng xử lý (3 bước):**

```
Trong ván PvA:
    GameMoveRecorder ghi từng nước AI (state, action, reward từng bước)

Hết ván (không hòa, không undo):
    Bước 1 — Ghi nhận:  build_transitions() chuẩn hoá reward nước cuối
    Bước 2 — Học:       đưa vào buffer + chạy 32 bước gradient (nếu buffer ≥ 8)
    Bước 3 — Cải tiến:  lưu models/dqn_{size}.pth + nạp lại trọng số agent đang chơi
```

**Reward nước cuối ván (online):**

| Kết quả ván | Outcome | Reward nước AI cuối |
|-------------|---------|---------------------|
| Người thắng | `ai_loss` | **−1.0** (AI học từ sai lầm) |
| AI thắng | `ai_win` | **+1.0** (củng cố chiến thắng) |
| Hòa | `draw` | Không học |

**Siêu tham số học online** (`config.py`):

| Tham số | Giá trị | Vai trò |
|---------|---------|---------|
| `ONLINE_LEARN_ENABLED` | `True` | Bật/tắt học online |
| `ONLINE_LEARN_GRADIENT_STEPS` | 32 | Số bước gradient sau mỗi ván |
| `ONLINE_LEARN_MIN_SAMPLES` | 8 | Tối thiểu mẫu trong buffer mới chạy gradient |

**Điều kiện kích hoạt:**

| Điều kiện | Học? |
|-----------|------|
| Chế độ **Người vs AI** + AI loại **DQN / Hybrid** | ✅ |
| Chế độ **Minimax** thuần | ❌ (không có mạng neural) |
| Người **Undo** trong ván | ❌ (`invalidate()` huỷ bản ghi) |
| Buffer < 8 mẫu | Chỉ ghi nhớ (`buffered_only`), **chưa** lưu model |

**Tích hợp giao diện:**

- **Web** (`web/session.py`): học đồng bộ ngay khi ván kết thúc; JSON trả về trường `online_learn`.
- **Desktop** (`ui/screens/game_screen.py`): học trong thread nền để không treo UI; hiện thông báo trên HUD.

### 4.5.8. Áp dụng kiến thức đã học khi chơi

Kiến thức sau train **không nằm trong code Python** mà nằm trong **trọng số file `.pth`**. Khi chơi, AI nạp file đó và dùng mạng để chọn nước.

**Hai instance mạng tách biệt:**

| Instance | Dùng khi | File liên quan |
|----------|----------|----------------|
| `DQNTrainer.policy_net` | Train offline / học online | Ghi ra `.pth` qua `save_agent()` |
| `DQNAgent.network` | Suy luận khi chơi | Đọc từ `.pth` qua `load()` |

Hai mạng **không chia sẻ RAM** — đồng bộ qua file checkpoint. Sau học online, `OnlineLearner.reload_agent_weights()` gọi `dqn.load(path)` trên agent đang chơi; với Hybrid còn xóa `_eval_cache`.

**Luồng chọn nước — DQN thuần** (`dqn_agent.get_move`):

```
1. find_tactical_move()     → thắng ngay / chặn thua ngay? → return (luật cứng, không qua DQN)
2. epsilon-greedy           → random trong candidate_moves (theo độ khó)
3. encode_board()           → tensor (3, H, W)
4. network.forward()        → Q-value mọi ô
5. mask ô không hợp lệ      → q[~legal] = -inf
6. argmax Q                 → chọn ô Q cao nhất
```

**Luồng chọn nước — Hybrid** (`hybrid_agent`):

```
1. Tactical (giống trên)
2. Minimax + Alpha-Beta (depth 2–3)
3. Tại node lá: score = 55% × heuristic + 45% × Q-value (DQN)
4. Truyền điểm ngược → chọn nước minimax tốt nhất
```

Nếu chưa có file `.pth`, Hybrid **fallback** về Minimax thuần (chỉ heuristic) — tránh mạng ngẫu nhiên làm AI yếu hơn.

**Khi nào thay đổi có hiệu lực?**

| Tình huống | Model mới có hiệu lực? |
|------------|------------------------|
| Học online xong, `model_saved = True` | ✅ Ngay ván sau (cùng session, agent được `reload`) |
| Bấm **Chơi lại** (web/desktop) | ✅ Session/agent mới → `create_agent()` → `load()` từ `.pth` |
| Chỉ `buffered_only` (buffer < 8) | ❌ Dữ liệu nằm trong buffer online, **chưa** ghi file — ván sau vẫn dùng model cũ |
| Chơi **Minimax** thuần | ❌ Không dùng DQN |

**Sơ đồ tổng thể train → chơi:**

```
[train.py / online_learner]
        │
        ▼
  Replay Buffer → Gradient → policy_net (cập nhật trọng số)
        │
        ▼
  save_agent() → models/dqn_15.pth
        │
        ▼
  DQNAgent.load() / reload_agent_weights()
        │
        ▼
  get_move(): tactical → (ε-greedy) → argmax Q → nước đi
```

Em kiểm thử nhanh: sau một ván học online (buffer đủ mẫu), trọng số mạng thay đổi đo được và Q-value trên cùng một thế cờ khác biệt rõ so với trước khi học — xác nhận pipeline ghi file → nạp lại → suy luận hoạt động đúng.

## 4.6. Giao diện người dùng

### 4.6.1. Web UI (khuyến nghị)

```bash
python web_main.py
# Mở http://127.0.0.1:8000
```

**Luồng request:**

```
Browser click ô (row, col)
    → POST /api/move
    → session.py quản lý CaroEnv
    → step(move người chơi)
    → agent.get_move(env) nếu PvAI
    → ghi nước AI vào GameMoveRecorder
    → hết ván: online_learner học + lưu model (PvA + DQN/Hybrid)
    → JSON: board, last_move, winner, ai_move, win_prob, online_learn
    → app.js vẽ lại bàn cờ (+ thông báo "AI đã học từ thất bại" nếu có)
```

### 4.6.2. Desktop UI (Pygame)

```bash
python main.py
```

- Menu chọn chế độ, AI, độ khó.
- Game screen: click đặt quân, HUD sidebar, animation thắng.
- Phím `R` chơi lại, `ESC` về menu.
- PvA + DQN/Hybrid: sau ván thua/thắng, AI học online (thread nền) và hiện thông báo trên HUD.

### 4.6.3. Dev server

```bash
make dev   # hot reload qua dev.py
```

---

# CHƯƠNG 5. KẾT HỢP HAI THUẬT TOÁN: HYBRID AGENT

> **Chương quan trọng nhất của báo cáo.** Nếu chỉ đọc một chương về “kết hợp Minimax và DQN”, hãy đọc chương này.  
> (Phần tóm tắt nhanh nằm ngay sau Mục lục — mục **“TÓM TẮT ĐỒ ÁN — KẾT HỢP HAI THUẬT TOÁN”**.)

Đây là **phần trọng tâm** của đồ án — giải thích vì sao kết hợp, kết hợp như thế nào, và code làm gì từng dòng.

**Hai thuật toán được kết hợp:**

| Thuật toán 1 | Thuật toán 2 | Cách ghép |
|--------------|--------------|-----------|
| Minimax + Alpha-Beta (tìm kiếm cây) | DQN (mạng nơ-ron Q-value) | Minimax duyệt cây; **DQN thay thế/bổ sung heuristic tại node lá** |

## 5.1. Động lực và ý tưởng

### Bảng so sánh điểm mạnh / yếu

| | Minimax + Heuristic | DQN thuần |
|---|---------------------|-----------|
| **Mạnh** | Nhìn trước vài nước; logic minh bạch; không cần train | Học mẫu phức tạp; đánh giá nhanh (1 forward) |
| **Yếu** | Heuristic viết tay có giới hạn; depth cao → chậm | Không nhìn xa; cần nhiều data; yếu khi chưa train |

### Ý tưởng lai ghép

> **Dùng Minimax làm "bộ não chiến thuật"** (duyệt cây, tính phản ứng đối thủ),  
> **dùng DQN làm "trực giác"** (đánh giá thế cờ phức tạp ở node lá).

Minimax thuần dừng sớm ở depth 2–4 vì heuristic không đủ sâu. DQN thuần không duyệt cây nên dễ miss combo vài nước. Hybrid **ghép tầm nhìn của Minimax với khả năng học của DQN**.

## 5.2. Kiến trúc lai ghép

```
HybridAgent extends MinimaxAgent
    │
    ├── Kế thừa: _alpha_beta(), TT, move ordering, tactical
    │
    ├── Thêm: DQNAgent (inference only, ε=0)
    │
    └── Ghi đè: _evaluate_leaf()  ← ĐIỂM KẾT HỢP DUY NHẤT
```

Em chú ý thiết kế này rất gọn: **không viết lại Minimax**, chỉ thay cách chấm điểm lá — đúng tinh thần Open-Closed Principle.

### Sơ đồ cây tìm kiếm Hybrid (depth = 3, EXPERT)

```
                         [GỐC — AI (MAX) chọn nước]
                        /         |         \
                   nước m1     nước m2     nước m3      ← max_branch ≤ 14
                   /    \      /    \       /    \
              [MIN]  [MIN] [MIN] [MIN]  [MIN] [MIN]   ← depth 2: đối phản ứng
               / \    / \    ...
          [LÁ] [LÁ] [LÁ] [LÁ]                         ← depth 0
           │    │    │    │
           ▼    ▼    ▼    ▼
      0.55×H + 0.45×DQN  (mỗi lá = 1 forward CNN)
           │    │    │    │
           └────┴────┴────┘
                     │
              Alpha-Beta truyền ngược
                     │
              best_move tại gốc
```

## 5.3. Công thức đánh giá node lá

### Bước 1 — Heuristic

```python
heuristic = evaluate_position(env.winner, env.board, ai_player)
# = score(ta) - score(đối), hoặc ±10^9 nếu thắng/thua
```

### Bước 2 — DQN raw score

```python
q = dqn._predict_q_numpy(env, current_player)  # vector H*W
best_q = max(q[legal_mask])
if current_player != ai_player:
    best_q = -best_q   # đảo góc nhìn về ai_player
```

### Bước 3 — Scale Q về cùng magnitude

Q thô thường ∈ [−1, 1] hoặc nhỏ, heuristic ∈ [−10⁶, 10⁶]. Không scale thì trộn vô nghĩa.

```python
if abs(dqn_score) <= 10.0:
    scale = max(5000.0, abs(heuristic) * 0.5, 1.0)
    dqn_scaled = dqn_score * scale
```

### Bước 4 — Trộn có trọng số

```python
w = HYBRID_LEAF_HEURISTIC_WEIGHT  # 0.55
score = w * heuristic + (1 - w) * dqn_scaled
```

**Tại sao w = 0.55 (heuristic nặng hơn)?**

- Heuristic **ổn định**, không dao động theo giai đoạn train.
- DQN có thể **overfit** hoặc dự đoán sai khi chưa train đủ.
- Heuristic đảm bảo Hybrid **không tệ hơn** Minimax quá nhiều khi DQN nhiễu.

### Bước 5 — Fallback khi chưa có model

```python
if not self.dqn._model_loaded:
    return heuristic   # Thuần Minimax
    # + tăng depth lên bằng Minimax thuần (Expert = 4)
```

Tránh trường hợp mạng **ngẫu nhiên** phá hỏng cây tìm kiếm.

### Bước 6 — Cache

```python
cache_key = (board.tobytes(), current_player, ai_player)
# Tối đa 8192 entries, clear mỗi lượt get_move()
```

## 5.4. Tham số Hybrid theo độ khó

| Difficulty | Depth | Max branch | Radius | Ghi chú |
|------------|-------|------------|--------|---------|
| Dễ | 1 | 6 | 2 | Nhanh, phù hợp máy yếu |
| Trung bình | 2 | 8 | 2 | Cân bằng |
| Khó | 2 | 12 | 2 | Nhiều nhánh hơn |
| Chuyên gia | 3 | 14 | 3 | Depth max 3 (không 4) vì DQN chậm |

**Lý do Expert chỉ depth 3:** mỗi node lá = 1 forward CNN. Depth 4 × 14 nhánh × nhiều tầng → UI có thể treo **hàng chục giây đến vài phút**. Comment trong `config.py` ghi rõ điều này — em thấy đây là trade-off thực tế, không phải lỗi thiết kế.

## 5.5. Ví dụ minh hoạ từng bước

**Tình huống:** Bàn 15×15, giữa ván, AI (X) đi, chế độ Hybrid EXPERT (depth=3).

**Bước 1 — Tactical:**

```
find_tactical_move(X):
  - Thắng ngay?  Không
  - Chặn thua?   Không
  - Tấn công tứ mở? Không
  → return None → vào Minimax
```

**Bước 2 — Sinh nước ứng viên:**

```
candidate_moves(radius=3) → ~40 ô gần cụm quân
sort by move_priority → lấy top 14 (max_branch)
```

**Bước 3 — Alpha-Beta depth 3:**

Giả sử thử nước `m = (7, 8)`:

```
push(7,8) → lượt O
  O thử các nước phản ứng...
    push(o_move) → lượt X (depth 1)
      X thử tiếp... → depth 0 (LÁ)
        evaluate_leaf():
          heuristic = +4,200 (ta hơn tam mở)
          dqn_raw = +0.31 → scale ×5000 = +1,550
          score = 0.55×4200 + 0.45×1550 = +3,007
      pop
    pop
  pop
→ Giá trị minimax cho nhánh (7,8) = +2,800 (ví dụ)
```

Lặp cho 13 nhánh còn lại, chọn nhánh có giá trị cao nhất.

**Bước 4 — Kết quả:** AI đặt quân tại ô có minimax value max — vừa tính phản ứng O, vừa có "cảm giác" DQN về thế cờ cuối nhánh.

## 5.6. So sánh ba agent tổng hợp

| Tiêu chí | Minimax | DQN | Hybrid |
|----------|---------|-----|--------|
| Cần train | Không | Có | Có (fallback OK) |
| Nhìn xa | 1–4 ply | Không | 1–3 ply |
| Đánh giá | Heuristic | Q-network | 55% H + 45% DQN |
| Thời gian/lượt | 0.1–3s | ~0.05s | 1–30s |
| Ổn định | Cao | Trung bình | Cao (nhờ heuristic) |
| Phù hợp | Demo, máy yếu | Sau train | Chơi khó, đã train |

---

# CHƯƠNG 6. THỬ NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

## 6.1. Môi trường thử nghiệm

| Thành phần | Thông số |
|------------|----------|
| OS | macOS (darwin 25.3.0) |
| Python | 3.12 |
| CPU | Apple Silicon (MPS cho PyTorch) |
| Bàn cờ | 15×15 |
| Checkpoint | `models/dqn_15.pth` |

## 6.2. Quy trình huấn luyện DQN

**Siêu tham số** (từ `config.py`):

| Tham số | Giá trị |
|---------|---------|
| Learning rate | 1e-4 |
| Gamma | 0.99 |
| Batch size | 64 |
| Buffer capacity | 50,000 |
| Epsilon start → end | 1.0 → 0.05 |
| Epsilon decay | 0.9995 |
| Target sync | mỗi 200 bước |
| Train every | 4 bước |
| Min buffer (offline) | 256 transitions |
| Gradient steps (online) | 32 bước / ván |
| Min buffer (online) | 8 transitions |

**Quy trình em thực hiện (train offline):**

```
Giai đoạn 1: 800 ep vs Minimax depth=2
Giai đoạn 2: 500 ep vs Minimax depth=3 (resume checkpoint)
Giai đoạn 3: 300 ep vs Minimax depth=3 (fine-tune)
```

**Học online bổ sung:** Sau khi có checkpoint ban đầu, mỗi ván PvA thắng/thua với người chơi có thể tinh chỉnh thêm model mà không cần chạy lại toàn bộ `train.py` (xem mục 4.5.7).

## 6.3. Kết quả huấn luyện thực tế

Trích từ `models/train_log.txt`:

| Episode | ε | Loss TB | Buffer | X thắng % | Ghi chú |
|---------|---|---------|--------|-----------|---------|
| 50 | 0.589 | 0.139 | 1,059 | 54.0% | Khám phá nhiều |
| 100 | 0.330 | 1.166 | 2,219 | 53.0% | Bắt đầu khai thác |
| 200 | 0.106 | 9.518 | 4,491 | 50.5% | Eval: DQN thắng 50% vs Minimax d=2 |
| 300 | 0.332 | 8.055 | 2,203 | 59.0% | Sau resume, cải thiện |

**Nhận xét:**

1. **Tỷ lệ thắng ~50%** vs Minimax depth 2 nghĩa là DQN đã học được cách **cân bằng** với đối thủ có logic — không còn random.
2. **Loss tăng** (0.14 → 17+) là hiện tượng **bình thường** với DQN: target network và policy network liên tục thay đổi, Q-value scale lớn dần — không đồng nghĩa model tệ đi.
3. Train vs Minimax depth 3 **khó hơn** — cần thêm episode hoặc curriculum (tăng depth dần).
4. Tổng thời gian một lần train 200 ep: **~2154 giây (~36 phút)** trên máy em.

## 6.4. Kiểm thử tự động

```bash
python agent_tools.py test    # pytest
python agent_tools.py lint    # ruff + mypy
python agent_tools.py eval --games 20
```

| File test | Kiểm tra gì |
|-----------|-------------|
| `test_caro_env.py` | Luật game, thắng, hòa, push/pop |
| `test_minimax.py` | Chọn nước thắng/chặn, alpha-beta |
| `test_dqn.py` | Encode, forward, save/load |
| `test_online_learner.py` | Ghi nước AI, học online, undo invalidate |
| `test_hybrid.py` | Fallback heuristic, trộn điểm |
| `test_heuristic_tactical.py` | Thứ tự ưu tiên tactical |
| `test_threats.py` | Phát hiện tam/tứ mở |
| `test_web_session.py` | API web session |

## 6.5. Đánh giá chủ quan khi chơi thử

Em và các bạn trong nhóm chơi thử trên web UI:

| AI | Độ khó | Nhận xét |
|----|--------|----------|
| Minimax | Dễ (d=1) | Dễ thắng nếu biết tạo 2 đầu mở |
| Minimax | TB (d=2) | Bắt đầu khó, chặn tam mở ổn |
| Minimax | CGE (d=4) | Khó rõ, phản xạ tứ mở nhanh |
| DQN | Đã train | Thỉnh thoảng nước lạ, không bỏ lỡ chặn thua |
| Hybrid | CGE (d=3) | Cân bằng — chặt + thỉnh thoảng tấn công bất ngờ |
| Hybrid | Chưa train DQN | Giống Minimax (fallback) — vẫn chơi được |

---

# CHƯƠNG 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 7.1. Kết luận

Qua đồ án **Xây dựng hệ thống AI chơi Cờ Caro kết hợp Minimax + Alpha-Beta và DQN**, nhóm em đã:

1. **Nắm vững lý thuyết** tìm kiếm đối kháng (Minimax, Alpha-Beta) và học tăng cường (DQN) — nội dung trực tiếp của môn Trí tuệ nhân tạo.

2. **Cài đặt thành công** ba loại agent (Minimax, DQN, Hybrid) trên nền tảng Python với kiến trúc tách lớp rõ ràng.

3. **Kết hợp hai thuật toán** theo hướng lai ghép thực tế: Minimax duyệt cây có giới hạn, DQN đánh giá node lá — không phải chạy riêng lẻ mà **tích hợp trong cùng một pipeline ra quyết định**.

4. **Xây dựng sản phẩm hoàn chỉnh** có giao diện web/desktop, pipeline huấn luyện, unit test — đủ demo và báo cáo.

Bài học lớn nhất với em: **không có thuật toán "tốt nhất" tuyệt đối** — Minimax mạnh về logic và ổn định, DQN mạnh về học mẫu phức tạp, Hybrid là compromise hợp lý khi muốn cả hai. Quan trọng là hiểu **trade-off** giữa thời gian, độ sâu, và chất lượng đánh giá.

## 7.2. Hạn chế

- DQN với ~1,500 episode **chưa vượt** Minimax EXPERT trên bàn 15×15.
- Hybrid **chậm** do forward DQN lặp ở nhiều node lá.
- Chưa có đánh giá Elo khách quan giữa các agent.
- Reward shaping có thể chưa tối ưu cho opening/endgame.

## 7.3. Hướng phát triển

| Hướng | Mô tả |
|-------|-------|
| Curriculum learning | Tăng dần depth Minimax đối thủ khi train |
| MCTS + NN | Hướng AlphaZero — search linh hoạt hơn depth cố định |
| Prioritized replay | Ưu tiên transition "surprise" trong buffer |
| Hybrid thông minh | Chỉ gọi DQN khi heuristic score các nước gần nhau |
| Elo rating | Đo sức mạnh agent khách quan |
| Mobile UI | Responsive web hoặc app |

---

# TÀI LIỆU THAM KHẢO

1. Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson — Chương 5 (Adversarial Search), Chương 22 (Reinforcement Learning).

2. Mnih, V. et al. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602 — Paper DQN gốc.

3. AIMA Python — `games.py`: https://aima.cs.berkeley.edu/python/games.html

4. Nguyễn Văn An. *Xây dựng hệ thống chơi game cờ Caro sử dụng thuật toán Minimax và Alpha-Beta Pruning*. ICTU Repository.

5. Báo cáo mẫu — *Áp dụng Minmax và cắt tỉa Alpha-Beta xây dựng trò chơi cờ Caro*. 123docz.com.

6. Giáo trình môn **Trí tuệ nhân tạo** — Khoa CNTT, ĐHQG-HCM.

7. PyTorch Documentation — https://pytorch.org/docs/

8. Source code dự án: `core/`, `ai/`, `web/`, `train.py`, `config.py`.

---

# PHỤ LỤC

## Phụ lục A — Hướng dẫn cài đặt và chạy

```bash
# 1. Tạo môi trường
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Chạy game (web)
python web_main.py
# → http://127.0.0.1:8000

# 3. Chạy game (desktop)
python main.py

# 4. Huấn luyện DQN
python train.py --mode minimax --opponent-depth 2 --episodes 800

# 5. Kiểm thử
python agent_tools.py test
```

## Phụ lục B — Bảng hằng số `config.py` quan trọng

| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `WIN_LENGTH` | 5 | Số quân thắng |
| `HYBRID_LEAF_HEURISTIC_WEIGHT` | 0.55 | Trọng số heuristic ở node lá Hybrid |
| `AI_MOVE_TIMEOUT_SEC` | 45.0 | Timeout AI trước khi fallback |
| `DQN_GAMMA` | 0.99 | Chiết khấu phần thưởng |
| `DQN_PLAY_EPSILON[EXPERT]` | 0.0 | Không random khi chơi khó |

## Phụ lục C — Pseudo-code Hybrid đầy đủ

```python
class HybridAgent(MinimaxAgent):

    def get_move(env):
        eval_cache.clear()
        return MinimaxAgent.get_move(env)   # tactical + alpha-beta

    def _evaluate_leaf(env, ai_player):
        if env.done:
            return evaluate_position(...)   # ±10^9

        h = evaluate_position(...)

        if not dqn.model_loaded:
            return h                        # fallback

        if cache_hit:
            return cached_score

        q = dqn.predict_best_q(env, ai_player)
        q_scaled = scale(q, h)
        score = 0.55 * h + 0.45 * q_scaled
        cache.store(score)
        return score
```

## Phụ lục D — Ma trận phân công nhóm

| Thành viên | Phần việc chính |
|------------|-----------------|
| Nguyễn Minh Quân | Minimax, Hybrid, báo cáo tổng hợp |
| Phạm Thị Lan Anh | DQN model, trainer, train pipeline |
| Trần Đức Huy | Web UI, FastAPI server, session |
| Lê Hoàng Phúc | Pygame UI, pytest, demo |

---

<div align="center">

**— HẾT —**

</div>
