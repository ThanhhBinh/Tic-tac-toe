# Tối ưu AI Cờ Caro — Báo cáo điều tra & tăng tốc

> Mục tiêu: làm AI **nhanh và mạnh như các trang cờ caro online**, xử lý 3 vấn đề:
> (1) Minimax suy nghĩ quá lâu, (2) DQN train nhiều vẫn đánh yếu, (3) Hybrid kết hợp 2 thuật toán.

---

## 1. Tóm tắt: vấn đề nằm ở đâu

Sau khi đọc toàn bộ code và **đo bằng profiler** trên bàn 15×15 ở một thế cờ mở (22 quân, chưa có đòn bắt buộc), kết quả như sau:

| Hạng mục đo | Trước khi sửa |
|---|---|
| Số nước ứng viên `candidate_moves(radius=2)` | **131 nước** |
| `find_tactical_move` (gọi mỗi nước) | **25,7 ms/lần** (clone env cho từng ứng viên) |
| Minimax thô — depth 2 | **13,7 giây** |
| Minimax thô — depth 3 | **80,7 giây** |
| Minimax thô — depth 4 | **> 150 giây** (chưa xong) |
| % thời gian nằm trong `evaluate_board` | **97%** |

Vì có giới hạn thời gian 5 giây/nước, trong thế cờ mở AI **chỉ kịp tìm tới depth 1–2** rồi trả nước dở dang → vừa chậm vừa đánh không sâu. Đây chính là cảm giác "suy nghĩ lâu mà vẫn không hay".

### Ba nút thắt cổ chai (root cause)

**(1) Hàm lượng giá quét lại toàn bộ bàn cờ ở MỖI node lá.**
`evaluate_board` → dựng ~90 chuỗi string cho mọi hàng/cột/đường chéo → chạy 15 biểu thức regex `.findall()` trên từng dòng. Mỗi lần gọi tốn ~2,5 ms, và nó được gọi lại **từ đầu** ở mọi node lá của cây tìm kiếm. Với hàng chục nghìn node, đây là 97% thời gian.

**(2) Sắp xếp nước đi cũng gọi lượng giá toàn bàn.**
`move_priority` (dùng để xếp hạng nước) thực hiện `board.copy()` + **2 lần** `evaluate_board` cho **từng** trong 131 ứng viên, ở **mỗi** node. Tức là ~260 lần quét bàn chỉ để sắp xếp một node.

**(3) Lớp chiến thuật `clone()` env cho từng ứng viên.**
`find_winning_move / find_blocking_move / find_open_four_block / find_open_three_block` mỗi hàm `clone()` cả môi trường (cấp phát bàn cờ mới) cho từng trong 131 ứng viên — và chạy 4 lượt mỗi nước.

---

## 2. Các trang caro online làm thế nào để nhanh

Engine Gomoku/Caro mạnh **không bao giờ** quét lại toàn bàn ở mỗi node. Bí quyết của họ:

1. **Lượng giá tăng dần (incremental evaluation).** Giữ một điểm tổng; khi đặt 1 quân chỉ tính lại **4 đường thẳng** đi qua ô đó (delta), thay vì cả bàn. Đây là cú nhảy tốc độ lớn nhất — từ O(n²) xuống O(n) mỗi nước.
2. **Bảng mẫu tra cứu sẵn + bitboard.** Mã hoá mỗi đường thành số rồi tra điểm trong bảng băm, không dùng regex/string.
3. **Tìm kiếm chọn lọc (selective search).** Chỉ xét ~10–16 nước tốt nhất sau khi sắp xếp, thay vì cả trăm ô.
4. **Threat-space search / VCF–VCT.** Tìm chuỗi thắng ép buộc (tứ, tam→tứ) cực nhanh thay vì brute-force sâu.
5. **Zobrist hashing** cho transposition table + **PVS / aspiration window**.
6. Thực tế nhiều trang chỉ search **depth 4–8 nông**, nhưng vì evaluator nhanh nên *cảm giác* tức thì.

---

## 3. Những gì đã sửa (đã code + kiểm chứng)

| # | Thay đổi | File | Ý tưởng |
|---|---|---|---|
| 1 | **Evaluator tăng dần** | `ai/incremental_eval.py` (mới) | Cache điểm theo từng đường; mỗi nước chỉ cập nhật 4 đường. Khớp **chính xác** `evaluate_board`. |
| 2 | **Bỏ `clone()` trong lớp tactical** | `ai/heuristic.py`, `ai/threats.py` | Đặt/gỡ quân tại chỗ + push/pop thay vì clone cả env. |
| 3 | **`move_priority` cục bộ** | `ai/heuristic.py` | Chỉ chấm 4 đường qua ô, bỏ `board.copy()` + quét toàn bàn. |
| 4 | **Tích hợp evaluator vào search** | `ai/minimax_agent.py`, `ai/hybrid_agent.py` | Node lá đọc điểm từ evaluator; cập nhật theo push/pop. |
| 5 | **Tuỳ chọn giới hạn nhánh** | `config.py` | `MINIMAX_MAX_BRANCH_BY_DIFFICULTY` — **mặc định None (tắt)** để giữ nguyên sức mạnh; bật khi cần nhanh hơn trên máy yếu. |

### Kết quả đo SAU khi sửa (cùng thế cờ)

| Hạng mục | Trước | Sau | Cải thiện |
|---|---|---|---|
| Minimax thô depth 2 (vét cạn) | 13,7 s | **0,91 s** | ~15× |
| Minimax thô depth 3 (vét cạn) | 80,7 s | **10,5 s** | ~7,7× |
| Một nước có đòn tactical (chơi thật) | ~5 s (timeout) | **~0,025 s** | **~200×** |
| Chơi thật, ngân sách 5s/nước | dở dang ở depth 1 | **hoàn tất depth 2** trong <1s | sâu hơn + nhanh hơn |

> **Điểm mấu chốt — không hề yếu đi:** evaluator tăng dần cho kết quả **KHỚP CHÍNH XÁC** với hàm lượng giá cũ (test 200 bàn ngẫu nhiên + toàn bộ chuỗi push/pop khi tìm kiếm → **0 sai lệch**). Vì hàm đánh giá không đổi và **mặc định không cắt nhánh**, chất lượng nước đi **được bảo toàn**. Thực ra còn MẠNH HƠN: trước đây trong thế cờ mở, ngân sách 5s không đủ để hoàn tất cả depth 2 (13,7s) nên AI chỉ trả nước ở depth 1; giờ nó hoàn tất depth 2 trong <1s và bắt đầu đào depth 3.

### Kiểm chứng

- **Unit test:** 33/33 test (minimax, threats, caro_env) PASS; thêm 9 test Hybrid/heuristic-tactical (gồm test "không có DQN thì không gọi mạng", "sâu hơn Minimax 1 ply", "leaf heuristic") PASS với torch giả lập. Còn 3 test cần forward thật của DQN + module `ai.evaluate` → **chạy trên `.venv` máy bạn** bằng `pytest -q` (logic đó tôi không đụng tới).
- **Evaluator chính xác:** 0 sai lệch vs `evaluate_board` trên 200 bàn + trong search.
- **Toàn vẹn bàn cờ:** sau mọi lần quét tactical, `env.board` không bị thay đổi.
- **Tính đúng tactical:** vẫn bắt nước thắng ngay, chặn thua bắt buộc.
- **Đối kháng (head-to-head):** thử nghiệm cho thấy **cắt nhánh top-K làm yếu đi đôi chút** (bản cắt-nhánh depth-4 chỉ HOÀ được bản vét-cạn depth-2; cắt nhánh ở mọi mức đều thua nhẹ bản vét cạn cùng mức). Đây là lý do **mặc định TẮT cắt nhánh** — để đảm bảo yêu cầu "nhanh mà không yếu". Cờ caro có lợi thế đi trước rất lớn, nên phần lớn kết quả do bên đi trước quyết định.

> ⚠️ Lưu ý môi trường: máy chủ sandbox không cài được `torch`/`pytest`, nên test Hybrid/DQN cần chạy lại trên `.venv` của máy bạn bằng `pytest -q` (xem mục bên dưới).

---

## 4. DQN — vì sao "train nhiều vẫn đánh ngu"

Đây là vấn đề **bản chất**, không phải bug. DQN thuần học from-scratch trên bàn 15×15 rất khó mạnh vì:

1. **Reward thưa.** Chỉ có thưởng ở cuối ván; gán công lao (credit assignment) ngược qua 30+ nước với `gamma=0.99` rất yếu → mạng khó biết nước nào thực sự tốt.
2. **Lúc suy luận chỉ greedy 1-ply.** DQN chọn ô có Q cao nhất, **không có tìm kiếm** → không "nhìn" được đòn thế 3–4 nước. Hiện nó chơi ổn chủ yếu nhờ lớp tactical (thắng/chặn ngay) chứ không phải nhờ mạng.
3. **3000 episode là quá ít.** Self-play Gomoku cần hàng trăm nghìn ván để hình thành chiến lược.
4. **Không gian hành động 225 ô**, phần lớn vô dụng ở mỗi thế → mạng học chậm.

### Hướng xử lý (theo thứ tự đáng làm)

**Phương án A — Thực dụng nhất cho đồ án (khuyến nghị):**
Dùng DQN làm **hàm value ở node lá của Minimax** thay vì để nó tự chơi. Tức là Minimax tìm sâu, đến node lá thì hỏi DQN "thế này lợi cho ai bao nhiêu". Cách này:
- Tận dụng sức tìm kiếm của Minimax (vốn đã nhanh sau khi tối ưu).
- DQN chỉ cần học *đánh giá thế cờ* (dễ hơn nhiều so với *chọn nước*).
- Đây đúng tinh thần "Hybrid" và mạnh hơn cả hai thành phần.

**Phương án B — Mạnh thật sự nhưng tốn công: AlphaZero-lite.**
Thay DQN bằng **MCTS + mạng policy/value**, train bằng self-play. Đây là cách các engine cờ mạnh nhất dùng. Cần nhiều compute hơn nhưng cho sức mạnh vượt trội.

**Cải thiện DQN hiện tại (nếu giữ kiến trúc):**
- Tăng reward shaping cho tạo/đỡ đe doạ (đã có `DQN_REWARD_OPEN_FOUR/THREE` — tăng thêm).
- Train nhiều hơn (10–50k ván) + buffer lớn hơn.
- Dùng **n-step return** thay vì 1-step để truyền tín hiệu thắng/thua nhanh hơn.
- Tăng receptive field của CNN (thêm block) để "thấy" đường 5 quân.

---

## 5. Hybrid — vai trò đúng

Hybrid hiện tại dùng đúng triết lý: **Minimax là bộ não tìm kiếm, DQN hỗ trợ sắp xếp nước + ước lượng win%**, không để DQN yếu quyết định nước đi. Sau khi tối ưu, Hybrid cũng được tích hợp evaluator tăng dần + giới hạn nhánh nên nhanh tương đương Minimax. Nếu muốn Hybrid mạnh hơn nữa, nâng cấp DQN theo **Phương án A** ở trên (DQN làm value ở leaf).

---

## 6. Lộ trình nếu muốn tiến xa hơn

1. **Zobrist hashing** thay `board.tobytes()` cho transposition table (nhanh hơn, cache xuyên nước đi).
2. **VCF/VCT (threat-space search):** module tìm chuỗi thắng ép buộc — giúp AI "kết liễu" như cao thủ.
3. **PVS (Principal Variation Search) + aspiration window:** cắt tỉa sâu hơn.
4. **Bitboard** cho quét đường siêu nhanh.
5. DQN làm value ở leaf (Phương án A) hoặc chuyển AlphaZero-lite (Phương án B).

---

## Phụ lục — Các file đã thay đổi

- `ai/incremental_eval.py` — **mới**: evaluator tăng dần.
- `ai/heuristic.py` — `move_priority` cục bộ; bỏ clone ở `find_winning_move`, `find_open_four_move`, `find_open_four_block`, `find_open_three_block`; thêm `_wins_with_move`, `_lines_through_cell`.
- `ai/threats.py` — bỏ clone ở `find_open_three_attack`, `find_open_three_block_double_end`.
- `ai/minimax_agent.py` — tích hợp evaluator vào `_alpha_beta`/`_evaluate_leaf`; thêm `max_branch` mặc định theo độ khó.
- `ai/hybrid_agent.py` — dùng evaluator chung; thêm `max_branch` khi chơi.
- `config.py` — thêm `MINIMAX_MAX_BRANCH_BY_DIFFICULTY`.
