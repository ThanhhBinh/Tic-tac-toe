/**
 * AI Cờ Caro — Web client (Chrome)
 * Giao tiếp REST API với FastAPI backend.
 */

const $ = (sel) => document.querySelector(sel);

const setupScreen = $("#setup-screen");
const gameScreen = $("#game-screen");
const setupForm = $("#setup-form");
const boardEl = $("#board");
const endModal = $("#end-modal");
const endBanner = $("#end-banner");

const THEME_KEY = "caro-theme";

let sessionId = null;
let state = null;
let modalDismissed = false;
let lastRenderedMoveCount = 0;
let avaTimer = null;
let avaRunning = false;
let aiThinkingActive = false;
let aiThinkTimer = null;
let aiThinkStart = 0;
let learnPollTimer = null;

const AVA_MOVE_DELAY_MS = 320;

/** Chờ kết quả học online (chạy nền sau khi ván kết thúc). */
function scheduleOnlineLearnPoll() {
  if (learnPollTimer != null) {
    clearInterval(learnPollTimer);
    learnPollTimer = null;
  }
  if (!sessionId || !state?.done || state.settings?.mode !== "Player vs AI") {
    return;
  }
  if (state.online_learn) {
    return;
  }

  let attempts = 0;
  learnPollTimer = setInterval(async () => {
    attempts += 1;
    if (attempts > 24 || !sessionId) {
      clearInterval(learnPollTimer);
      learnPollTimer = null;
      return;
    }
    try {
      const next = await api(`/api/games/${sessionId}`);
      if (next.online_learn) {
        updateUI(next);
        clearInterval(learnPollTimer);
        learnPollTimer = null;
      }
    } catch (err) {
      console.error(err);
    }
  }, 500);
}

/** Hiển thị / ẩn thanh suy nghĩ AI ở sidebar (không che bàn cờ). */
function setAiThinking(active, label = "AI suy nghĩ") {
  aiThinkingActive = active;
  const block = $("#ai-think-block");
  if (!block) return;

  if (active) {
    block.classList.remove("hidden");
    $("#ai-think-label").textContent = label;
    aiThinkStart = performance.now();
    if (aiThinkTimer != null) clearInterval(aiThinkTimer);
    aiThinkTimer = setInterval(() => {
      const sec = (performance.now() - aiThinkStart) / 1000;
      $("#ai-think-time").textContent = `${sec.toFixed(1)}s`;
    }, 100);
  } else {
    block.classList.add("hidden");
    if (aiThinkTimer != null) {
      clearInterval(aiThinkTimer);
      aiThinkTimer = null;
    }
    $("#ai-think-time").textContent = "0.0s";
  }

  if (state) {
    renderBoard(state);
    updateHUD(state);
  }
}

/** Dừng vòng lặp AI vs AI. */
function stopAvaLoop() {
  if (avaTimer != null) {
    clearTimeout(avaTimer);
    avaTimer = null;
  }
  avaRunning = false;
  setAiThinking(false);
}

/** Tiến một lượt AI trong chế độ AvA. */
async function runAvaStep() {
  avaTimer = null;
  if (!sessionId || !state || state.done || !state.is_ava) {
    stopAvaLoop();
    return;
  }
  if (state.is_human_turn) {
    stopAvaLoop();
    return;
  }

  avaRunning = true;
  setAiThinking(true, "AI vs AI");
  try {
    const next = await api(`/api/games/${sessionId}/ava-step`, { method: "POST" });
    updateUI(next, { fromAva: true });
    if (!next.done && next.is_ava && !next.is_human_turn) {
      avaTimer = setTimeout(runAvaStep, AVA_MOVE_DELAY_MS);
    } else {
      avaRunning = false;
      setAiThinking(false);
    }
  } catch (err) {
    console.error(err);
    avaRunning = false;
    setAiThinking(false);
    alert(err.message);
  }
}

/** Bắt đầu tự chạy nếu đang ở chế độ AI vs AI. */
function scheduleAvaLoop() {
  stopAvaLoop();
  if (
    state?.is_ava &&
    !state.done &&
    !state.is_human_turn &&
    gameScreen.classList.contains("active")
  ) {
    avaTimer = setTimeout(runAvaStep, AVA_MOVE_DELAY_MS);
  }
}

/** Gọi API JSON. */
async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return data;
}

/** Ẩn/hiện tuỳ chọn theo chế độ chơi. */
function updateSetupFields() {
  const modeEl = $("#mode");
  if (!modeEl) return;
  const isPva = modeEl.value === "Player vs AI";
  $("#ai-first-field")?.classList.toggle("hidden", !isPva);
}

/** Nạp tuỳ chọn form từ server. */
async function loadOptions() {
  const opts = await api("/api/options");
  fillSelect("#mode", opts.modes);
  fillSelect("#ai_type", opts.ai_types);
  fillSelect("#difficulty", opts.difficulties);
  fillSelect("#board_size", opts.board_sizes.map(String));
  $("#board_size").value = String(opts.board_sizes.includes(15) ? 15 : opts.board_sizes.at(-1));
  $("#mode").value = "Player vs AI";
  updateSetupFields();
}

function fillSelect(selector, values) {
  const el = $(selector);
  el.innerHTML = values
    .map((v) => `<option value="${v}">${v}</option>`)
    .join("");
}

/** Áp dụng theme sáng/tối (lưu localStorage). */
function applyTheme(theme) {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem(THEME_KEY, next);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  applyTheme(current === "dark" ? "light" : "dark");
}

function initTheme() {
  applyTheme(localStorage.getItem(THEME_KEY) || "dark");
}

/** Chuyển màn hình. */
function showScreen(name) {
  setupScreen.classList.toggle("active", name === "setup");
  gameScreen.classList.toggle("active", name === "game");
}

/** Tạo ván mới. */
async function startGame(formData) {
  const aiFirst = formData.get("ai_first") === "on";
  showScreen("game");
  if (aiFirst) {
    setAiThinking(true, "AI mở đầu");
  }

  try {
    state = await api("/api/games", {
      method: "POST",
      body: JSON.stringify({
        mode: formData.get("mode"),
        ai_type: formData.get("ai_type"),
        difficulty: formData.get("difficulty"),
        board_size: parseInt(formData.get("board_size"), 10),
        double_end_block_rule: formData.get("double_end_block_rule") === "on",
        threat_warnings: formData.get("threat_warnings") === "on",
        ai_aggressive: formData.get("ai_aggressive") === "on",
        ai_first: aiFirst,
      }),
    });
    sessionId = state.session_id;
    modalDismissed = false;
    lastRenderedMoveCount = 0;
    updateUI(state);
  } finally {
    setAiThinking(false);
  }
}

/** Cập nhật toàn bộ UI từ state server. */
function updateUI(s, options = {}) {
  state = s;
  renderBoard(s);
  updateHUD(s);
  updateEndUI(s);
  if (!options.fromAva) {
    scheduleAvaLoop();
  }
}

/** Vẽ bàn cờ. */
function renderBoard(s) {
  const size = s.board_size;
  const animateNew = s.move_count > lastRenderedMoveCount;
  lastRenderedMoveCount = s.move_count;

  boardEl.style.gridTemplateColumns = `repeat(${size}, 1fr)`;
  boardEl.innerHTML = "";

  const winSet = new Set(
    (s.winning_line || []).map(([r, c]) => `${r},${c}`)
  );
  const last = s.last_move;

  const winThreats = new Set(
    (s.threats?.win_moves || []).map(([r, c]) => `${r},${c}`)
  );
  const blockThreats = new Set(
    (s.threats?.block_moves || []).map(([r, c]) => `${r},${c}`)
  );
  const doubleThreats = new Set(
    (s.threats?.double_end_blocks || []).map(([r, c]) => `${r},${c}`)
  );
  const threatStones = new Set(
    (s.threats?.threat_stones || []).map(([r, c]) => `${r},${c}`)
  );

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      cell.dataset.row = r;
      cell.dataset.col = c;
      cell.setAttribute("aria-label", `Ô ${r}, ${c}`);

      const val = s.board[r][c];
      if (val === 0 && s.is_human_turn && !s.done && !aiThinkingActive) {
        cell.classList.add("playable");
        cell.addEventListener("click", () => onCellClick(r, c));
      }

      if (last && last[0] === r && last[1] === c && !s.done) {
        cell.classList.add("last-move");
      }
      if (winSet.has(`${r},${c}`)) {
        cell.classList.add("win-cell");
      }
      if (winThreats.has(`${r},${c}`)) {
        cell.classList.add("threat-win");
      } else if (blockThreats.has(`${r},${c}`)) {
        cell.classList.add("threat-block");
      } else if (doubleThreats.has(`${r},${c}`)) {
        cell.classList.add("threat-double");
      }

      if (val === 1 || val === 2) {
        const stone = document.createElement("span");
        stone.className = `stone ${val === 1 ? "stone-x" : "stone-o"}`;
        if (threatStones.has(`${r},${c}`)) {
          stone.classList.add("stone-threat");
        }
        if (animateNew && last && last[0] === r && last[1] === c) {
          stone.style.animation = "placeStone 0.28s cubic-bezier(0.34, 1.4, 0.64, 1) forwards";
        } else {
          stone.style.transform = "scale(1)";
        }
        cell.appendChild(stone);
      }

      boardEl.appendChild(cell);
    }
  }
}

/** Cập nhật sidebar / HUD. */
function updateHUD(s) {
  $("#game-mode-label").textContent = s.settings.mode;
  $("#game-status").textContent = s.status_text;
  $("#move-count").textContent = s.move_count;

  const isPva = s.settings.mode === "Player vs AI";
  const isAva = s.is_ava || s.settings.mode === "AI vs AI";
  $("#undo-row").classList.toggle("hidden", !isPva);
  $("#btn-undo").disabled = !s.can_undo || aiThinkingActive;
  $("#btn-redo").disabled = !s.can_redo || aiThinkingActive;

  if (isAva) {
    $("#label-x").textContent = "AI";
    $("#label-o").textContent = "AI";
  } else {
    const human = s.human_player || "X";
    $("#label-x").textContent = human === "X" ? "Bạn" : "AI";
    $("#label-o").textContent = human === "O" ? "Bạn" : "AI";
  }

  $("#card-x").classList.toggle("active", !s.done && s.current_player === "X");
  $("#card-o").classList.toggle("active", !s.done && s.current_player === "O");

  $("#ai-type-label").textContent = s.settings.ai_type;
  $("#difficulty-label").textContent = s.settings.difficulty;

  if (s.win_probability != null) {
    const pct = Math.round(s.win_probability * 100);
    const isHeuristic = s.win_probability_source === "heuristic";
    $("#win-prob").textContent = isHeuristic ? `~${pct}%` : `${pct}%`;
    $("#win-prob-bar").style.width = `${pct}%`;
    $("#win-prob-bar-text").textContent = isHeuristic
      ? `~${pct}% (heuristic)`
      : `${pct}%`;
  } else {
    $("#win-prob").textContent = "—";
    $("#win-prob-bar").style.width = "0%";
    $("#win-prob-bar-text").textContent = "N/A";
  }

  if (s.threats?.message) {
    const banner = $("#threat-banner");
    banner.textContent = s.threats.message;
    banner.classList.remove("hidden");
    banner.classList.toggle(
      "threat-banner-danger",
      s.threats.message.includes("Chặn 2 đầu") || (s.threats.block_moves?.length > 0)
    );
  } else {
    $("#threat-banner").classList.add("hidden");
    $("#threat-banner").classList.remove("threat-banner-danger");
  }

  if (s.done) {
    $("#keyboard-hint").textContent = "Ván đã kết thúc";
  } else if (aiThinkingActive) {
    $("#keyboard-hint").textContent = "AI đang tính — bàn cờ vẫn hiển thị bình thường";
  } else if (isAva && !s.is_human_turn) {
    $("#keyboard-hint").textContent = avaRunning
      ? "AI vs AI — đang tự đấu…"
      : "AI vs AI — sắp bắt đầu…";
  } else if (s.is_human_turn) {
    $("#keyboard-hint").textContent = "Click ô trống · U: Quay lại · T: Sáng/Tối";
  } else {
    $("#keyboard-hint").textContent = "Đang chờ AI…";
  }
}

/** Pop-up / banner kết thúc. */
function updateEndUI(s) {
  if (!s.done) {
    endModal.classList.add("hidden");
    endBanner.classList.add("hidden");
    return;
  }

  const title = s.is_draw ? "HÒA!" : `${s.winner} THẮNG!`;
  const sub = s.is_draw
    ? "Hai bên cân tài cân sức"
    : `Sau ${s.move_count} nước đi`;

  $("#end-title").textContent = title;
  $("#end-subtitle").textContent = sub;
  $("#end-icon").textContent = s.is_draw ? "🤝" : "🏆";
  $("#banner-text").textContent = title;

  const learnEl = $("#end-learn");
  if (learnEl) {
    const learn = s.online_learn;
    if (learn && learn.outcome === "ai_loss") {
      if (learn.model_saved) {
        learnEl.textContent =
          `🧠 AI đã học từ thất bại (${learn.ai_moves} nước, ${learn.gradient_steps} bước cập nhật)`;
      } else if (learn.buffered_only) {
        learnEl.textContent =
          `📝 AI đã ghi nhớ thất bại (${learn.ai_moves} nước) — cần thêm ván để cập nhật model`;
      } else {
        learnEl.textContent =
          `📝 AI đã ghi nhớ thất bại (${learn.ai_moves} nước)`;
      }
      learnEl.classList.remove("hidden");
    } else if (learn && learn.outcome === "ai_win") {
      if (learn.model_saved) {
        learnEl.textContent =
          `🧠 AI củng cố chiến thắng (${learn.gradient_steps} bước cập nhật)`;
      } else {
        learnEl.textContent = `📝 AI đã ghi nhớ chiến thắng`;
      }
      learnEl.classList.remove("hidden");
    } else {
      learnEl.textContent = "";
      learnEl.classList.add("hidden");
    }
  }

  if (modalDismissed) {
    endModal.classList.add("hidden");
    endBanner.classList.remove("hidden");
  } else {
    endModal.classList.remove("hidden");
    endBanner.classList.add("hidden");
  }

  scheduleOnlineLearnPoll();
}

function dismissModal() {
  modalDismissed = true;
  endModal.classList.add("hidden");
  endBanner.classList.remove("hidden");
}

function showModal() {
  modalDismissed = false;
  if (state?.done) {
    endModal.classList.remove("hidden");
    endBanner.classList.add("hidden");
  }
}

/** Đặt quân — hiện thanh suy nghĩ ở sidebar khi chờ AI. */
async function onCellClick(row, col) {
  if (!sessionId || !state?.is_human_turn || state.done || aiThinkingActive) return;
  setAiThinking(true, "AI phản hồi");
  try {
    const next = await api(`/api/games/${sessionId}/move`, {
      method: "POST",
      body: JSON.stringify({ row, col }),
    });
    updateUI(next);
  } catch (err) {
    console.error(err);
    alert(err.message);
  } finally {
    setAiThinking(false);
  }
}

async function undo() {
  if (!sessionId || aiThinkingActive) return;
  try {
    updateUI(await api(`/api/games/${sessionId}/undo`, { method: "POST" }));
    modalDismissed = false;
  } catch (err) {
    console.error(err);
  }
}

async function redo() {
  if (!sessionId || aiThinkingActive) return;
  try {
    updateUI(await api(`/api/games/${sessionId}/redo`, { method: "POST" }));
  } catch (err) {
    console.error(err);
  }
}

function backToMenu() {
  stopAvaLoop();
  if (sessionId) {
    api(`/api/games/${sessionId}`, { method: "DELETE" }).catch(() => {});
  }
  sessionId = null;
  state = null;
  showScreen("setup");
}

/** Sự kiện UI. */
setupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(setupForm);
  try {
    await startGame(fd);
  } catch (err) {
    setAiThinking(false);
    showScreen("setup");
    alert("Không tạo được ván: " + err.message);
  }
});

$("#mode")?.addEventListener("change", updateSetupFields);

$("#btn-back").addEventListener("click", backToMenu);
$("#btn-new").addEventListener("click", () => {
  if (setupForm) setupForm.requestSubmit();
});
$("#btn-undo").addEventListener("click", undo);
$("#btn-redo").addEventListener("click", redo);
$("#btn-view-board").addEventListener("click", dismissModal);
$("#modal-backdrop").addEventListener("click", dismissModal);
$("#btn-show-modal").addEventListener("click", showModal);
$("#btn-play-again").addEventListener("click", () => setupForm.requestSubmit());
$("#btn-to-menu").addEventListener("click", backToMenu);

$("#theme-toggle-setup")?.addEventListener("click", toggleTheme);
$("#theme-toggle-game")?.addEventListener("click", toggleTheme);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (state?.done && !modalDismissed) dismissModal();
    else if (gameScreen.classList.contains("active")) backToMenu();
  }
  if (e.key === " " && state?.done && !modalDismissed) {
    e.preventDefault();
    dismissModal();
  }
  if (gameScreen.classList.contains("active") && state?.settings?.mode === "Player vs AI") {
    if (e.key === "u" || (e.ctrlKey && e.key === "z" && !e.shiftKey)) {
      e.preventDefault();
      undo();
    }
    if (e.key === "y" || (e.ctrlKey && e.key === "z" && e.shiftKey)) {
      e.preventDefault();
      redo();
    }
  }
  if (e.key === "t" && !e.ctrlKey && !e.metaKey) {
    toggleTheme();
  }
});

initTheme();
loadOptions().catch(console.error);
