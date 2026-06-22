/**
 * Dashboard truy vết học DQN — buffer, lịch sử, so sánh trước/sau.
 */

const $ = (sel) => document.querySelector(sel);
const THEME_KEY = "caro-theme";
let compareChart = null;

function initTheme() {
  const btn = $("#theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    localStorage.setItem(THEME_KEY, next);
    if (window.lastCompare) renderCompareChart(window.lastCompare);
  });
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Loi API");
  }
  return res.json();
}

function formatMove(move) {
  if (!move) return "—";
  return `(${move[0]}, ${move[1]})`;
}

function formatTime(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("vi-VN");
  } catch {
    return iso;
  }
}

function renderBoardMini(board, size, highlightMove = null) {
  let html = `<div class="mini-board" style="--bs:${size}">`;
  for (let r = 0; r < size; r += 1) {
    for (let c = 0; c < size; c += 1) {
      const v = board[r][c];
      const cls = v === 1 ? "x" : v === 2 ? "o" : "";
      let extra = "";
      if (highlightMove && highlightMove[0] === r && highlightMove[1] === c) {
        extra = " highlight-dqn";
      }
      html += `<span class="mini-cell ${cls}${extra}"></span>`;
    }
  }
  html += "</div>";
  return html;
}

function rewardClass(reward) {
  if (reward > 0.01) return "pos";
  if (reward < -0.01) return "neg";
  return "neu";
}

function showError(msg) {
  const panel = $("#error-panel");
  panel.textContent = msg;
  panel.classList.remove("hidden");
}

function hideError() {
  $("#error-panel").classList.add("hidden");
}

function showLoading(show) {
  $("#loading-panel").classList.toggle("hidden", !show);
  $("#btn-compare").disabled = show;
}

function renderStatus(status) {
  const el = $("#status-section");
  const bufPct = status.buffer_capacity
    ? Math.round((status.buffer_size / status.buffer_capacity) * 100)
    : 0;
  const modelDate = status.model?.modified_at
    ? formatTime(status.model.modified_at)
    : "Chưa có file";
  const backupDate = status.model_backup?.modified_at
    ? formatTime(status.model_backup.modified_at)
    : "Chưa có (cần học online 1 lần)";

  el.innerHTML = `
    <article class="stat-card ${status.can_train_now ? "ok" : "warn"}">
      <div class="stat-label">Replay buffer</div>
      <div class="stat-value">${status.buffer_size}</div>
      <div class="stat-sub">/ ${status.buffer_capacity} mau (${bufPct}%) · can >= ${status.min_samples_to_train} de train</div>
    </article>
    <article class="stat-card">
      <div class="stat-label">Hoc online</div>
      <div class="stat-value">${status.online_learn_enabled ? "Bat" : "Tat"}</div>
      <div class="stat-sub">${status.gradient_steps_per_game} buoc gradient / van</div>
    </article>
    <article class="stat-card">
      <div class="stat-label">Model hien tai</div>
      <div class="stat-value" style="font-size:1rem">${status.model ? status.model.path : "—"}</div>
      <div class="stat-sub">${modelDate}</div>
    </article>
    <article class="stat-card ${status.has_backup ? "ok" : "warn"}">
      <div class="stat-label">Backup (truoc hoc)</div>
      <div class="stat-value" style="font-size:1rem">${status.has_backup ? "Co" : "Chua"}</div>
      <div class="stat-sub">${backupDate}</div>
    </article>
    <article class="stat-card">
      <div class="stat-label">Su kien hoc</div>
      <div class="stat-value">${status.learn_log_count}</div>
      <div class="stat-sub">trong learn_log_${status.board_size}.jsonl</div>
    </article>
  `;
}

function renderBufferCard(sample, boardSize) {
  const rc = rewardClass(sample.reward);
  const doneTag = sample.done ? " · ket thuc van" : "";
  return `
    <article class="buffer-card">
      <div class="buffer-card-head">
        <span>#${sample.index} · ${sample.perspective || "X"}</span>
        <span class="reward-tag ${rc}">r = ${sample.reward}${doneTag}</span>
      </div>
      ${renderBoardMini(sample.board, boardSize, sample.move)}
      <p class="move-line">Nuoc AI: ${formatMove(sample.move)} · nguon: ${sample.source || "online_pva"}</p>
    </article>
  `;
}

function renderHistory(events) {
  const el = $("#history-list");
  if (!events.length) {
    el.innerHTML = `<p class="empty-hint">Chua co van hoc nao. Choi PvA voi DQN/Hybrid va thang hoac thua de tao du lieu.</p>`;
    return;
  }

  el.innerHTML = events
    .map((ev) => {
      const pillClass =
        ev.buffered_only ? "buffered" : ev.outcome === "ai_loss" ? "loss" : "win";
      const pillText = ev.buffered_only
        ? "Chi ghi nho"
        : ev.outcome === "ai_loss"
          ? "AI thua — hoc tu sai lam"
          : "AI thang — cung co";
      const saved = ev.model_saved ? "Da luu model" : "Chua cap nhat file .pth";
      return `
        <article class="history-item">
          <div class="history-item-head">
            <span class="history-time">${formatTime(ev.timestamp)}</span>
            <span class="outcome-pill ${pillClass}">${pillText}</span>
          </div>
          <p class="history-meta">
            ${ev.ai_moves} nuoc AI · ${ev.gradient_steps} buoc gradient · loss TB ${ev.avg_loss ?? "—"}
            · buffer sau: ${ev.buffer_size_after} · ${saved}
          </p>
          ${ev.transitions?.length ? `
            <details style="margin-top:8px">
              <summary style="cursor:pointer;color:var(--accent-cool)">Xem ${ev.transitions.length} transition cua van nay</summary>
              <div class="buffer-grid" style="margin-top:10px">
                ${ev.transitions.map((t) => renderBufferCard(t, ev.board_size || 15)).join("")}
              </div>
            </details>
          ` : ""}
        </article>
      `;
    })
    .join("");
}

function renderBuffer(samples, boardSize) {
  const el = $("#buffer-list");
  if (!samples.length) {
    el.innerHTML = `<p class="empty-hint">Buffer trong — du lieu se xuat hien sau van PvA dau tien.</p>`;
    return;
  }
  el.innerHTML = samples.map((s) => renderBufferCard(s, boardSize)).join("");
}

function getChartColors() {
  const style = getComputedStyle(document.documentElement);
  return {
    text: style.getPropertyValue("--text").trim() || "#eef1f6",
    grid: style.getPropertyValue("--glass-border").trim() || "rgba(255,255,255,0.08)",
  };
}

function renderCompareChart(data) {
  if (compareChart) {
    compareChart.destroy();
    compareChart = null;
  }
  if (!data.has_backup || !data.summary) return;

  const c = getChartColors();
  const s = data.summary;
  compareChart = new Chart($("#chart-learn-compare"), {
    type: "bar",
    data: {
      labels: ["Truoc hoc (backup)", "Sau hoc (hien tai)"],
      datasets: [{
        label: "Dung TH chien thuat",
        data: [s.before_tactical_correct, s.after_tactical_correct],
        backgroundColor: ["#5b9dffaa", "#4ade80aa"],
        borderColor: ["#5b9dff", "#4ade80"],
        borderWidth: 1,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: s.tactical_total || 9,
          ticks: { stepSize: 1, color: c.text },
          grid: { color: c.grid },
        },
        x: { ticks: { color: c.text }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function verdictLabel(v) {
  const map = {
    improved: ["Cai thien", "verdict-improved"],
    regressed: ["Te hon", "verdict-regressed"],
    changed: ["Doi nuoc", "verdict-changed"],
    unchanged: ["Giu nguyen", "verdict-unchanged"],
  };
  const [text, cls] = map[v] || ["—", ""];
  return `<span class="${cls}">${text}</span>`;
}

function renderCompare(data) {
  window.lastCompare = data;
  const section = $("#compare-section");
  section.classList.remove("hidden");

  if (!data.has_current) {
    $("#compare-headline").textContent = data.message || "Chua co model.";
    return;
  }

  $("#compare-headline").textContent = data.headline || "—";

  const s = data.summary;
  const deltaCls = s.delta_tactical_pct > 0 ? "pos" : s.delta_tactical_pct < 0 ? "neg" : "";
  const betterText =
    s.is_better === true ? "Tot hon" : s.is_better === false ? "Chua tot hon" : "—";

  $("#compare-scores").innerHTML = `
    <div class="score-compare-card glass">
      <div class="stat-label">Truoc (backup)</div>
      <div class="before" style="font-size:1.8rem;font-weight:700">${s.before_tactical_pct}%</div>
      <div class="stat-sub">${s.before_tactical_correct}/${s.tactical_total} TH dung</div>
    </div>
    <div class="score-compare-card glass">
      <div class="stat-label">Sau (hien tai)</div>
      <div class="after" style="font-size:1.8rem;font-weight:700">${s.after_tactical_pct}%</div>
      <div class="stat-sub">${s.after_tactical_correct}/${s.tactical_total} TH dung</div>
    </div>
    <div class="score-compare-card glass">
      <div class="stat-label">Chenh lech</div>
      <div class="delta ${deltaCls}" style="font-size:1.8rem;font-weight:700">${s.delta_tactical_pct > 0 ? "+" : ""}${s.delta_tactical_pct}%</div>
      <div class="stat-sub">${betterText}</div>
    </div>
    <div class="score-compare-card glass">
      <div class="stat-label">Theo TH</div>
      <div style="font-size:1rem;font-weight:600">+${s.scenarios_improved} / -${s.scenarios_regressed}</div>
      <div class="stat-sub">cai thien / te hon</div>
    </div>
  `;

  renderCompareChart(data);

  $("#scenario-compare-list").innerHTML = (data.scenarios || [])
    .map((sc) => {
      const b = sc.before;
      const a = sc.after;
      return `
        <article class="scenario-card">
          <header class="scenario-head">
            <div>
              <span class="scenario-id">${sc.id.toUpperCase()}</span>
              <h4>${sc.name}</h4>
              <p>${sc.category} · ${verdictLabel(sc.verdict)}</p>
            </div>
          </header>
          <div class="scenario-body">
            ${renderBoardMini(sc.board, sc.board_size, a?.move)}
            <div class="scenario-table-wrap">
              <div class="scenario-compare-row">
                <div class="col-before">
                  <strong>Truoc</strong><br/>
                  Nuoc: ${formatMove(b?.move)}<br/>
                  dH: ${b?.heuristic_delta ?? "—"}<br/>
                  Chien thuat: ${b?.is_expected === true ? "dung" : b?.is_expected === false ? "sai" : "—"}
                </div>
                <div class="col-after">
                  <strong>Sau</strong><br/>
                  Nuoc: ${formatMove(a?.move)}<br/>
                  dH: ${a?.heuristic_delta ?? "—"}<br/>
                  Chien thuat: ${a?.is_expected === true ? "dung" : a?.is_expected === false ? "sai" : "—"}
                </div>
              </div>
            </div>
          </div>
        </article>
      `;
    })
    .join("");

  const bufEl = $("#buffer-compare-list");
  const rows = data.buffer_transitions || [];
  if (!rows.length) {
    bufEl.innerHTML = `<p class="empty-hint">Chua co transition trong buffer de so sanh Q-value.</p>`;
  } else if (!data.has_backup) {
    bufEl.innerHTML = rows
      .map(
        (r) => `
      <div class="buffer-compare-item">
        ${renderBoardMini(r.board, data.board_size, r.learned_move)}
        <div>
          <p><strong>Nuoc AI da di:</strong> ${formatMove(r.learned_move)} · r=${r.reward}</p>
          <p>Q(s,a) hien tai: <strong>${r.after_q_at_move}</strong> · greedy: ${formatMove(r.after_greedy_move)}</p>
        </div>
      </div>`
      )
      .join("");
  } else {
    bufEl.innerHTML = rows
      .map((r) => {
        const qCls = (r.q_delta ?? 0) < 0 ? "neg" : (r.q_delta ?? 0) > 0 ? "pos" : "";
        const repeatBefore = r.before_would_repeat ? " (van chon lai)" : "";
        const repeatAfter = r.after_would_repeat ? " (van chon lai)" : "";
        return `
      <div class="buffer-compare-item">
        ${renderBoardMini(r.board, data.board_size, r.learned_move)}
        <div>
          <p><strong>Nuoc da hoc:</strong> ${formatMove(r.learned_move)} · reward=${r.reward}${r.done ? " · cuoi van" : ""}</p>
          <p>Q truoc: ${r.before_q_at_move} → sau: ${r.after_q_at_move}
            <span class="q-delta ${qCls}">(d ${r.q_delta > 0 ? "+" : ""}${r.q_delta})</span></p>
          <p>Greedy truoc: ${formatMove(r.before_greedy_move)}${repeatBefore}</p>
          <p>Greedy sau: ${formatMove(r.after_greedy_move)}${repeatAfter}
            ${r.move_changed ? " · <strong>doi nuoc uu tien</strong>" : ""}</p>
        </div>
      </div>`;
      })
      .join("");
  }
}

async function loadDashboard(boardSize) {
  hideError();
  const [status, history, buffer] = await Promise.all([
    api(`/api/learn/status?board_size=${boardSize}`),
    api(`/api/learn/history?board_size=${boardSize}&limit=20`),
    api(`/api/learn/buffer?board_size=${boardSize}&limit=24`),
  ]);
  renderStatus(status);
  renderHistory(history.events);
  renderBuffer(buffer.samples, boardSize);
}

async function runCompare(formData) {
  hideError();
  showLoading(true);
  try {
    const body = {
      board_size: Number(formData.get("board_size")),
      difficulty: formData.get("difficulty"),
      double_end_block_rule: true,
      ai_aggressive: true,
    };
    const data = await api("/api/learn/compare", {
      method: "POST",
      body: JSON.stringify(body),
    });
    renderCompare(data);
    $("#compare-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    showLoading(false);
  }
}

async function loadBoardSizes() {
  const opts = await api("/api/options");
  const sel = $("#board_size");
  sel.innerHTML = opts.board_sizes
    .map((n) => `<option value="${n}">${n}×${n}</option>`)
    .join("");
  sel.value = String(opts.board_sizes.includes(15) ? 15 : opts.board_sizes.at(-1));
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadBoardSizes().then(() => {
    const bs = Number($("#board_size").value);
    loadDashboard(bs).catch((err) => showError(err.message));
  });

  $("#learn-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runCompare(new FormData(e.target));
  });

  $("#btn-refresh").addEventListener("click", () => {
    const bs = Number($("#board_size").value);
    loadDashboard(bs).catch((err) => showError(err.message));
  });

  $("#board_size").addEventListener("change", () => {
    const bs = Number($("#board_size").value);
    loadDashboard(bs).catch((err) => showError(err.message));
    $("#compare-section").classList.add("hidden");
  });
});
