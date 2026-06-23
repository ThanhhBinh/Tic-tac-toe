/**
 * Trang so sánh Minimax / DQN / Hybrid — benchmark lưu SQLite (models/benchmark_cache.db).
 */

const $ = (sel) => document.querySelector(sel);

const AGENT_KEYS = ["minimax", "dqn", "hybrid"];
const AGENT_COLORS = {
  minimax: "#5b9dff",
  dqn: "#4ade80",
  hybrid: "#fbbf24",
};
const AGENT_LABELS = {
  minimax: "Minimax",
  dqn: "DQN",
  hybrid: "Hybrid",
};

const THEME_KEY = "caro-theme";
let charts = {};

function compareParams(formData) {
  return {
    scenario_set: formData.get("scenario_set") || "basic",
    difficulty: formData.get("difficulty") || "MEDIUM",
    board_size: Number(formData.get("board_size") || 15),
    double_end_block_rule: true,
    ai_aggressive: true,
  };
}

function cacheQueryString(params) {
  const q = new URLSearchParams({
    scenario_set: params.scenario_set,
    difficulty: params.difficulty,
    board_size: String(params.board_size),
    double_end_block_rule: "true",
    ai_aggressive: "true",
  });
  return q.toString();
}

function showCacheStatus(data) {
  const el = $("#cache-status");
  if (!el) return;
  if (data && data.from_cache) {
    const when = data.cached_at
      ? new Date(data.cached_at).toLocaleString("vi-VN")
      : "";
    el.textContent = when
      ? `Đã lưu DB — tải tức thì (${when})`
      : "Đã lưu DB — tải tức thì, không chạy lại";
    el.classList.remove("hidden");
  } else if (data && data.run_elapsed_ms != null) {
    el.textContent = `Vừa chạy xong (${(data.run_elapsed_ms / 1000).toFixed(1)}s) — đã ghi DB`;
    el.classList.remove("hidden");
  } else {
    el.classList.add("hidden");
    el.textContent = "";
  }
}

async function fetchCachedCompare(params) {
  const res = await fetch(`/api/compare/result?${cacheQueryString(params)}`);
  if (res.status === 404) return null;
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Lỗi tải cache");
  }
  return res.json();
}

function updateRunButton(hasCache) {
  const btn = $("#btn-run");
  if (!btn) return;
  btn.textContent = hasCache ? "Tải kết quả" : "Chạy benchmark";
}

function getChartColors() {
  const style = getComputedStyle(document.documentElement);
  return {
    text: style.getPropertyValue("--text").trim() || "#eef1f6",
    muted: style.getPropertyValue("--text-muted").trim() || "#8b95a8",
    grid: style.getPropertyValue("--glass-border").trim() || "rgba(255,255,255,0.08)",
  };
}

function initTheme() {
  const btn = $("#theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    localStorage.setItem(THEME_KEY, next);
    if (window.lastBenchmark) renderCharts(window.lastBenchmark);
  });
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Lỗi API");
  }
  return res.json();
}

function destroyCharts() {
  Object.values(charts).forEach((c) => c.destroy());
  charts = {};
}

function chartDefaults() {
  const c = getChartColors();
  return {
    color: c.text,
    borderColor: c.grid,
  };
}

function renderCharts(data) {
  destroyCharts();
  if (typeof Chart === "undefined") {
    document.querySelectorAll(".chart-card canvas").forEach((canvas) => {
      const card = canvas.closest(".chart-card");
      if (card && !card.querySelector(".chart-fallback")) {
        const note = document.createElement("p");
        note.className = "chart-fallback";
        note.textContent = "Không tải được Chart.js (CDN). Số liệu vẫn hiển thị ở bảng bên dưới.";
        canvas.replaceWith(note);
      }
    });
    return;
  }
  const { summary } = data;
  const labels = AGENT_KEYS.map((k) => AGENT_LABELS[k]);
  const colors = AGENT_KEYS.map((k) => AGENT_COLORS[k]);
  const defaults = chartDefaults();

  charts.composite = new Chart($("#chart-composite"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Điểm /100",
        data: AGENT_KEYS.map((k) => summary[k].composite_score),
        backgroundColor: colors.map((c) => c + "cc"),
        borderColor: colors,
        borderWidth: 1,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: defaults.color }, grid: { color: defaults.borderColor } },
        x: { ticks: { color: defaults.color }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  charts.tactical = new Chart($("#chart-tactical"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Đúng / Tổng",
        data: AGENT_KEYS.map((k) => summary[k].tactical_correct),
        backgroundColor: colors.map((c) => c + "99"),
        borderColor: colors,
        borderWidth: 1,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: summary.minimax.tactical_total,
          ticks: { stepSize: 1, color: defaults.color },
          grid: { color: defaults.borderColor },
        },
        x: { ticks: { color: defaults.color }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  charts.time = new Chart($("#chart-time"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "ms",
        data: AGENT_KEYS.map((k) => summary[k].avg_think_ms),
        backgroundColor: colors.map((c) => c + "77"),
        borderColor: colors,
        borderWidth: 1,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, ticks: { color: defaults.color }, grid: { color: defaults.borderColor } },
        x: { ticks: { color: defaults.color }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  const maxRank = 3;
  const maxDelta = Math.max(...AGENT_KEYS.map((x) => summary[x].avg_heuristic_delta), 1);
  const maxTime  = Math.max(...AGENT_KEYS.map((x) => summary[x].avg_think_ms), 1);
  charts.radar = new Chart($("#chart-radar"), {
    type: "radar",
    data: {
      labels: ["Chiến thuật", "Heuristic Δ", "Tốc độ", "TH dẫn đầu", "Điểm tổng"],
      datasets: AGENT_KEYS.map((k) => {
        const s = summary[k];
        return {
          label: AGENT_LABELS[k],
          data: [
            s.tactical_rate * 100,
            (s.avg_heuristic_delta / maxDelta) * 100,
            (1 - s.avg_think_ms / maxTime) * 100,
            (s.best_rank_count / data.scenario_count) * 100,
            s.composite_score,
          ],
          borderColor: AGENT_COLORS[k],
          backgroundColor: AGENT_COLORS[k] + "33",
          borderWidth: 2,
          pointRadius: 3,
        };
      }),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 8, right: 16, bottom: 8, left: 16 } },
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: { display: false, backdropColor: "transparent" },
          grid: { color: defaults.borderColor },
          angleLines: { color: defaults.borderColor },
          pointLabels: {
            color: defaults.color,
            font: { size: 10 },
            padding: 6,
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: defaults.color, boxWidth: 10, padding: 10, font: { size: 11 } },
        },
      },
    },
  });
}

function renderScoreBreakdown(summary) {
  const b = summary.score_breakdown;
  if (!b) return "";
  return `<p class="score-breakdown">CT ${b.tactical} · TH ${b.rank} · H ${b.heuristic} · Tốc độ ${b.speed}</p>`;
}

function renderSummaryCards(data) {
  const el = $("#summary-cards");
  el.innerHTML = AGENT_KEYS.map((key) => {
    const s = data.summary[key];
    const rankBadge = s.overall_rank === 1 ? "rank-gold" : s.overall_rank === 2 ? "rank-silver" : "rank-bronze";
    return `
      <article class="agent-summary glass">
        <div class="agent-summary-head">
          <span class="agent-dot" style="background:${AGENT_COLORS[key]}"></span>
          <h4>${s.label}</h4>
          <span class="rank-badge ${rankBadge}">#${s.overall_rank}</span>
        </div>
        <p class="agent-model-name">${s.name}</p>
        <dl class="agent-stats">
          <div><dt>Chiến thuật</dt><dd>${s.tactical_correct}/${s.tactical_total} (${(s.tactical_rate * 100).toFixed(0)}%)</dd></div>
          <div><dt>Heuristic Δ</dt><dd>${s.avg_heuristic_delta > 0 ? "+" : ""}${s.avg_heuristic_delta}</dd></div>
          <div><dt>Thời gian TB</dt><dd>${s.avg_think_ms} ms</dd></div>
          <div><dt>Dẫn đầu TH</dt><dd>${s.best_rank_count}/${data.scenario_count}</dd></div>
          <div><dt>Điểm tổng</dt><dd class="score-big">${s.composite_score}</dd></div>
          ${renderScoreBreakdown(s)}
        </dl>
      </article>
    `;
  }).join("");
}

function renderOverview(data) {
  const ov = data.overview;
  const section = $("#overview-section");
  if (!ov || !section) return;

  section.classList.remove("hidden");
  const lead = $("#overview-lead");
  if (lead) {
    lead.textContent =
      "TH01–03 kiểm tra chiến thuật bắt buộc; TH04+ là ván thực (chỉ search/DQN, không shortcut). "
      + "Điểm tổng = chiến thuật + dẫn đầu TH + heuristic + tốc độ — không cộng bonus giả.";
  }

  const table = $("#pairwise-table");
  if (table && ov.pairwise) {
    const header = `<tr><th></th>${AGENT_KEYS.map((k) => `<th>${AGENT_LABELS[k]}</th>`).join("")}</tr>`;
    const rows = AGENT_KEYS.map((a) => {
      const cells = AGENT_KEYS.map((b) => {
        if (a === b) return `<td>—</td>`;
        const p = ov.pairwise[a][b];
        const cls = p.wins > p.losses ? "win-cell" : "";
        return `<td class="${cls}">${p.wins}W ${p.ties}H ${p.losses}L</td>`;
      }).join("");
      return `<tr><th>${AGENT_LABELS[a]}</th>${cells}</tr>`;
    }).join("");
    table.innerHTML = `<thead>${header}</thead><tbody>${rows}</tbody>`;
  }

  const note = $("#pairwise-note");
  if (note && ov.hybrid_vs_minimax) {
    const h = ov.hybrid_vs_minimax;
    note.textContent =
      `Hybrid thắng ${h.hybrid_wins}/${h.scenario_count} TH so với Minimax `
      + `(Minimax thắng ${h.minimax_wins}, hòa ${h.ties}). `
      + "W = xếp hạng nước đi tốt hơn trên cùng một TH.";
  }

  const rolesEl = $("#roles-list");
  if (rolesEl && ov.roles) {
    rolesEl.innerHTML = AGENT_KEYS.map((k) => {
      const r = ov.roles[k];
      return `
        <div class="role-row">
          <strong>${r.label}</strong>
          ${r.strength}
          <span>Dùng khi: ${r.best_for}</span>
        </div>`;
    }).join("");
  }

  const metricsEl = $("#overview-metrics");
  if (metricsEl && ov.metrics_explained) {
    metricsEl.innerHTML = ov.metrics_explained.map((m) => `<li>${m}</li>`).join("");
  }
}

function renderEvaluation(data) {
  const ev = data.evaluation;
  const strength = data.strength_winner || data.winner;

  $("#winner-title").textContent =
    `${data.winner.label} — điểm tổng hợp cao nhất (${data.summary[data.winner.key].composite_score}/100)`;

  const strengthLine = $("#strength-headline");
  if (strengthLine && strength) {
    const s = data.summary[strength.key];
    strengthLine.textContent =
      `Chất lượng nước đi: ${strength.label} dẫn đầu `
      + `${s.best_rank_count}/${data.scenario_count} TH`;
  }

  $("#winner-headline").textContent = ev.headline;

  $("#eval-bullets").innerHTML = ev.bullets
    .map((b) => `<li>${b}</li>`)
    .join("");

  $("#eval-agents").innerHTML = ev.agent_details
    .map(
      (d) => `
      <div class="eval-agent-row">
        <strong>${d.agent}</strong>
        <p><span class="eval-tag eval-good">Mạnh</span> ${d.strengths}</p>
        <p><span class="eval-tag eval-warn">Yếu</span> ${d.weaknesses}</p>
      </div>
    `
    )
    .join("");
}

function formatMove(move) {
  if (!move) return "—";
  return `(${move[0]}, ${move[1]})`;
}

function renderBoardMini(board, size, highlightMoves = {}) {
  let html = `<div class="mini-board" style="--bs:${size}">`;
  for (let r = 0; r < size; r += 1) {
    for (let c = 0; c < size; c += 1) {
      const v = board[r][c];
      const cls = v === 1 ? "x" : v === 2 ? "o" : "";
      let extra = "";
      for (const [agent, move] of Object.entries(highlightMoves)) {
        if (move && move[0] === r && move[1] === c) {
          extra += ` highlight-${agent}`;
        }
      }
      html += `<span class="mini-cell ${cls}${extra}"></span>`;
    }
  }
  html += "</div>";
  return html;
}

function renderScenarios(data) {
  const el = $("#scenario-list");
  el.innerHTML = data.scenarios
    .map((sc, idx) => {
      const highlights = {};
      AGENT_KEYS.forEach((k) => {
        highlights[k] = sc.agents[k].move;
      });

      const rows = AGENT_KEYS.map((k) => {
        const a = sc.agents[k];
        const ok = a.is_expected === true ? "ok" : a.is_expected === false ? "fail" : "na";
        const okText = a.is_expected === true ? "✓ Đúng" : a.is_expected === false ? "✗ Sai" : "—";
        return `
          <tr class="agent-row-${k}">
            <td><span class="agent-dot inline" style="background:${AGENT_COLORS[k]}"></span> ${AGENT_LABELS[k]}</td>
            <td class="mono">${formatMove(a.move)}</td>
            <td class="mono">${a.think_ms} ms</td>
            <td class="mono">${a.heuristic_delta > 0 ? "+" : ""}${a.heuristic_delta}</td>
            <td><span class="result-pill ${ok}">${okText}</span></td>
            <td><span class="rank-pill rank-${a.rank}">#${a.rank}</span></td>
          </tr>
        `;
      }).join("");

      const expected =
        sc.expected_moves && sc.expected_moves.length
          ? sc.expected_moves.map(formatMove).join(", ")
          : "Theo heuristic (chất lượng nước đi)";

      return `
        <article class="scenario-card">
          <header class="scenario-head">
            <div>
              <span class="scenario-id">${sc.id ? sc.id.toUpperCase() : "TH" + String(idx + 1).padStart(2, "0")}</span>
              <h4>${sc.name}</h4>
              <p>${sc.description}</p>
            </div>
            <span class="category-tag">${sc.category}</span>
          </header>
          <div class="scenario-body">
            ${renderBoardMini(sc.board, sc.board_size, highlights)}
            <div class="scenario-table-wrap">
              <p class="expected-line"><strong>Đáp án chuẩn:</strong> ${expected}</p>
              <table class="scenario-table">
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th>Nước đi</th>
                    <th>Thời gian</th>
                    <th>Δ Heuristic</th>
                    <th>Chiến thuật</th>
                    <th>Hạng</th>
                  </tr>
                </thead>
                <tbody>${rows}</tbody>
              </table>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function showLoading(show) {
  $("#loading-panel").classList.toggle("hidden", !show);
  $("#btn-run").disabled = show;
}

function showError(msg) {
  const panel = $("#error-panel");
  panel.textContent = msg;
  panel.classList.remove("hidden");
}

function hideError() {
  $("#error-panel").classList.add("hidden");
}

function renderResults(data) {
  window.lastBenchmark = data;
  $("#results-panel").classList.remove("hidden");
  showCacheStatus(data);
  renderOverview(data);
  renderSummaryCards(data);
  renderEvaluation(data);
  renderCharts(data);
  renderScenarios(data);
}

function applyPageLabels(scenarioSet, data) {
  const scenarioLabels = {
    basic: "10 TH (3 chiến thuật + 7 ván thực)",
    advanced: "10 ván thực nâng cao",
    all: "20 TH (tiêu chuẩn + nâng cao)",
  };
  const setLabel = scenarioLabels[scenarioSet] || "benchmark";
  const subtitle = $("#compare-subtitle");
  if (subtitle) subtitle.textContent = `Minimax · DQN · Hybrid — ${setLabel}`;
  const secTitle = $("#scenario-section-title");
  if (secTitle) secTitle.textContent = `${data.scenario_count} tình huống thử nghiệm`;
  const secDesc = $("#scenario-section-desc");
  if (secDesc) {
    secDesc.textContent = scenarioSet === "advanced"
      ? "10 ván replay (search-only) — bàn dày, phân biệt depth search + DQN ordering"
      : scenarioSet === "all"
      ? "20 TH: 3 chiến thuật + 17 ván thực (search-only, không bonus giả)"
      : "3 TH chiến thuật + 7 ván replay search-only — ai search sâu hơn sẽ dẫn đầu TH";
  }
}

async function refreshCacheHint(formData) {
  const params = compareParams(formData);
  try {
    const cached = await fetchCachedCompare(params);
    updateRunButton(Boolean(cached));
    return cached;
  } catch {
    updateRunButton(false);
    return null;
  }
}

async function runBenchmark(formData) {
  hideError();
  const scenarioSet = formData.get("scenario_set") || "basic";
  const params = compareParams(formData);

  showLoading(true);
  $("#results-panel").classList.add("hidden");

  const loadMsg = $("#loading-msg");
  const hint = $("#loading-hint");
  if (loadMsg) loadMsg.textContent = "Đang tải kết quả…";

  try {
    const cached = await fetchCachedCompare(params);
    if (cached) {
      if (loadMsg) loadMsg.textContent = "Đã tải từ DB — không chạy lại benchmark";
      applyPageLabels(scenarioSet, cached);
      renderResults(cached);
      updateRunButton(true);
      return;
    }

    const timeHints = {
      basic: "~10–25 giây",
      advanced: "~20–45 giây",
      all: "EXPERT có thể vài phút",
    };
    if (loadMsg) {
      loadMsg.textContent = `Chưa có trong DB — đang chạy benchmark (${timeHints[scenarioSet] || "…"})…`;
    }
    if (hint) {
      hint.textContent = "Lần chạy này sẽ được lưu vĩnh viễn; lần sau chỉ tải từ DB.";
    }

    const data = await api("/api/compare/run", {
      method: "POST",
      body: JSON.stringify({ ...params, force: false }),
    });
    applyPageLabels(scenarioSet, data);
    renderResults(data);
    updateRunButton(true);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    showLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  const form = $("#compare-form");
  loadBoardSizes().then(async () => {
    if (!form) return;
    const cached = await refreshCacheHint(new FormData(form));
    if (cached) {
      applyPageLabels(cached.scenario_set || "basic", cached);
      renderResults(cached);
    }
  });
  form.addEventListener("change", () => {
    refreshCacheHint(new FormData(form));
    $("#results-panel").classList.add("hidden");
    showCacheStatus(null);
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runBenchmark(new FormData(e.target));
  });
});

async function loadBoardSizes() {
  try {
    const opts = await api("/api/options");
    const sel = $("#board_size");
    sel.innerHTML = opts.board_sizes
      .map((n) => `<option value="${n}">${n}×${n}</option>`)
      .join("");
    sel.value = String(opts.board_sizes.includes(15) ? 15 : opts.board_sizes.at(-1));
  } catch (err) {
    console.error(err);
  }
}
