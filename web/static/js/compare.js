/**
 * Trang so sánh Minimax / DQN / Hybrid — 10 TH benchmark.
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

// Cache kết quả benchmark theo key scenario_set__difficulty__board_size
const benchmarkCache = new Map();

function cacheKey(formData) {
  return [
    formData.get("scenario_set") || "basic",
    formData.get("difficulty") || "MEDIUM",
    formData.get("board_size") || "15",
  ].join("__");
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
          <div><dt>Dẫn đầu TH</dt><dd>${s.best_rank_count}/10</dd></div>
          <div><dt>Điểm tổng</dt><dd class="score-big">${s.composite_score}</dd></div>
        </dl>
      </article>
    `;
  }).join("");
}

function renderEvaluation(data) {
  const ev = data.evaluation;
  $("#winner-title").textContent = `${data.winner.label} tốt nhất tổng thể`;
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
  renderSummaryCards(data);
  renderEvaluation(data);
  renderCharts(data);
  renderScenarios(data);
}

function applyPageLabels(scenarioSet, data) {
  const scenarioLabels = {
    basic: "10 TH tiêu chuẩn (3 chiến thuật + 7 chiến lược)",
    advanced: "10 TH chiến lược nâng cao",
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
      ? "10 TH chiến lược phức tạp — không có nước tức thời, agents phải tự đánh giá vị trí sâu"
      : scenarioSet === "all"
      ? "20 TH: 3 chiến thuật + 7 chiến lược tiêu chuẩn + 10 chiến lược nâng cao"
      : "3 TH chiến thuật bắt buộc + 7 TH chiến lược — phân biệt rõ ràng 3 agents";
  }
}

async function runBenchmark(formData) {
  hideError();
  const scenarioSet = formData.get("scenario_set") || "basic";

  // Trả kết quả từ cache ngay lập tức nếu đã chạy trước đó
  const key = cacheKey(formData);
  if (benchmarkCache.has(key)) {
    const cached = benchmarkCache.get(key);
    applyPageLabels(scenarioSet, cached);
    renderResults(cached);
    return;
  }

  showLoading(true);
  $("#results-panel").classList.add("hidden");

  const timeHints = {
    basic:    "~5–15 giây",
    advanced: "~15–35 giây",
    all:      "~25–50 giây",
  };
  const loadMsg = $("#loading-msg");
  if (loadMsg) loadMsg.textContent = `Đang chạy benchmark — ước lượng ${timeHints[scenarioSet] || "30–60 giây"}…`;

  try {
    const body = {
      difficulty: formData.get("difficulty"),
      board_size: Number(formData.get("board_size")),
      double_end_block_rule: true,
      ai_aggressive: true,
      scenario_set: scenarioSet,
    };
    const data = await api("/api/compare/run", {
      method: "POST",
      body: JSON.stringify(body),
    });
    benchmarkCache.set(key, data);   // Lưu cache
    applyPageLabels(scenarioSet, data);
    renderResults(data);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    showLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  loadBoardSizes();
  $("#compare-form").addEventListener("submit", (e) => {
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
