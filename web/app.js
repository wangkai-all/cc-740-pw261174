/* Sequential inspection web UI — vanilla JS, no build step. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const api = async (path, options = {}) => {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
};
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const STATUS_LABEL = { continue: "继续采样", accepted: "接受", rejected: "拒收" };
let currentBatchId = null;
let presets = [];

// ---------- tabs ----------
document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "audit") loadAudit();
    if (btn.dataset.tab === "plans") loadPresetsTable();
  });
});

// ---------- presets / plans ----------
async function loadPresets() {
  const data = await api("/api/presets");
  presets = data.presets;
  $("#preset-select").innerHTML = presets
    .map((p) => `<option value="${esc(p.name)}">${esc(p.label || p.name)}</option>`)
    .join("");
  const picks = $("#experiment-plan-picks");
  picks.innerHTML = presets
    .map(
      (p, i) =>
        `<label style="display:flex;align-items:center;gap:8px;grid-column:1/-1">
        <input type="checkbox" name="plan" value="${esc(p.name)}" ${i < 2 ? "checked" : ""}>
        ${esc(p.label || p.name)} <span class="muted">(${esc(p.method)})</span></label>`
    )
    .join("");
}

async function loadPresetsTable() {
  const tbody = $("#preset-table tbody");
  tbody.innerHTML = presets
    .map((p) => {
      const alpha = p.method === "cs_beta_mixture" ? p.alpha_cs : p.alpha;
      return `<tr><td>${esc(p.name)}</td><td>${esc(p.method)}</td>
        <td>${esc(p.sampling)}</td><td>${p.p0}</td><td>${p.p1}</td>
        <td>${alpha ?? ""}</td><td>${p.beta ?? ""}</td><td>${p.N ?? ""}</td></tr>`;
    })
    .join("");
}

$("#plan-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  ["p0", "p1", "alpha", "beta", "alpha_cs"].forEach((k) => {
    if (payload[k] !== "") payload[k] = Number(payload[k]);
  });
  if (payload.N !== "") payload.N = Number(payload.N);
  else delete payload.N;
  try {
    const data = await api("/api/plans", { method: "POST", body: payload });
    $("#plan-assumptions").textContent =
      JSON.stringify(data.plan, null, 2) + "\n\n假设: " + data.assumptions;
  } catch (err) {
    $("#plan-assumptions").textContent = "方案无效: " + err.message;
  }
});

// ---------- batches ----------
async function loadBatches(selectId) {
  const data = await api("/api/batches");
  const tbody = $("#batch-list tbody");
  tbody.innerHTML = data.batches
    .map(
      (b) => `<tr data-id="${esc(b.batch_id)}">
      <td>${esc(b.batch_id)}</td><td>${esc(b.method)}</td>
      <td><span class="pill ${b.status}">${STATUS_LABEL[b.status]}</span></td>
      <td>${b.n}/${b.k}</td><td>${b.round}</td>
      <td>${b.released ? esc(b.released_at) : "—"}</td></tr>`
    )
    .join("");
  tbody.querySelectorAll("tr").forEach((tr) =>
    tr.addEventListener("click", () => selectBatch(tr.dataset.id))
  );
}

$("#create-batch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const presetName = $("#preset-select").value;
  const preset = presets.find((p) => p.name === presetName);
  const lot_note = e.target.elements.lot_note.value;
  try {
    const data = await api("/api/batches", {
      method: "POST",
      body: { plan: preset, lot_note },
    });
    await loadBatches();
    selectBatch(data.batch_id);
  } catch (err) {
    alert("创建失败: " + err.message);
  }
});

async function selectBatch(batchId) {
  currentBatchId = batchId;
  const state = await api(`/api/batches/${encodeURIComponent(batchId)}`);
  $("#batch-detail-card").hidden = false;
  $("#history-card").hidden = false;
  $("#batch-id").textContent = batchId;
  $("#batch-assumptions").textContent = state.assumptions;

  $("#decision-banner").innerHTML =
    `<div class="banner ${state.status}">当前判定：${STATUS_LABEL[state.status]}
     ${state.released ? `（已放行，时间 ${esc(state.released_at)} 不可改写）` : ""}</div>`;

  const c = state.current;
  const statRows =
    c.statistic_name === "confidence_sequence"
      ? [["方法", "Beta 混合置信序列"], ["n", c.n], ["不合格 k", c.k],
         ["区间下界", fmt(c.cs_lower)], ["区间上界", fmt(c.cs_upper)],
         ["p0 / p1", `${state.plan.p0} / ${state.plan.p1}`]]
      : [["方法", state.plan.method], ["n", c.n], ["不合格 k", c.k],
         ["对数似然比", fmt(c.statistic)],
         ["下边界 B", fmt(c.lower_boundary)], ["上边界 A", fmt(c.upper_boundary)]];
  $("#batch-stats").innerHTML = statRows
    .map(([d, v]) => `<dt>${d}</dt><dd>${esc(v)}</dd>`)
    .join("");

  $("#boundary-view").textContent = JSON.stringify(
    { projection: state.projection, rounds: state.rounds },
    null, 2
  );

  $("#history-table tbody").innerHTML = state.history
    .map((h) => {
      const stat =
        h.statistic_name === "confidence_sequence"
          ? `[${fmt(h.cs_lower)}, ${fmt(h.cs_upper)}]`
          : fmt(h.statistic);
      return `<tr><td>${h.seq}</td><td>${esc(h.item_id)}</td>
        <td>${h.k > (state.history[h.seq - 2]?.k ?? 0) ? "不合格" : "合格"}</td>
        <td>${h.n}</td><td>${h.k}</td><td>${stat}</td>
        <td><span class="pill ${h.decision}">${STATUS_LABEL[h.decision]}</span></td></tr>`;
    })
    .join("");

  $("#withdrawn-table tbody").innerHTML =
    state.withdrawn
      .map(
        (w) => `<tr><td>${esc(w.at)}</td><td>${esc(w.item_id)}</td>
        <td>${esc(w.reason)}</td><td>${esc(w.mode)}</td></tr>`
      )
      .join("") +
    state.corrections
      .map(
        (c2) => `<tr><td>${esc(c2.at)}</td>
        <td>${esc(c2.item_id ?? "（整批更正）")}</td>
        <td>${esc(c2.reason)}</td><td>post-release correction</td></tr>`
      )
      .join("");

  const decided = state.status !== "continue";
  $("#open-round-btn").disabled = decided || state.released;
  $("#release-btn").disabled = decided === false || state.released;
}

function fmt(x) {
  if (x === null || x === undefined) return "—";
  if (x === Infinity || x === "Infinity") return "∞";
  if (x === -Infinity || x === "-Infinity") return "-∞";
  return typeof x === "number" ? Number(x.toFixed(5)).toString() : String(x);
}

$("#intake-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fb = $("#intake-feedback");
  const text = e.target.elements.entries.value.trim();
  const entries = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [item_id, verdict] = line.split(",").map((s) => s.trim());
      if (!item_id || verdict === undefined)
        throw new Error(`行格式错误: ${line}（应为 件号,合格/不合格）`);
      return { item_id, defective: verdict === "不合格" || verdict === "1" };
    });
  try {
    const data = await api(
      `/api/batches/${encodeURIComponent(currentBatchId)}/items`,
      { method: "POST", body: { entries } }
    );
    const i = data.intake;
    fb.className = "feedback ok";
    fb.textContent =
      `记录 ${i.recorded.length} 件；拦截重复 ${i.blocked_duplicates.length} 件` +
      (i.not_recorded_after_decision.length
        ? `；判定后未再收 ${i.not_recorded_after_decision.length} 件`
        : "");
    e.target.reset();
    await loadBatches();
    await selectBatch(currentBatchId);
  } catch (err) {
    fb.className = "feedback err";
    fb.textContent = err.message;
  }
});

$("#open-round-btn").addEventListener("click", async () => {
  await api(`/api/batches/${encodeURIComponent(currentBatchId)}/round`, {
    method: "POST",
  });
  await selectBatch(currentBatchId);
});

$("#withdraw-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const item_id = e.target.elements.item_id.value;
  const reason = e.target.elements.reason.value;
  try {
    await api(`/api/batches/${encodeURIComponent(currentBatchId)}/withdraw`, {
      method: "POST",
      body: { item_id, reason },
    });
    e.target.reset();
    await loadBatches();
    await selectBatch(currentBatchId);
  } catch (err) {
    alert("撤回失败: " + err.message);
  }
});

$("#release-btn").addEventListener("click", async () => {
  if (!confirm("放行后统计状态与放行时间将冻结，只能追加后续更正。确认？")) return;
  await api(`/api/batches/${encodeURIComponent(currentBatchId)}/release`, {
    method: "POST",
  });
  await loadBatches();
  await selectBatch(currentBatchId);
});

$("#correction-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api(`/api/batches/${encodeURIComponent(currentBatchId)}/correction`, {
      method: "POST",
      body: { note: e.target.elements.note.value },
    });
    e.target.reset();
    await selectBatch(currentBatchId);
  } catch (err) {
    alert("更正失败: " + err.message);
  }
});

$("#verify-btn").addEventListener("click", async () => {
  const data = await api(
    `/api/batches/${encodeURIComponent(currentBatchId)}/verify`
  );
  $("#verify-view").textContent = JSON.stringify(data, null, 2);
});

// ---------- experiment ----------
$("#experiment-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const chosen = [...document.querySelectorAll('input[name="plan"]:checked')].map(
    (el) => el.value
  );
  if (!chosen.length) return alert("至少选择一套风险配置");
  const plans = chosen.map((name) => presets.find((p) => p.name === name));
  const body = {
    plans,
    true_ps: fd.get("true_ps").split(",").map((s) => Number(s.trim())),
    replications: Number(fd.get("replications")),
    round_size: Number(fd.get("round_size")),
    base_seed: Number(fd.get("base_seed")),
  };
  $("#experiment-results").textContent = "运行中……";
  try {
    const data = await api("/api/simulate", { method: "POST", body });
    renderExperiment(data);
  } catch (err) {
    $("#experiment-results").textContent = "实验失败: " + err.message;
  }
});

function renderExperiment(data) {
  const maxN = Math.max(
    ...data.grids.flatMap((g) => g.cells.map((c) => c.avg_n))
  );
  $("#experiment-results").innerHTML = data.grids
    .map((g, gi) => {
      const p = g.plan;
      const title = `${gi + 1}. ${esc(p.name || p.method)} — ${esc(p.method)}`;
      const rows = g.cells
        .map((c) => {
          const acceptW = (c.accept_rate * 160).toFixed(1);
          const rejectW = (c.reject_rate * 160).toFixed(1);
          const nW = (c.avg_n / maxN * 160).toFixed(1);
          return `<div class="bar-cell"><span class="label">p=${c.true_p}</span>
            <div class="bar" style="width:${acceptW}px" title="接受率"></div>
            <span>接受 ${(c.accept_rate * 100).toFixed(1)}%</span>
            <div class="bar reject" style="width:${rejectW}px"></div>
            <span>拒收 ${(c.reject_rate * 100).toFixed(1)}%</span></div>
          <div class="bar-cell"><span class="label">平均检验量</span>
            <div class="bar" style="width:${nW}px;opacity:.7"></div>
            <span>ASN=${c.avg_n.toFixed(1)}（区间 ${c.min_n}–${c.max_n}）</span></div>`;
        })
        .join("");
      return `<div class="experiment-grid"><h4>${title}</h4>${rows}</div>`;
    })
    .join("") +
    `<p class="muted">种子 ${data.base_seed}，每点 ${data.replications} 次重复；各配置使用配对的相同虚构货物。</p>`;
}

// ---------- hand check / audit ----------
$("#handcheck-btn").addEventListener("click", async () => {
  const data = await api("/api/handcheck");
  $("#handcheck-view").textContent = JSON.stringify(data, null, 2);
});

async function loadAudit() {
  const data = await api("/api/audit");
  $("#audit-view").textContent = data.records
    .map((r) => JSON.stringify(r, ensureFinite))
    .join("\n");
}
$("#audit-refresh").addEventListener("click", loadAudit);

function ensureFinite(_key, value) {
  if (value === Infinity) return "Infinity";
  if (value === -Infinity) return "-Infinity";
  return value;
}

// ---------- boot ----------
(async function boot() {
  await loadPresets();
  await loadPresetsTable();
  await loadBatches();
})();
