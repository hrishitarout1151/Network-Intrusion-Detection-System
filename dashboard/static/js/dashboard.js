const ENGINE_META = [
  { key: "signature_score", label: "Signature", color: "#E5484D" },
  { key: "isolation_score", label: "Isolation Forest", color: "#8B7FE8" },
  { key: "ewma_score", label: "EWMA Baseline", color: "#45C6D6" },
  { key: "rf_confidence", label: "Random Forest", color: "#E8C468" },
];

let state = { payload: null, filter: "ALL", selectedIdx: null, typeChart: null };

async function loadDashboard() {
  setBusy(true, "Loading…");
  try {
    const res = await fetch("/api/dashboard");
    if (res.status === 404) {
      setBusy(false, "No data — run a simulation");
      return;
    }
    state.payload = await res.json();
    render();
    setBusy(false, "Live");
  } catch (e) {
    setBusy(false, "Error");
    console.error(e);
  }
}

function setBusy(busy, text) {
  const pill = document.getElementById("statusPill");
  pill.classList.toggle("busy", busy);
  document.getElementById("statusText").textContent = text;
}

function render() {
  const p = state.payload;
  if (!p) return;
  document.getElementById("kpiWindows").textContent = p.totals.total_windows.toLocaleString();
  document.getElementById("kpiCritical").textContent = p.totals.critical;
  document.getElementById("kpiHigh").textContent = p.totals.high;
  document.getElementById("kpiMedium").textContent = p.totals.medium;
  document.getElementById("kpiIPs").textContent = p.totals.unique_ips_flagged;
  renderTable();
  renderTypeChart();
}

function renderTable() {
  const p = state.payload;
  const tbody = document.getElementById("alertRows");
  tbody.innerHTML = "";
  const rows = p.alerts.filter(a => state.filter === "ALL" || a.severity === state.filter);

  rows.forEach((a, i) => {
    const tr = document.createElement("tr");
    if (state.selectedIdx === a._origIdx) tr.classList.add("selected");
    const winTime = new Date(a.window_start).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    tr.innerHTML = `
      <td><span class="sev-badge sev-${a.severity}">${a.severity}</span></td>
      <td class="ip-cell">${a.src_ip}</td>
      <td>${a.rf_predicted_type}</td>
      <td class="risk-cell">${a.risk_score.toFixed(2)}</td>
      <td>${winTime}</td>
    `;
    tr.addEventListener("click", () => selectAlert(a, tr));
    tbody.appendChild(tr);
  });

  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-dim); padding:30px;">No alerts at this severity</td></tr>`;
  }
}

function selectAlert(alert, trEl) {
  document.querySelectorAll("#alertRows tr").forEach(r => r.classList.remove("selected"));
  trEl.classList.add("selected");
  renderFusion(alert);
}

function renderFusion(a) {
  const body = document.getElementById("fusionBody");

  const gaugeSvg = buildGaugeSvg(a);

  const bars = ENGINE_META.map(m => {
    const v = a[m.key] || 0;
    return `
      <div class="engine-row">
        <span class="label">${m.label}</span>
        <div class="engine-bar-track"><div class="engine-bar-fill" style="width:${(v * 100).toFixed(0)}%; background:${m.color};"></div></div>
        <span class="val">${(v * 100).toFixed(0)}%</span>
      </div>`;
  }).join("");

  let reasonsHtml;
  if (a.signature_hits && a.signature_hits.length > 0) {
    reasonsHtml = a.signature_hits.map(h => `<div class="reason-item"><b>${h.rule}</b> — ${h.reason}</div>`).join("");
  } else {
    reasonsHtml = `<p class="no-sig">No deterministic signature fired — this alert is driven by the anomaly / behavioral engines only.</p>`;
  }

  body.innerHTML = `
    <div class="fusion-target">
      <div class="ip">${a.src_ip}</div>
      <div class="meta">Predicted: <b>${a.rf_predicted_type}</b> &nbsp;·&nbsp; Severity: ${a.severity} &nbsp;·&nbsp; Window ${new Date(a.window_start).toLocaleString()}</div>
    </div>
    <div class="gauge-wrap">${gaugeSvg}</div>
    <div class="engine-rows">${bars}</div>
    <div class="reasons">
      <h3>Why this fired</h3>
      ${reasonsHtml}
      <div class="reason-item" style="border-left-color:#45C6D6;"><b>Top behavioral deviation</b> — ${a.ewma_top_deviation}</div>
    </div>
  `;
}

function buildGaugeSvg(a) {
  const size = 180, cx = size / 2, cy = size / 2;
  const baseRadius = 42, gap = 12;
  let rings = "";
  ENGINE_META.forEach((m, i) => {
    const r = baseRadius + i * gap;
    const v = Math.max(0, Math.min(1, a[m.key] || 0));
    const circ = 2 * Math.PI * r;
    const dash = circ * v;
    rings += `
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#1B2438" stroke-width="7"/>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${m.color}" stroke-width="7"
        stroke-dasharray="${dash} ${circ}" stroke-linecap="round"
        transform="rotate(-90 ${cx} ${cy})" opacity="0.95"/>
    `;
  });
  const pct = Math.round(a.risk_score * 100);
  return `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      ${rings}
      <text x="${cx}" y="${cy - 4}" text-anchor="middle" font-family="Space Grotesk" font-weight="700" font-size="26" fill="#E8ECF4">${pct}</text>
      <text x="${cx}" y="${cy + 15}" text-anchor="middle" font-family="JetBrains Mono" font-size="9.5" fill="#7C8AA3" letter-spacing="1">RISK</text>
    </svg>`;
}

function renderTypeChart() {
  const p = state.payload;
  const types = Object.keys(p.severity_by_type || {});
  const severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"];
  const colors = { CRITICAL: "#E5484D", HIGH: "#F5A623", MEDIUM: "#E8C468", LOW: "#45C6D6", INFO: "#4E5A73" };

  const datasets = severities.map(sev => ({
    label: sev,
    data: types.map(t => (p.severity_by_type[t] && p.severity_by_type[t][sev]) || 0),
    backgroundColor: colors[sev],
    borderRadius: 3,
  })).filter(ds => ds.data.some(v => v > 0));

  const ctx = document.getElementById("typeChart");
  if (state.typeChart) state.typeChart.destroy();
  state.typeChart = new Chart(ctx, {
    type: "bar",
    data: { labels: types, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, ticks: { color: "#7C8AA3", font: { family: "JetBrains Mono", size: 11 } }, grid: { color: "#1B2438" } },
        y: { stacked: true, ticks: { color: "#7C8AA3", font: { family: "JetBrains Mono", size: 11 } }, grid: { color: "#1B2438" } },
      },
      plugins: { legend: { labels: { color: "#E8ECF4", font: { family: "JetBrains Mono", size: 11 } } } },
    },
  });
}

document.getElementById("severityFilters").addEventListener("click", (e) => {
  if (!e.target.dataset.sev) return;
  document.querySelectorAll("#severityFilters .chip").forEach(c => c.classList.remove("active"));
  e.target.classList.add("active");
  state.filter = e.target.dataset.sev;
  renderTable();
});

document.getElementById("simulateBtn").addEventListener("click", async () => {
  const btn = document.getElementById("simulateBtn");
  btn.disabled = true;
  btn.textContent = "Simulating…";
  setBusy(true, "Regenerating traffic + retraining…");
  try {
    const res = await fetch("/api/simulate", { method: "POST" });
    state.payload = await res.json();
    render();
    setBusy(false, "Live");
  } catch (e) {
    setBusy(false, "Error");
  }
  btn.disabled = false;
  btn.textContent = "Run New Simulation";
});

loadDashboard();
