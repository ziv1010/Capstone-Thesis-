// ============================================================================
// HGT Explainability Visualizer — experiment-by-experiment frontend
// ============================================================================

const state = {
  casePage: 1,
  caseLimit: 50,
  selectedCase: null,
  tableName: "typed_path_importance",
  loaded: {},
};

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 });

const COLUMN_LABELS = {
  path_family: "Legal path",
  relation_type: "Relation type",
  relation: "Relation",
  src_type: "Source",
  dst_type: "Target",
  evidence_type: "Evidence type",
  evidence_name: "Evidence",
  n_groups: "Groups",
  n_cases: "Cases",
  mean_delta_pred_proba: "Mean signed Δp",
  mean_abs_delta_pred_proba: "Mean importance",
  sum_abs_delta_pred_proba: "Total importance",
  flip_rate: "Flip rate",
  mean_masked_edge_count: "Masked edges",
  mean_attention_score: "Mean attention",
  counterfactual_attention_overlap: "Top-k overlap",
  ranker: "Ranker",
  k_requested: "k",
  mean_sufficiency_auc: "Sufficiency AUC",
  mean_comprehensiveness_auc: "Comprehensiveness AUC",
  confidence_bucket: "Bucket",
  accuracy: "Accuracy",
  mean_confidence: "Mean confidence",
  mean_evidence_purity: "Evidence purity",
  mean_top_abs_delta_pred_proba: "Top |Δp|",
  most_common_evidence_types: "Most common evidence types",
  bucket_share: "Bucket share",
  community_id: "Community",
  size: "Size",
  dominant_label: "Dominant label",
  dominant_domain_bucket: "Domain",
  high_confidence_wrong_n: "Confident wrong",
  embedding_cluster_id: "Embedding cluster",
  noise_rate: "Noise rate",
  normalized_mutual_info_all: "NMI",
  v_measure_all: "V-measure",
  adjusted_rand_all: "ARI",
  identity_scope: "Identity scope",
  group_by: "Grouped by",
  known_eval_case_share: "Known eval cases",
  eval_identity_overlap_share: "Identity overlap",
  identity_auc_roc: "Identity AUC",
  identity_log_loss_delta_vs_domain: "Log-loss Δ vs domain",
  identity_brier_delta_vs_domain: "Brier Δ vs domain",
  permutation_auc_p_value: "Permutation p",
  counterfactual_mean_abs_delta_pred_proba: "CF mean |Δp|",
  counterfactual_flip_rate: "CF flip rate",
  identity_name: "Identity",
  identity_type: "Identity type",
  train_support: "Train support",
  train_positive_rate: "Train label 1 rate",
  smoothed_lift_from_train_prior: "Lift vs train prior",
  eval_support: "Eval support",
  identity_score_combined: "Identity score",
  domain_baseline_score: "Domain baseline",
  identity_score_gap_vs_domain: "Score gap",
  identity_train_support_combined: "Identity support",
  mask_name: "Mask",
  mask_family: "Mask family",
  masked_edge_share: "Masked edge share",
  baseline_accuracy: "Baseline accuracy",
  masked_accuracy: "Masked accuracy",
  accuracy_drop: "Accuracy drop",
  baseline_macro_f1: "Baseline macro-F1",
  masked_macro_f1: "Masked macro-F1",
  macro_f1_drop: "Macro-F1 drop",
  confidence_drop: "Original confidence drop",
  mean_confidence_drop: "Max confidence drop",
  top_k_hubs: "Top-k hubs",
  feature_name: "Authority",
  feature_type: "Type",
  hub_rank: "Hub rank",
};

const VIEW_IDS = new Set(["overview", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9", "exp10", "expMask", "aggregate", "cases", "tables"]);

// ---------------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------------

function value(v, fallback = "") {
  return v === null || v === undefined || Number.isNaN(v) ? fallback : v;
}

function number(v, digits = 4) {
  if (v === null || v === undefined || v === "") return "";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (Math.abs(n) >= 1000) return fmt.format(n);
  return n.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function pct(v) {
  if (v === null || v === undefined || v === "") return "";
  const n = Number(v);
  if (!Number.isFinite(n)) return "";
  return `${(n * 100).toFixed(1)}%`;
}

function escapeHtml(s) {
  return String(value(s))
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path) {
  const response = await fetch(path);
  const payload = await response.json();
  if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
  return payload;
}

function cleanViewName(name) {
  return VIEW_IDS.has(name) ? name : "overview";
}

function setView(name, options = {}) {
  const viewName = cleanViewName(name);
  document.body.dataset.view = viewName;
  document.querySelectorAll(".tab").forEach((b) => {
    const active = b.dataset.view === viewName;
    b.classList.toggle("active", active);
    b.setAttribute("aria-current", active ? "page" : "false");
  });
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === viewName));
  if (options.updateHash !== false) history.replaceState(null, "", `#${viewName}`);
  if (options.scroll !== false) window.scrollTo({ top: 0, behavior: "smooth" });
  return viewName;
}

function byId(id) {
  return document.getElementById(id);
}

function metric(label, valueText, note = "", tone = "") {
  return `
    <div class="metric ${tone ? `metric-${tone}` : ""}" title="${escapeHtml(note)}">
      <div class="label">${escapeHtml(label)}</div>
      <div class="value">${escapeHtml(valueText)}</div>
      ${note ? `<div class="metric-note">${escapeHtml(note)}</div>` : ""}
    </div>
  `;
}

function bar(valueRaw, maxRaw, cls = "") {
  const valueNum = Math.abs(Number(valueRaw || 0));
  const maxNum = Math.max(Math.abs(Number(maxRaw || 0)), 1e-9);
  const width = Math.min(100, (valueNum / maxNum) * 100);
  return `<div class="bar-bg"><div class="bar-fill ${cls}" style="width:${width}%"></div></div>`;
}

function relationExplanation(relation) {
  const t = String(value(relation));
  if (!t) return "";
  if (t.startsWith("rev_")) return `${t}: reverse graph edge. The original legal relation is ${t.slice(4)}.`;
  return `${t}: original graph relation`;
}

function displayRelation(text, compact = false) {
  const raw = String(value(text));
  if (!raw) return "";
  const label = `<span class="rel ${raw.startsWith("rev_") ? "reverse" : ""}" title="${escapeHtml(relationExplanation(raw))}">${escapeHtml(raw)}</span>`;
  if (raw.startsWith("rev_")) {
    const note = `reverse of ${raw.slice(4)}`;
    return compact ? `${label}<span class="cell-note inline">${escapeHtml(note)}</span>` : `${label}<div class="cell-note">${escapeHtml(note)}</div>`;
  }
  return label;
}

function displayPath(path) {
  const raw = String(value(path));
  if (!raw) return "";
  const parts = raw.split("->");
  return `<span class="path-chain">${parts
    .map((part, i) => (i % 2 === 1 ? displayRelation(part, true) : `<span class="path-node">${escapeHtml(part)}</span>`))
    .join('<span class="arrow">-&gt;</span>')}</span>`;
}

function labelPill(label) {
  const t = String(value(label));
  const cls = t === "1" ? "good" : t === "-1" ? "bad" : "";
  return `<span class="pill ${cls}">${escapeHtml(t)}</span>`;
}

function renderInlineMarkup(text) {
  return String(value(text))
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function compactText(text, max = 96) {
  const s = String(value(text)).trim();
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1).trim()}…`;
}

function displayList(text, limit = 4) {
  const raw = String(value(text)).trim();
  if (!raw || raw === "None" || raw === "null") return `<span class="muted">none</span>`;
  const parts = raw
    .split(/\s+\|\s+|;\s+|\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
  if (!parts.length) return escapeHtml(raw);
  const shown = parts.slice(0, limit);
  const extra = parts.length - shown.length;
  return `<span class="mini-list">${shown.map((part) => `<span>${escapeHtml(compactText(part, 72))}</span>`).join("")}${extra > 0 ? `<em>+${extra} more</em>` : ""}</span>`;
}

function bestBy(rows, scoreFn) {
  return (rows || []).reduce((best, row) => {
    const score = Number(scoreFn(row));
    if (!Number.isFinite(score)) return best;
    if (!best || score > best.score) return { row, score };
    return best;
  }, null)?.row || {};
}

function rankerRow(rows, name) {
  return (rows || []).find((row) => String(row.ranker).toLowerCase() === name) || {};
}

function confidenceHighShare(rows) {
  const data = rows || [];
  const total = data.reduce((sum, row) => sum + (Number(row.n_cases) || 0), 0);
  if (!total) return null;
  const high = data.reduce((sum, row) => {
    const lower = Number(String(row.bucket || "").split(/[–-]/)[0]);
    return sum + (Number.isFinite(lower) && lower >= 0.8 ? Number(row.n_cases) || 0 : 0);
  }, 0);
  return high / total;
}

function storyCard({ kicker, valueText, title, body, tone = "neutral", detail }) {
  return `
    <article class="result-card result-${escapeHtml(tone)}">
      <div class="result-kicker">${escapeHtml(kicker)}</div>
      <div class="result-value">${escapeHtml(valueText)}</div>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body)}</p>
      ${detail ? `<div class="result-detail">${detail}</div>` : ""}
    </article>`;
}

function renderResultStory(targetId, cards) {
  const target = byId(targetId);
  if (!target) return;
  const cleanCards = (cards || []).filter(Boolean);
  if (!cleanCards.length) {
    target.innerHTML = "";
    return;
  }
  target.innerHTML = cleanCards.map(storyCard).join("");
}

// ---------------------------------------------------------------------------
// Generic table renderer
// ---------------------------------------------------------------------------

function renderSimpleTable(rows, columns, options = {}) {
  if (!rows || rows.length === 0) return `<div class="detail-body empty">No rows</div>`;
  const maxAbs = Math.max(...rows.map((row) => Math.abs(Number(row[options.barColumn] || 0))), 0);
  const head = columns
    .map((c) => `<th class="${c.num ? "num" : ""}"${c.help ? ` title="${escapeHtml(c.help)}"` : ""}>${escapeHtml(c.label)}</th>`)
    .join("");
  const body = rows
    .map((row, rowIndex) => {
      const cells = columns
        .map((c) => {
          let content = "";
          if (c.format === "pct") content = pct(row[c.key]);
          else if (c.format === "num") content = number(row[c.key], c.digits ?? 4);
          else if (c.format === "label") content = labelPill(row[c.key]);
          else if (c.format === "relation") content = displayRelation(row[c.key]);
          else if (c.format === "path") content = displayPath(row[c.key]);
          else if (c.format === "list") content = displayList(row[c.key], c.limit || 4);
          else if (typeof row[c.key] === "boolean") content = `<span class="pill ${row[c.key] ? "good" : "bad"}">${row[c.key] ? "Available" : "Missing"}</span>`;
          else content = escapeHtml(row[c.key]);
          if (c.key === options.barColumn) {
            const cls = Number(row[c.key]) < 0 ? "neg" : "pos";
            content = `<div class="bar-cell"><div>${content || number(row[c.key], 4)}</div>${bar(row[c.key], maxAbs, cls)}</div>`;
          }
          return `<td class="${c.num ? "num" : ""}">${content}</td>`;
        })
        .join("");
      let click = "";
      if (options.clickable) click = ` class="clickable" data-case-index="${escapeHtml(row.case_index)}"`;
      else if (options.clickableCommunity) click = ` class="clickable community-row" data-community-id="${escapeHtml(row.community_id)}"`;
      else if (options.clickableEvidence) click = ` class="clickable evidence-row" data-row-index="${rowIndex}"`;
      return `<tr${click}>${cells}</tr>`;
    })
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// ---------------------------------------------------------------------------
// Findings list renderer
// ---------------------------------------------------------------------------

function renderFindings(targetId, findings) {
  const ul = byId(targetId);
  if (!ul) return;
  if (!findings || findings.length === 0) {
    ul.innerHTML = `<li class="finding-card finding-empty">No findings could be auto-computed for this run yet.</li>`;
    return;
  }
  ul.innerHTML = findings
    .map((f, i) => {
      const text = String(f);
      const lower = text.toLowerCase();
      const tone = lower.includes("wrong") || lower.includes("alarm") || lower.includes("leak")
        ? "warn"
        : lower.includes("low") || lower.includes("weak")
          ? "caution"
          : "neutral";
      const label = lower.includes("attention")
        ? "Attention check"
        : lower.includes("wrong") || lower.includes("failure")
          ? "Audit target"
          : lower.includes("identity") || lower.includes("judge") || lower.includes("court")
            ? "Leakage signal"
            : lower.includes("accuracy") || lower.includes("confidence")
              ? "Model behavior"
              : lower.includes("nmi") || lower.includes("embedding") || lower.includes("cluster")
                ? "Representation"
                : "Finding";
      return `
        <li class="finding-card finding-${tone}">
          <span class="finding-index">${String(i + 1).padStart(2, "0")}</span>
          <div>
            <div class="finding-label">${escapeHtml(label)}</div>
            <p>${renderInlineMarkup(text)}</p>
          </div>
        </li>`;
    })
    .join("");
}

// ---------------------------------------------------------------------------
// SVG line / bar charts
// ---------------------------------------------------------------------------

function renderLineChart(rows, metricKey, yLabel, opts = {}) {
  if (!rows || rows.length === 0) return `<div class="detail-body empty">No curve data.</div>`;
  const width = 760, height = 300;
  const margin = { top: 24, right: 24, bottom: 48, left: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const colors = { counterfactual: "#176b87", attention: "#9a5b13", random: "#657383" };
  const byRanker = new Map();
  rows.forEach((row) => {
    const ranker = String(row.ranker);
    const x = Number(row.k_requested);
    const y = Number(row[metricKey]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    if (!byRanker.has(ranker)) byRanker.set(ranker, []);
    byRanker.get(ranker).push({ x, y });
  });
  const allPoints = [...byRanker.values()].flat();
  if (!allPoints.length) return `<div class="detail-body empty">No curve data.</div>`;
  const maxX = Math.max(...allPoints.map((p) => p.x), 1);
  const maxY = Math.max(opts.minTopY || 1, ...allPoints.map((p) => p.y));
  const minY = Math.min(0, ...allPoints.map((p) => p.y));
  const xS = (x) => margin.left + (x / maxX) * plotW;
  const yS = (y) => margin.top + (1 - ((y - minY) / (maxY - minY || 1))) * plotH;
  const ticks = [0, Math.ceil(maxX / 4), Math.ceil(maxX / 2), Math.ceil((3 * maxX) / 4), maxX].filter((t, i, a) => a.indexOf(t) === i);
  const yTicks = [minY, (minY + maxY) / 2, maxY];
  const lines = [...byRanker.entries()].map(([r, pts]) => {
    pts.sort((a, b) => a.x - b.x);
    const d = pts.map((p) => `${xS(p.x)},${yS(p.y)}`).join(" ");
    return `<polyline class="chart-line" points="${d}" stroke="${colors[r] || "#333"}"></polyline>`;
  }).join("");
  const dots = [...byRanker.entries()].map(([r, pts]) => pts.map((p) => `<circle cx="${xS(p.x)}" cy="${yS(p.y)}" r="3" fill="${colors[r] || "#333"}"></circle>`).join("")).join("");
  const legend = [...byRanker.keys()].map((r) => `<span class="legend-item"><span style="background:${colors[r] || "#333"}"></span>${escapeHtml(r)}</span>`).join("");
  return `
    <div class="chart-inner">
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(yLabel)}">
        <line class="axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"></line>
        <line class="axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"></line>
        ${ticks.map((t) => `
          <line class="grid-line" x1="${xS(t)}" y1="${margin.top}" x2="${xS(t)}" y2="${height - margin.bottom}"></line>
          <text class="tick" x="${xS(t)}" y="${height - margin.bottom + 20}" text-anchor="middle">${t}</text>`).join("")}
        ${yTicks.map((t) => `
          <line class="grid-line" x1="${margin.left}" y1="${yS(t)}" x2="${width - margin.right}" y2="${yS(t)}"></line>
          <text class="tick" x="${margin.left - 10}" y="${yS(t) + 4}" text-anchor="end">${number(t, 2)}</text>`).join("")}
        ${lines}${dots}
        <text class="axis-label" x="${margin.left + plotW / 2}" y="${height - 10}" text-anchor="middle">top-k evidence groups</text>
        <text class="axis-label" transform="translate(16 ${margin.top + plotH / 2}) rotate(-90)" text-anchor="middle">${escapeHtml(yLabel)}</text>
      </svg>
      <div class="legend">${legend}</div>
    </div>`;
}

function renderBarChart(rows, xKey, yKey, opts = {}) {
  if (!rows || rows.length === 0) return `<div class="detail-body empty">No data.</div>`;
  const width = 760, height = 240;
  const margin = { top: 18, right: 18, bottom: 44, left: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const maxY = Math.max(1, ...rows.map((r) => Number(r[yKey]) || 0));
  const barW = plotW / rows.length * 0.78;
  const bars = rows.map((r, i) => {
    const v = Number(r[yKey]) || 0;
    const x = margin.left + (i + 0.11) * (plotW / rows.length);
    const h = (v / maxY) * plotH;
    const y = margin.top + plotH - h;
    return `<rect class="bar-rect" x="${x}" y="${y}" width="${barW}" height="${h}"></rect>
            <text class="tick" x="${x + barW / 2}" y="${margin.top + plotH + 16}" text-anchor="middle">${escapeHtml(r[xKey])}</text>`;
  }).join("");
  return `
    <div class="chart-inner">
      <svg viewBox="0 0 ${width} ${height}" role="img">
        <line class="axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"></line>
        <line class="axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"></line>
        <text class="tick" x="${margin.left - 10}" y="${margin.top + 6}" text-anchor="end">${number(maxY, 0)}</text>
        <text class="tick" x="${margin.left - 10}" y="${margin.top + plotH}" text-anchor="end">0</text>
        ${bars}
        <text class="axis-label" x="${margin.left + plotW / 2}" y="${height - 8}" text-anchor="middle">${escapeHtml(opts.xLabel || xKey)}</text>
      </svg>
    </div>`;
}

// ---------------------------------------------------------------------------
// Network-style explanation graphs
// ---------------------------------------------------------------------------

const GRAPH_PALETTE = ["#176b87", "#2f855a", "#9a6a19", "#6b5ca5", "#b83232", "#52616f", "#148c8c", "#875f2a"];

function graphLabel(text, max = 38) {
  return compactText(String(value(text, "")).replace(/\s+/g, " ").trim(), max);
}

function featureColor(type) {
  const t = String(type || "").toLowerCase();
  if (t === "relation_type") return "#52616f";
  if (["judge", "court", "petitioner", "respondent", "lawyer", "defence_lawyer", "petitioner_lawyer"].includes(t)) return "#b36b13";
  if (["provision", "statute", "act", "section"].includes(t)) return "#176b87";
  if (["precedent", "case"].includes(t)) return "#2f855a";
  if (t.includes("argument")) return "#6b5ca5";
  return "#52616f";
}

function featureFill(type) {
  const t = String(type || "").toLowerCase();
  if (t === "relation_type") return "#f7fafc";
  if (["judge", "court", "petitioner", "respondent", "lawyer", "defence_lawyer", "petitioner_lawyer"].includes(t)) return "#fff7e6";
  if (["provision", "statute", "act", "section"].includes(t)) return "#edf8fb";
  if (["precedent", "case"].includes(t)) return "#eefaf3";
  if (t.includes("argument")) return "#f4f0ff";
  return "#ffffff";
}

function relationColor(row) {
  const text = `${row?.relation_types || ""} ${row?.path_family || ""} ${row?.evidence_type || ""}`.toLowerCase();
  if (text.includes("precedent") || text.includes("cites")) return "#2f855a";
  if (text.includes("provision") || text.includes("statute") || text.includes("section")) return "#176b87";
  if (text.includes("judge") || text.includes("court") || text.includes("lawyer") || text.includes("petitioner") || text.includes("respondent")) return "#b36b13";
  if (text.includes("argument")) return "#6b5ca5";
  return "#52616f";
}

function isIdentityNode(row) {
  const type = String(row?.feature_type || row?.evidence_type || "").toLowerCase();
  return ["judge", "court", "petitioner", "respondent", "lawyer", "defence_lawyer", "petitioner_lawyer"].includes(type);
}

function svgText(text, x, y, opts = {}) {
  const anchor = opts.anchor || "middle";
  const cls = opts.cls || "graph-label";
  const max = opts.max || 32;
  return `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${escapeHtml(graphLabel(text, max))}</text>`;
}

function svgFeatureCard(item, x, y, w = 210, h = 54, opts = {}) {
  const type = item.feature_type || item.evidence_type || "evidence";
  const color = featureColor(type);
  const fill = featureFill(type);
  const identity = isIdentityNode(item);
  const title = `${type}: ${item.feature_name || item.evidence_name || ""}`;
  const labelMax = opts.metaText ? Math.min(opts.max || 30, 27) : (opts.max || 30);
  return `
    <g class="feature-card ${identity ? "identity" : ""}">
      <title>${escapeHtml(title)}</title>
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${color}" stroke-width="${identity ? 2.8 : 1.4}"></rect>
      <text class="graph-tag" x="${x + 12}" y="${y + 18}" fill="${color}">${escapeHtml(graphLabel(type, 20))}</text>
      ${opts.metaText ? `<text class="graph-micro card-meta" x="${x + w - 10}" y="${y + 18}" text-anchor="end">${escapeHtml(opts.metaText)}</text>` : ""}
      ${!opts.metaText && item.idf !== undefined && item.idf !== null ? `<text class="graph-micro card-meta" x="${x + w - 10}" y="${y + 18}" text-anchor="end">idf ${number(item.idf, 1)}</text>` : ""}
      <text class="graph-label strong" x="${x + 12}" y="${y + 38}" text-anchor="start">${escapeHtml(graphLabel(item.feature_name || item.evidence_name || "unnamed evidence", labelMax))}</text>
    </g>`;
}

function renderCommunityEmbeddingFlow(flow) {
  if (!flow || !flow.available || !(flow.links || []).length) {
    return `<div class="detail-body empty">Community-to-embedding flow data is not available.</div>`;
  }
  const sources = flow.source_totals || [];
  const targets = flow.target_totals || [];
  const links = flow.links || [];
  const width = 1120;
  const height = Math.max(460, 110 + Math.max(sources.length, targets.length) * 43);
  const top = 66;
  const bottom = height - 54;
  const yScale = (rows, i) => rows.length === 1 ? (top + bottom) / 2 : top + (i / (rows.length - 1)) * (bottom - top);
  const sourceY = new Map(sources.map((row, i) => [row.id, yScale(sources, i)]));
  const targetY = new Map(targets.map((row, i) => [row.id, yScale(targets, i)]));
  const targetColor = new Map(targets.map((row, i) => [row.id, GRAPH_PALETTE[i % GRAPH_PALETTE.length]]));
  const maxValue = Math.max(...links.map((row) => Number(row.value) || 0), 1);
  const sourceX = 70, sourceW = 220, targetX = 830, targetW = 230;
  const paths = links
    .slice()
    .sort((a, b) => (Number(b.value) || 0) - (Number(a.value) || 0))
    .map((row) => {
      const y1 = sourceY.get(row.source) || top;
      const y2 = targetY.get(row.target) || top;
      const stroke = targetColor.get(row.target) || "#176b87";
      const sw = Math.max(2, 2 + Math.sqrt((Number(row.value) || 0) / maxValue) * 26);
      return `<path class="flow-link" d="M ${sourceX + sourceW} ${y1} C 455 ${y1}, 555 ${y2}, ${targetX} ${y2}" stroke="${stroke}" stroke-width="${sw}">
        <title>${escapeHtml(row.source_label)} -> ${escapeHtml(row.target_label)}: ${number(row.value, 0)} cases</title>
      </path>`;
    }).join("");
  const sourceNodes = sources.map((row) => {
    const y = sourceY.get(row.id);
    return `<g class="flow-node">
      <rect x="${sourceX}" y="${y - 16}" width="${sourceW}" height="32" rx="7"></rect>
      ${svgText(row.label, sourceX + 12, y - 1, { anchor: "start", max: 24, cls: "graph-label strong" })}
      <text class="graph-micro" x="${sourceX + sourceW - 10}" y="${y + 10}" text-anchor="end">${number(row.value, 0)}</text>
    </g>`;
  }).join("");
  const targetNodes = targets.map((row) => {
    const y = targetY.get(row.id);
    const color = targetColor.get(row.id) || "#176b87";
    return `<g class="flow-node target">
      <rect x="${targetX}" y="${y - 16}" width="${targetW}" height="32" rx="7" stroke="${color}"></rect>
      ${svgText(row.label, targetX + 12, y - 1, { anchor: "start", max: 25, cls: "graph-label strong" })}
      <text class="graph-micro" x="${targetX + targetW - 10}" y="${y + 10}" text-anchor="end">${number(row.value, 0)}</text>
    </g>`;
  }).join("");
  return `
    <div class="graph-summary">
      <span><strong>${number(flow.shown_communities, 0)}</strong> largest communities shown</span>
      <span><strong>${number(flow.linked_cases, 0)}</strong> visible case assignments</span>
      <span><strong>${number(flow.total_cases, 0)}</strong> total cases in split table</span>
    </div>
    <div class="graph-frame graph-flow-frame">
      <svg class="graph-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Community to embedding cluster flow">
        <text class="graph-axis-title" x="${sourceX}" y="34">Leiden structural communities</text>
        <text class="graph-axis-title" x="${targetX}" y="34">HDBSCAN embedding clusters</text>
        ${paths}
        ${sourceNodes}
        ${targetNodes}
      </svg>
    </div>`;
}

function renderCaseContrastGraph(graph) {
  if (!graph || !graph.available) {
    return `<div class="detail-body empty">${escapeHtml(graph?.reason || "Select an opposite-case pair to render the contrast graph.")}</div>`;
  }
  const summary = graph.summary || {};
  const shared = graph.shared_features || [];
  const queryOnly = graph.query_only_features || [];
  const oppositeOnly = graph.opposite_only_features || [];
  const maxRows = Math.max(shared.length, queryOnly.length, oppositeOnly.length, 4);
  const width = 1180;
  const height = Math.max(560, 160 + maxRows * 68);
  const centerY = height / 2;
  const stackY = (rows, i) => {
    const blockH = Math.max(rows.length, 1) * 62;
    return centerY - blockH / 2 + i * 62;
  };
  const qx = 112, ox = 1068;
  const qOnlyX = 245, sharedX = 486, oppOnlyX = 727;
  const cardW = 205, cardH = 52;
  const edges = [];
  queryOnly.forEach((item, i) => edges.push(`<path class="contrast-edge" d="M ${qx + 52} ${centerY} C 185 ${centerY}, 205 ${stackY(queryOnly, i) + 26}, ${qOnlyX} ${stackY(queryOnly, i) + 26}" stroke="${featureColor(item.feature_type)}"></path>`));
  shared.forEach((item, i) => {
    const y = stackY(shared, i) + 26;
    edges.push(`<path class="contrast-edge shared" d="M ${qx + 52} ${centerY} C 260 ${centerY}, 330 ${y}, ${sharedX} ${y}" stroke="${featureColor(item.feature_type)}"></path>`);
    edges.push(`<path class="contrast-edge shared" d="M ${sharedX + cardW} ${y} C 790 ${y}, 900 ${centerY}, ${ox - 52} ${centerY}" stroke="${featureColor(item.feature_type)}"></path>`);
  });
  oppositeOnly.forEach((item, i) => edges.push(`<path class="contrast-edge" d="M ${oppOnlyX + cardW} ${stackY(oppositeOnly, i) + 26} C 950 ${stackY(oppositeOnly, i) + 26}, 980 ${centerY}, ${ox - 52} ${centerY}" stroke="${featureColor(item.feature_type)}"></path>`));
  const queryCards = queryOnly.map((item, i) => svgFeatureCard(item, qOnlyX, stackY(queryOnly, i), cardW, cardH, { max: 27 })).join("");
  const sharedCards = shared.map((item, i) => svgFeatureCard(item, sharedX, stackY(shared, i), cardW, cardH, { max: 27 })).join("");
  const oppositeCards = oppositeOnly.map((item, i) => svgFeatureCard(item, oppOnlyX, stackY(oppositeOnly, i), cardW, cardH, { max: 27 })).join("");
  return `
    <div class="graph-summary">
      <span><strong>Query ${escapeHtml(summary.case_index)}</strong> label ${escapeHtml(summary.target_label)}</span>
      <span><strong>Opposite ${escapeHtml(summary.nearest_opposite_case_index)}</strong> label ${escapeHtml(summary.nearest_opposite_target_label)}</span>
      <span><strong>${number(summary.cosine_similarity, 4)}</strong> cosine similarity</span>
    </div>
    <div class="graph-frame graph-contrast-frame">
      <svg class="graph-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Query case and nearest opposite-label case contrast">
        <text class="graph-axis-title" x="${qOnlyX}" y="38">Query-only evidence</text>
        <text class="graph-axis-title" x="${sharedX}" y="38">Shared evidence</text>
        <text class="graph-axis-title" x="${oppOnlyX}" y="38">Opposite-only evidence</text>
        ${edges.join("")}
        <g class="case-node">
          <circle cx="${qx}" cy="${centerY}" r="52"></circle>
          ${svgText(`Case ${summary.case_index}`, qx, centerY - 8, { max: 18, cls: "graph-label strong" })}
          ${svgText(`target ${summary.target_label}`, qx, centerY + 14, { max: 18, cls: "graph-micro" })}
        </g>
        <g class="case-node opposite">
          <circle cx="${ox}" cy="${centerY}" r="52"></circle>
          ${svgText(`Case ${summary.nearest_opposite_case_index}`, ox, centerY - 8, { max: 18, cls: "graph-label strong" })}
          ${svgText(`target ${summary.nearest_opposite_target_label}`, ox, centerY + 14, { max: 18, cls: "graph-micro" })}
        </g>
        ${queryCards || svgText("No query-only rows", qOnlyX + cardW / 2, centerY, { max: 26, cls: "graph-empty-label" })}
        ${sharedCards || svgText("No shared feature rows", sharedX + cardW / 2, centerY, { max: 26, cls: "graph-empty-label" })}
        ${oppositeCards || svgText("No opposite-only rows", oppOnlyX + cardW / 2, centerY, { max: 26, cls: "graph-empty-label" })}
      </svg>
    </div>
    <div class="graph-actions">
      <button data-open-case="${escapeHtml(summary.case_index)}">Open query in Case Explorer</button>
      ${String(summary.nearest_opposite_split || "") === "test" ? `<button data-open-case="${escapeHtml(summary.nearest_opposite_case_index)}">Open opposite case</button>` : ""}
    </div>`;
}

function renderLocalExplanationSubgraph(summary, localGraph) {
  const graph = Array.isArray(localGraph)
    ? { available: true, summary: {}, paths: [], evidence: localGraph, connected_cases: [] }
    : (localGraph || {});
  const evidence = graph.evidence || [];
  if (!graph.available || !evidence.length) {
    return `<div class="case-graph-block detail-body empty">No local graph rows are available for this case.</div>`;
  }

  const paths = graph.paths || [];
  const connected = graph.connected_cases || [];
  const stats = graph.summary || {};
  const rows = Math.max(paths.length, evidence.length, connected.length, 5);
  const width = 1220;
  const height = Math.max(600, 142 + rows * 56);
  const top = 88;
  const bottom = height - 78;
  const yScale = (list, i) => list.length <= 1 ? (top + bottom) / 2 : top + (i / (list.length - 1)) * (bottom - top);
  const pathY = new Map(paths.map((row, i) => [row.id, yScale(paths, i)]));
  const evidenceY = new Map(evidence.map((row, i) => [row.id, yScale(evidence, i)]));
  const connectedY = new Map(connected.map((row, i) => [`case:${row.case_index}`, yScale(connected, i)]));
  const caseX = 100, pathX = 230, evidenceX = 520, connectedX = 865;
  const pathW = 220, evidenceW = 255, connectedW = 235;
  const centerY = (top + bottom) / 2;
  const maxImportance = Math.max(...evidence.map((row) => Number(row.importance || row.abs_delta_pred_proba) || 0), 1e-9);

  const caseToPathEdges = paths.map((row) => {
    const y = pathY.get(row.id);
    const sw = 1.6 + Math.sqrt((Number(row.importance) || 0) / maxImportance) * 5;
    return `<path class="local-edge" d="M ${caseX + 54} ${centerY} C 165 ${centerY}, 185 ${y}, ${pathX} ${y}" stroke="${relationColor(row)}" stroke-width="${sw}"></path>`;
  }).join("");
  const pathToEvidenceEdges = evidence.map((row) => {
    const y1 = pathY.get(row.path_id) ?? centerY;
    const y2 = evidenceY.get(row.id) ?? centerY;
    const sw = 1.4 + Math.sqrt((Number(row.importance || row.abs_delta_pred_proba) || 0) / maxImportance) * 5;
    return `<path class="local-edge evidence-edge" d="M ${pathX + pathW} ${y1} C 440 ${y1}, 460 ${y2}, ${evidenceX} ${y2}" stroke="${relationColor(row)}" stroke-width="${sw}">
      <title>${escapeHtml(row.path_family || row.relation_types || row.evidence_type)}</title>
    </path>`;
  }).join("");
  const connectedEdges = connected.map((row) => {
    const y = connectedY.get(`case:${row.case_index}`) ?? centerY;
    const sw = Math.min(7, 1.3 + Math.sqrt(Number(row.shared_feature_count) || 0));
    return `<path class="local-edge connected-edge" d="M ${caseX + 54} ${centerY} C 430 ${centerY}, 600 ${y}, ${connectedX} ${y}" stroke="#657383" stroke-width="${sw}"></path>`;
  }).join("");

  const pathCards = paths.map((row) => {
    const y = pathY.get(row.id) - 23;
    return `
      <g class="path-card">
        <rect x="${pathX}" y="${y}" width="${pathW}" height="46" rx="8" fill="#eef6fb" stroke="${relationColor(row)}"></rect>
        <text class="graph-label strong" x="${pathX + 11}" y="${y + 18}" text-anchor="start">${escapeHtml(graphLabel(row.label, 27))}</text>
        <text class="graph-micro" x="${pathX + 11}" y="${y + 35}" text-anchor="start">${number(row.group_count, 0)} groups | max |Δp| ${number(row.importance, 3)}</text>
      </g>`;
  }).join("");
  const evidenceCards = evidence.map((row) => {
    const y = evidenceY.get(row.id) - 24;
    return svgFeatureCard({
      evidence_type: row.evidence_type,
      evidence_name: row.label || row.evidence_name,
    }, evidenceX, y, evidenceW, 48, {
      max: 34,
      metaText: row.support_train_n !== undefined && row.support_train_n !== null ? `support ${number(row.support_train_n, 0)}` : "",
    });
  }).join("");
  const connectedCards = connected.map((row) => {
    const y = connectedY.get(`case:${row.case_index}`) - 24;
    const tone = String(row.target_label) === String(summary.target_label) ? "#2f855a" : "#b36b13";
    return `
      <g class="connected-case-card">
        <title>${escapeHtml(row.case_id || `Case ${row.case_index}`)}</title>
        <rect x="${connectedX}" y="${y}" width="${connectedW}" height="48" rx="8" fill="${tone === "#2f855a" ? "#effaf3" : "#fff7e8"}" stroke="${tone}"></rect>
        <text class="graph-label strong" x="${connectedX + 11}" y="${y + 18}" text-anchor="start">Case ${escapeHtml(row.case_index)}</text>
        <text class="graph-micro" x="${connectedX + 11}" y="${y + 35}" text-anchor="start">${number(row.shared_feature_count, 0)} shared features | label ${escapeHtml(row.target_label ?? "")}</text>
      </g>`;
  }).join("");
  const connectedButtons = connected.filter((row) => String(row.split || "") === "test").slice(0, 10).map((row) => (
    `<button data-open-case="${escapeHtml(row.case_index)}">Case ${escapeHtml(row.case_index)} (${number(row.shared_feature_count, 0)})</button>`
  )).join("");

  return `
    <div class="case-graph-block">
      <div class="case-graph-head">
        <h4>Local Explanation Subgraph</h4>
        <p>Shows the selected case pointing into its counterfactual path families, evidence groups, and feature-overlap neighbors. Counts include all generated groups; the figure shows the strongest nodes to keep it readable.</p>
      </div>
      <div class="graph-summary">
        <span><strong>${number(stats.total_groups, 0)}</strong> counterfactual groups</span>
        <span><strong>${number(stats.total_paths, 0)}</strong> path families</span>
        <span><strong>${number(stats.connected_case_count, 0)}</strong> feature-connected cases</span>
        <span><strong>${number(stats.shown_evidence, 0)}</strong> evidence nodes shown</span>
      </div>
      <div class="graph-frame local-graph-frame">
        <svg class="graph-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Local case explanation graph with paths and connected cases">
          <text class="graph-axis-title" x="${pathX}" y="42">Connected path families</text>
          <text class="graph-axis-title" x="${evidenceX}" y="42">Evidence groups</text>
          <text class="graph-axis-title" x="${connectedX}" y="42">Other connected cases</text>
          ${connectedEdges}
          ${caseToPathEdges}
          ${pathToEvidenceEdges}
          <g class="case-node center">
            <circle cx="${caseX}" cy="${centerY}" r="54"></circle>
            ${svgText(`Case ${summary.case_index}`, caseX, centerY - 9, { max: 20, cls: "graph-label strong" })}
            ${svgText(`pred ${summary.baseline_pred_label}`, caseX, centerY + 13, { max: 20, cls: "graph-micro" })}
          </g>
          ${pathCards}
          ${evidenceCards}
          ${connectedCards || svgText("No feature-overlap cases", connectedX + connectedW / 2, centerY, { cls: "graph-empty-label", max: 28 })}
        </svg>
      </div>
      ${connectedButtons ? `<div class="graph-actions connected-actions">${connectedButtons}</div>` : ""}
    </div>`;
}

// ---------------------------------------------------------------------------
// OVERVIEW
// ---------------------------------------------------------------------------

async function loadOverview() {
  const data = await api("/api/exp_overview");
  const h = data.headline || {};
  byId("outputDir").textContent = `${data.output_dir || ""} | pattern: ${data.pattern_dir || ""} | full graph: ${data.status?.full_graph_dir || ""}`;

  const liftText = h.comp_lift_vs_random ? `${h.comp_lift_vs_random.toFixed(1)}× over random` : "";
  byId("overviewHeadline").innerHTML = [
    metric("Cases Explained", number(h.n_cases, 0), "Cases with counterfactual analysis."),
    metric("Mask Groups Run", number(h.n_groups, 0), "Total counterfactual mask passes."),
    metric("Model Accuracy", pct(h.accuracy), "Saved-prediction accuracy on this split."),
    metric("Cases with Flips", pct(h.flip_rate), "Share of cases where some mask flips the predicted class."),
    metric("Counterfactual AUC", number(h.cf_comprehensiveness_auc, 3), liftText || "Mean comprehensiveness AUC.", "good"),
    metric("Attention Overlap", pct(h.attention_overlap), "Mean top-k agreement with the counterfactual ranking."),
    metric("Communities", number(h.n_communities, 0), "Leiden communities in the case-case projection."),
    metric("Embedding NMI", number(h.embedding_nmi, 3), "Embedding↔structural alignment."),
    metric("HDBSCAN Noise", pct(h.embedding_noise_rate), "Share of cases not assigned to any embedding cluster."),
    metric("Identity Share", pct(h.identity_evidence_share), "Mean importance share carried by judges/courts/parties.", "warn"),
    metric("Identity Shortcut", number(h.identity_shortcut_auc, 3), `${h.identity_shortcut_scope || "Top identity"} identity-only AUC.`, "warn"),
  ].join("");

  renderResultStory("overviewResultStory", [
    {
      kicker: "Where to start",
      valueText: pct(h.accuracy) || "—",
      title: "First check whether the model predictions are credible.",
      body: "Then move to counterfactual masking to see what actually changes those predictions.",
      tone: "neutral",
    },
    {
      kicker: "Main evidence claim",
      valueText: h.comp_lift_vs_random ? `${h.comp_lift_vs_random.toFixed(1)}x` : "—",
      title: "Counterfactual evidence beats random removal.",
      body: "Experiment 3 is the core faithfulness validation for the explanation method.",
      tone: "good",
    },
    {
      kicker: "Main representation claim",
      valueText: number(h.embedding_nmi, 3) || "—",
      title: "Embedding geometry is weakly aligned with structural communities.",
      body: "Read Experiments 4 and 5 together to separate corpus topology from learned outcome geometry.",
      tone: "good",
    },
    {
      kicker: "Main caveat",
      valueText: number(h.identity_shortcut_auc, 3) || pct(h.identity_evidence_share) || "—",
      title: "Identity evidence needs to be reported, not hidden.",
      body: "Read the identity shortcut audit after counterfactual masking to distinguish shortcut correlation from direct model reliance.",
      tone: "warn",
    },
  ]);

  byId("experimentMap").innerHTML = [
    expCard("1", "HGT Embeddings", "What did the trained model encode for each case?"),
    expCard("2", "Counterfactual Masking", "Which evidence causally drives each prediction?"),
    expCard("10", "Identity Shortcut Audit", "Can judge, party, court, or lawyer names predict held-out labels by themselves?"),
    expCard("3", "Faithfulness", "Is the counterfactual ranking actually causal vs attention/random?"),
    expCard("4", "Legal Communities", "What corpus-grounded legal patterns recur?"),
    expCard("5", "Embedding Clusters", "Did the model encode topology, or something more?"),
    expCard("6", "Opposite Cases", "What changes between this case and its closest opposite-label twin?"),
  ].join("");

  // Run status table
  const status = data.status || {};
  const fileRows = [];
  Object.entries(status.files || {}).forEach(([k, v]) => fileRows.push({ kind: "Counterfactual", file: k, exists: v.exists, rows: v.rows, bytes: v.bytes }));
  Object.entries(status.pattern_files || {}).forEach(([k, v]) => fileRows.push({ kind: "Pattern", file: k, exists: v.exists, rows: v.rows, bytes: v.bytes }));
  Object.entries(status.full_graph_files || {}).forEach(([k, v]) => fileRows.push({ kind: "Full graph", file: k, exists: v.exists, rows: v.rows, bytes: v.bytes }));
  byId("overviewRunBody").innerHTML = renderSimpleTable(fileRows, [
    { key: "kind", label: "Group" },
    { key: "file", label: "Table" },
    { key: "exists", label: "Available" },
    { key: "rows", label: "Rows", num: true, format: "num", digits: 0 },
    { key: "bytes", label: "Size (bytes)", num: true, format: "num", digits: 0 },
  ]);
}

function expCard(num, title, blurb) {
  return `<a class="experiment-card" href="#exp${num}" data-jump="exp${num}">
    <div class="experiment-card-num">${escapeHtml(num)}</div>
    <h4>${escapeHtml(title)}</h4>
    <p>${escapeHtml(blurb)}</p>
  </a>`;
}

// ---------------------------------------------------------------------------
// EXP 1: Embeddings
// ---------------------------------------------------------------------------

async function loadExp1() {
  const data = await api("/api/exp/embeddings");
  renderFindings("exp1Findings", data.findings);

  const m = data.manifest || {};
  const testSplit = (data.accuracy_by_split || [])[0] || {};
  const align = data.alignment_row || {};
  renderResultStory("exp1ResultStory", [
    {
      kicker: "Prediction readout",
      valueText: pct(testSplit.accuracy) || "—",
      title: "The saved test predictions are usable.",
      body: `${number(testSplit.n_cases, 0) || "—"} test cases, mean confidence ${number(testSplit.mean_confidence, 2) || "—"}.`,
      tone: "good",
    },
    {
      kicker: "Confidence shape",
      valueText: pct(confidenceHighShare(data.confidence_histogram)) || "—",
      title: "Most cases are decided, not borderline.",
      body: "Share of cases with predicted-class probability at or above 0.80.",
      tone: "neutral",
    },
    {
      kicker: "Embedding geometry",
      valueText: number(align.normalized_mutual_info_all, 3) || "—",
      title: "Weak alignment with structural communities.",
      body: "Low NMI means the learned representation is not merely recreating shared-authority topology.",
      tone: Number(align.normalized_mutual_info_all) < 0.1 ? "good" : "neutral",
    },
    {
      kicker: "Artifact scale",
      valueText: `${number(m.n_cases, 0) || "—"} x ${m.embedding_dim ?? "—"}`,
      title: "Embedding matrix ready for downstream audits.",
      body: `${number(data.n_clusters, 0)} dense HDBSCAN clusters are available for Experiment 5.`,
      tone: "neutral",
    },
  ]);

  byId("exp1Manifest").innerHTML = `
    <div class="kv-grid">
      <div class="kv"><span>Embedding dimension</span><strong>${escapeHtml(m.embedding_dim ?? "—")}</strong></div>
      <div class="kv"><span>Cases embedded</span><strong>${number(m.n_cases, 0) || "—"}</strong></div>
      <div class="kv"><span>Output classes</span><strong>${escapeHtml(m.n_classes ?? "—")}</strong></div>
      <div class="kv"><span>Embedding clusters</span><strong>${number(data.n_clusters, 0)}</strong></div>
      <div class="kv kv-wide"><span>Model checkpoint</span><code>${escapeHtml(m.model_path || "—")}</code></div>
      <div class="kv kv-wide"><span>Config</span><code>${escapeHtml(m.config_path || "—")}</code></div>
    </div>`;

  byId("exp1Splits").innerHTML = renderSimpleTable(data.accuracy_by_split, [
    { key: "split", label: "Split" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "accuracy", label: "Accuracy", num: true, format: "pct" },
    { key: "mean_confidence", label: "Mean confidence", num: true, format: "num" },
  ], { barColumn: "n_cases" });

  byId("exp1ConfidenceHist").innerHTML = renderBarChart(data.confidence_histogram, "bucket", "n_cases", { xLabel: "predicted-class probability bucket" });
}

// ---------------------------------------------------------------------------
// EXP 2: Counterfactual
// ---------------------------------------------------------------------------

async function loadExp2() {
  const data = await api("/api/exp/counterfactual");
  renderFindings("exp2Findings", data.findings);
  const topEvidence = bestBy(data.evidence_types, (row) => row.sum_abs_delta_pred_proba);
  const topPathByFlip = bestBy(data.path_families, (row) => (Number(row.n_cases) || 0) >= 50 ? row.flip_rate : -1);
  const topLeakage = bestBy(data.leakage, (row) => row.mean_abs_delta_pred_proba);
  const overlap = data.attention_overlap_summary || {};
  renderResultStory("exp2ResultStory", [
    {
      kicker: "Largest causal bucket",
      valueText: topEvidence.evidence_type || "—",
      title: `${number(topEvidence.sum_abs_delta_pred_proba, 0) || "—"} cumulative |Δp|`,
      body: `${number(topEvidence.n_cases, 0) || "—"} cases touched this evidence type; flip rate ${pct(topEvidence.flip_rate) || "—"}.`,
      tone: "neutral",
    },
    {
      kicker: "Most fragile legal path",
      valueText: pct(topPathByFlip.flip_rate) || "—",
      title: compactText(topPathByFlip.path_family || "No path found", 80),
      body: "Among supported paths, this one most often changes the predicted class when masked.",
      tone: Number(topPathByFlip.flip_rate) > 0.05 ? "warn" : "neutral",
    },
    {
      kicker: "Identity leakage watch",
      valueText: topLeakage.evidence_type || "—",
      title: `${number(topLeakage.mean_abs_delta_pred_proba, 3) || "—"} mean |Δp|`,
      body: `P95 ${number(topLeakage.p95_abs_delta_pred_proba, 3) || "—"}; flip rate ${pct(topLeakage.flip_rate) || "—"}. Treat high identity importance as an audit flag.`,
      tone: "warn",
    },
    {
      kicker: "Attention reliability",
      valueText: pct(overlap.mean) || "—",
      title: "Attention and counterfactual rankings mostly disagree.",
      body: "Use attention as a diagnostic, not as the explanation, when overlap is this low.",
      tone: Number(overlap.mean) < 0.3 ? "warn" : "neutral",
    },
  ]);

  byId("exp2Evidence").innerHTML = renderSimpleTable(data.evidence_types, [
    { key: "evidence_type", label: "Type" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "mean_abs_delta_pred_proba", label: "Mean |Δp|", num: true, format: "num", help: "Per-mask average importance for this type." },
    { key: "sum_abs_delta_pred_proba", label: "Total |Δp|", num: true, format: "num", help: "Cumulative importance across the corpus." },
    { key: "flip_rate", label: "Flip rate", num: true, format: "pct" },
  ], { barColumn: "sum_abs_delta_pred_proba" });

  byId("exp2Paths").innerHTML = renderSimpleTable(data.path_families, [
    { key: "path_family", label: "Path", format: "path" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "mean_abs_delta_pred_proba", label: "Mean |Δp|", num: true, format: "num" },
    { key: "sum_abs_delta_pred_proba", label: "Total |Δp|", num: true, format: "num" },
    { key: "flip_rate", label: "Flip rate", num: true, format: "pct" },
  ], { barColumn: "sum_abs_delta_pred_proba" });

  byId("exp2Relations").innerHTML = renderSimpleTable(data.relation_types, [
    { key: "relation_type", label: "Relation" },
    { key: "src_type", label: "Src" },
    { key: "relation", label: "Edge", format: "relation" },
    { key: "dst_type", label: "Dst" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "mean_abs_delta_pred_proba", label: "Mean |Δp|", num: true, format: "num" },
    { key: "sum_abs_delta_pred_proba", label: "Total |Δp|", num: true, format: "num" },
    { key: "flip_rate", label: "Flip rate", num: true, format: "pct" },
    { key: "mean_attention_score", label: "Mean attn", num: true, format: "num" },
  ], { barColumn: "sum_abs_delta_pred_proba" });

  byId("exp2Leakage").innerHTML = renderSimpleTable(data.leakage, [
    { key: "evidence_type", label: "Identity type" },
    { key: "n_groups", label: "Groups", num: true, format: "num", digits: 0 },
    { key: "mean_abs_delta_pred_proba", label: "Mean |Δp|", num: true, format: "num" },
    { key: "p95_abs_delta_pred_proba", label: "P95 |Δp|", num: true, format: "num" },
    { key: "max_abs_delta_pred_proba", label: "Max |Δp|", num: true, format: "num" },
    { key: "flip_rate", label: "Flip rate", num: true, format: "pct" },
  ], { barColumn: "mean_abs_delta_pred_proba" });

  const a = data.attention_overlap_summary || {};
  byId("exp2Attention").innerHTML = `
    <div class="kv-grid">
      <div class="kv"><span>Cases compared</span><strong>${number(a.n_cases, 0)}</strong></div>
      <div class="kv"><span>Mean overlap</span><strong>${pct(a.mean)}</strong></div>
      <div class="kv"><span>Median overlap</span><strong>${pct(a.median)}</strong></div>
    </div>
    <p class="panel-note-inline">
      Counterfactual is the faithful ranking. Attention is shown as a diagnostic — high overlap means the two agree for that case; low overlap means attention is unreliable as an explanation.
    </p>`;
}

// ---------------------------------------------------------------------------
// IDENTITY SHORTCUT AUDIT
// ---------------------------------------------------------------------------

async function loadExp10() {
  const data = await api("/api/exp/identity_shortcuts");
  renderFindings("exp10Findings", data.findings);
  if (!data.available) {
    ["exp10Summary", "exp10Skewed", "exp10Cases", "exp10MaskSummary", "exp10MaskDomains"].forEach((id) => {
      byId(id).innerHTML = `<div class="detail-body empty">Identity shortcut audit outputs not found.</div>`;
    });
    renderResultStory("exp10ResultStory", []);
    return;
  }

  const rows = data.summary || [];
  const topAuc = bestBy(rows.filter((row) => row.identity_scope !== "combined"), (row) => row.identity_auc_roc);
  const topFlip = bestBy(rows.filter((row) => row.identity_scope !== "combined"), (row) => row.counterfactual_flip_rate);
  const combined = rows.find((row) => row.identity_scope === "combined") || {};
  const maskRows = data.mask_summary || [];
  const allIdentityMask = maskRows.find((row) => row.mask_name === "no_all_identities") || {};
  const strongestIdentityMask = bestBy(maskRows.filter((row) => row.mask_name !== "no_all_identities"), (row) => row.accuracy_drop);
  renderResultStory("exp10ResultStory", [
    {
      kicker: "Strongest shortcut",
      valueText: topAuc.identity_scope || "—",
      title: `Identity-only AUC ${number(topAuc.identity_auc_roc, 3) || "—"}`,
      body: `Known eval-case coverage ${pct(topAuc.known_eval_case_share) || "—"}; permutation p ${number(topAuc.permutation_auc_p_value, 4) || "—"}.`,
      tone: Number(topAuc.identity_auc_roc) >= 0.65 ? "warn" : "neutral",
    },
    {
      kicker: "Strongest model reliance",
      valueText: topFlip.identity_scope || "—",
      title: `Counterfactual flip rate ${pct(topFlip.counterfactual_flip_rate) || "—"}`,
      body: `Mean counterfactual |Δp| ${number(topFlip.counterfactual_mean_abs_delta_pred_proba, 3) || "—"}. Compare with AUC to avoid overcalling leakage.`,
      tone: Number(topFlip.counterfactual_flip_rate) >= 0.05 ? "warn" : "neutral",
    },
    {
      kicker: "Combined identity coverage",
      valueText: pct(combined.known_eval_case_share) || "—",
      title: `Combined identity-only AUC ${number(combined.identity_auc_roc, 3) || "—"}`,
      body: `Domain log-loss delta ${number(combined.identity_log_loss_delta_vs_domain, 4) || "—"}; negative means better than domain-only baseline.`,
      tone: Number(combined.identity_auc_roc) >= 0.65 ? "warn" : "neutral",
    },
    {
      kicker: "Inference mask",
      valueText: pct(allIdentityMask.accuracy_drop) || "—",
      title: `All identities flip ${pct(allIdentityMask.flip_rate) || "—"} of predictions`,
      body: `Original-class confidence drop ${number(allIdentityMask.confidence_drop, 3) || "—"}; separate strongest mask is ${strongestIdentityMask.mask_name || "—"}.`,
      tone: Number(allIdentityMask.accuracy_drop) >= 0.02 ? "warn" : "neutral",
    },
    {
      kicker: "Interpretation",
      valueText: "Audit flag",
      title: "Shortcut risk is not proof of cheating.",
      body: "Use this with no-name, temporal, and identity-masking ablations to separate valid legal context from shortcut leakage.",
      tone: "neutral",
    },
  ]);

  byId("exp10MaskSummary").innerHTML = renderSimpleTable(maskRows, [
    { key: "mask_name", label: "Mask" },
    { key: "masked_edge_share", label: "Edges masked", num: true, format: "pct" },
    { key: "baseline_accuracy", label: "Base acc", num: true, format: "pct" },
    { key: "masked_accuracy", label: "Masked acc", num: true, format: "pct" },
    { key: "accuracy_drop", label: "Acc drop", num: true, format: "pct" },
    { key: "macro_f1_drop", label: "Macro-F1 drop", num: true, format: "pct" },
    { key: "confidence_drop", label: "Orig-conf drop", num: true, format: "num", help: "Drop in probability assigned to the original unmasked predicted class." },
    { key: "flip_rate", label: "Flip rate", num: true, format: "pct" },
  ], { barColumn: "accuracy_drop" });

  byId("exp10MaskDomains").innerHTML = renderSimpleTable(data.mask_domain_drops || [], [
    { key: "mask_name", label: "Mask" },
    { key: "domain_bucket", label: "Domain" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "accuracy_drop", label: "Acc drop", num: true, format: "pct" },
    { key: "macro_f1_drop", label: "Macro-F1 drop", num: true, format: "pct" },
    { key: "confidence_drop", label: "Orig-conf drop", num: true, format: "num" },
    { key: "flip_rate", label: "Flip rate", num: true, format: "pct" },
  ], { barColumn: "accuracy_drop" });

  byId("exp10Summary").innerHTML = renderSimpleTable(rows, [
    { key: "identity_scope", label: "Scope" },
    { key: "known_eval_case_share", label: "Known eval cases", num: true, format: "pct" },
    { key: "eval_identity_overlap_share", label: "Identity overlap", num: true, format: "pct" },
    { key: "identity_auc_roc", label: "Identity AUC", num: true, format: "num" },
    { key: "identity_log_loss_delta_vs_domain", label: "Log-loss Δ vs domain", num: true, format: "num" },
    { key: "identity_brier_delta_vs_domain", label: "Brier Δ vs domain", num: true, format: "num" },
    { key: "permutation_auc_p_value", label: "Permutation p", num: true, format: "num" },
    { key: "counterfactual_mean_abs_delta_pred_proba", label: "CF mean |Δp|", num: true, format: "num" },
    { key: "counterfactual_flip_rate", label: "CF flip rate", num: true, format: "pct" },
  ], { barColumn: "identity_auc_roc" });

  byId("exp10Skewed").innerHTML = renderSimpleTable(data.top_skewed_identities || [], [
    { key: "identity_type", label: "Type" },
    { key: "identity_name", label: "Identity" },
    { key: "train_support", label: "Train n", num: true, format: "num", digits: 0 },
    { key: "train_positive_rate", label: "Train label 1", num: true, format: "pct" },
    { key: "smoothed_lift_from_train_prior", label: "Lift vs prior", num: true, format: "num" },
    { key: "eval_support", label: "Eval n", num: true, format: "num", digits: 0 },
  ], { barColumn: "smoothed_lift_from_train_prior" });

  byId("exp10Cases").innerHTML = renderSimpleTable(data.case_scores || [], [
    { key: "case_index", label: "Case", num: true, format: "num", digits: 0 },
    { key: "case_id", label: "Case id" },
    { key: "target_label", label: "Target", format: "label" },
    { key: "pred_label", label: "Pred", format: "label" },
    { key: "correct", label: "Correct" },
    { key: "domain_bucket", label: "Domain" },
    { key: "identity_score_combined", label: "Identity score", num: true, format: "num" },
    { key: "domain_baseline_score", label: "Domain score", num: true, format: "num" },
    { key: "identity_score_gap_vs_domain", label: "Gap", num: true, format: "num" },
    { key: "identity_train_support_combined", label: "Train support", num: true, format: "num", digits: 0 },
  ], { clickable: true, barColumn: "identity_score_gap_vs_domain" });

  document.querySelectorAll("#exp10Cases tr.clickable").forEach((row) => {
    row.addEventListener("click", async () => {
      const viewName = setView("cases");
      await loadView(viewName);
      await loadCaseDetail(row.dataset.caseIndex);
    });
  });
}

// ---------------------------------------------------------------------------
// EXP 3: Faithfulness
// ---------------------------------------------------------------------------

async function loadExp3() {
  const data = await api("/api/exp/faithfulness");
  renderFindings("exp3Findings", data.findings);
  const cf = rankerRow(data.auc_summary, "counterfactual");
  const att = rankerRow(data.auc_summary, "attention");
  const rnd = rankerRow(data.auc_summary, "random");
  const highWrong = (data.bucket_summary || []).find((row) => row.confidence_bucket === "high_confidence_wrong") || {};
  const lowConf = (data.bucket_summary || []).find((row) => row.confidence_bucket === "low_confidence") || {};
  const compLift = Number(cf.mean_comprehensiveness_auc) / Math.max(Number(rnd.mean_comprehensiveness_auc) || 0, 1e-9);
  const suffDelta = Number(cf.mean_sufficiency_auc) - (Number(att.mean_sufficiency_auc) || 0);
  renderResultStory("exp3ResultStory", [
    {
      kicker: "Comprehensiveness lift",
      valueText: Number.isFinite(compLift) ? `${compLift.toFixed(1)}x` : "—",
      title: "Counterfactual evidence damages the prediction fastest.",
      body: `Mean AUC ${number(cf.mean_comprehensiveness_auc, 3) || "—"} vs random ${number(rnd.mean_comprehensiveness_auc, 3) || "—"}.`,
      tone: "good",
    },
    {
      kicker: "Sufficiency edge",
      valueText: number(suffDelta, 3) || "—",
      title: "Top-k counterfactual evidence preserves the prediction better than attention.",
      body: `Counterfactual ${number(cf.mean_sufficiency_auc, 3) || "—"} vs attention ${number(att.mean_sufficiency_auc, 3) || "—"}.`,
      tone: suffDelta > 0 ? "good" : "warn",
    },
    {
      kicker: "Confident wrong cases",
      valueText: number(highWrong.n_cases, 0) || "—",
      title: "Wrong predictions can still look internally clean.",
      body: `Mean confidence ${number(highWrong.mean_confidence, 2) || "—"} and evidence purity ${pct(highWrong.mean_evidence_purity) || "—"}.`,
      tone: "warn",
    },
    {
      kicker: "Borderline cases",
      valueText: number(lowConf.n_cases, 0) || "—",
      title: "Low-confidence predictions have larger single-evidence swings.",
      body: `Mean top |Δp| ${number(lowConf.mean_top_abs_delta_pred_proba, 3) || "—"}; these are good Case Explorer targets.`,
      tone: "neutral",
    },
  ]);

  byId("exp3Auc").innerHTML = renderSimpleTable(data.auc_summary, [
    { key: "ranker", label: "Ranker" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "mean_sufficiency_auc", label: "Sufficiency AUC", num: true, format: "num", help: "Higher = compact top-k preserves prediction." },
    { key: "mean_comprehensiveness_auc", label: "Comprehensiveness AUC", num: true, format: "num", help: "Higher = top-k removal collapses prediction faster." },
    { key: "median_sufficiency_auc", label: "Median suff.", num: true, format: "num" },
    { key: "median_comprehensiveness_auc", label: "Median comp.", num: true, format: "num" },
  ], { barColumn: "mean_comprehensiveness_auc" });

  byId("exp3Sufficiency").innerHTML = renderLineChart(data.curve_summary, "mean_sufficiency_proba", "predicted-class probability");
  byId("exp3Comprehensiveness").innerHTML = renderLineChart(data.curve_summary, "mean_comprehensiveness_drop_fraction", "probability drop fraction");

  byId("exp3Buckets").innerHTML = renderSimpleTable(data.bucket_summary, [
    { key: "confidence_bucket", label: "Bucket" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "accuracy", label: "Accuracy", num: true, format: "pct" },
    { key: "mean_confidence", label: "Mean confidence", num: true, format: "num" },
    { key: "mean_top_abs_delta_pred_proba", label: "Top |Δp|", num: true, format: "num" },
    { key: "mean_evidence_purity", label: "Evidence purity", num: true, format: "pct" },
    { key: "mean_support_train_n", label: "Train support", num: true, format: "num", digits: 0 },
    { key: "most_common_evidence_types", label: "Common evidence", format: "list", limit: 3 },
  ], { barColumn: "mean_top_abs_delta_pred_proba" });

  byId("exp3BucketEvidence").innerHTML = renderSimpleTable(data.bucket_evidence_types, [
    { key: "confidence_bucket", label: "Bucket" },
    { key: "evidence_type", label: "Evidence type" },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "bucket_share", label: "Share", num: true, format: "pct" },
    { key: "mean_top_abs_delta_pred_proba", label: "Top |Δp|", num: true, format: "num" },
    { key: "mean_evidence_purity", label: "Purity", num: true, format: "pct" },
  ], { barColumn: "n_cases" });
}

// ---------------------------------------------------------------------------
// EXP 4: Communities
// ---------------------------------------------------------------------------

async function loadExp4() {
  const data = await api("/api/exp/communities");
  if (!data.available) {
    ["exp4Top", "exp4Risky", "exp4Domains"].forEach((id) => {
      byId(id).innerHTML = `<div class="detail-body empty">Pattern outputs not found. Run pattern_why scripts then refresh.</div>`;
    });
    renderFindings("exp4Findings", []);
    renderResultStory("exp4ResultStory", []);
    return;
  }
  renderFindings("exp4Findings", data.findings);
  const largest = (data.top_communities || [])[0] || {};
  const risky = (data.risky_communities || [])[0] || {};
  const topDomain = bestBy(data.domain_counts, (row) => row.n_cases);
  renderResultStory("exp4ResultStory", [
    {
      kicker: "Structural coverage",
      valueText: number(data.n_communities, 0) || "—",
      title: "The corpus splits into recurring legal neighborhoods.",
      body: `${number(data.n_cases_in_communities, 0) || "—"} cases, size-weighted accuracy ${pct(data.weighted_accuracy) || "—"}.`,
      tone: "neutral",
    },
    {
      kicker: "Largest community",
      valueText: `#${largest.community_id ?? "—"}`,
      title: `${number(largest.size, 0) || "—"} cases, ${largest.dominant_domain_bucket || "unknown"} dominant domain`,
      body: `Dominant label ${largest.dominant_label ?? "—"}; accuracy ${pct(largest.accuracy) || "—"}.`,
      tone: "neutral",
    },
    {
      kicker: "Audit priority",
      valueText: `#${risky.community_id ?? "—"}`,
      title: `${number(risky.high_confidence_wrong_n, 0) || "—"} high-confidence wrong cases`,
      body: `${risky.dominant_domain_bucket || "Unknown domain"} community with ${number(risky.size, 0) || "—"} cases. Click the table row to inspect authorities.`,
      tone: "warn",
    },
    {
      kicker: "Dominant domain mass",
      valueText: topDomain.domain || "—",
      title: `${number(topDomain.n_cases, 0) || "—"} cases across ${number(topDomain.n_communities, 0) || "—"} communities`,
      body: "This helps distinguish genuine legal structure from isolated cluster artifacts.",
      tone: "neutral",
    },
  ]);

  const cols = [
    { key: "community_id", label: "ID", num: true, format: "num", digits: 0 },
    { key: "size", label: "Size", num: true, format: "num", digits: 0 },
    { key: "dominant_domain_bucket", label: "Domain" },
    { key: "dominant_label", label: "Label", format: "label" },
    { key: "accuracy", label: "Accuracy", num: true, format: "pct" },
    { key: "mean_confidence", label: "Confidence", num: true, format: "num" },
    { key: "high_confidence_wrong_n", label: "Confident wrong", num: true, format: "num", digits: 0 },
    { key: "top_provisions", label: "Top provisions", format: "list", limit: 3 },
  ];
  byId("exp4Top").innerHTML = renderSimpleTable(data.top_communities, cols, { clickableCommunity: true, barColumn: "size" });
  byId("exp4Risky").innerHTML = renderSimpleTable(data.risky_communities, cols, { clickableCommunity: true, barColumn: "high_confidence_wrong_n" });
  byId("exp4Domains").innerHTML = renderSimpleTable(data.domain_counts, [
    { key: "domain", label: "Domain bucket" },
    { key: "n_communities", label: "Communities", num: true, format: "num", digits: 0 },
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
  ], { barColumn: "n_cases" });

  document.querySelectorAll("#exp4Top tr.community-row, #exp4Risky tr.community-row").forEach((row) => {
    row.addEventListener("click", () => loadCommunityDetail(row.dataset.communityId));
  });
}

async function loadCommunityDetail(communityId) {
  const panel = byId("exp4DetailPanel");
  panel.hidden = false;
  const target = byId("exp4Detail");
  target.innerHTML = `<div class="inspector-loading">Loading…</div>`;
  const data = await api(`/api/community?community_id=${encodeURIComponent(communityId)}`);
  const profile = data.profile || {};
  const split = data.embedding_split || {};
  if (profile.community_id === undefined) {
    target.innerHTML = `<div class="detail-body empty">Community not found.</div>`;
    return;
  }
  target.innerHTML = `
    <div class="case-title">
      <h3>Community ${escapeHtml(profile.community_id)}</h3>
      <div class="case-meta">
        <span class="pill">${number(profile.size, 0)} cases</span>
        <span class="pill">Domain ${escapeHtml(profile.dominant_domain_bucket)}</span>
        <span class="pill">Dominant label ${escapeHtml(profile.dominant_label)}</span>
        <span class="pill">Accuracy ${pct(profile.accuracy)}</span>
      </div>
      <p class="case-reading">
        Top provisions: ${escapeHtml(profile.top_provisions || "none")}.<br>
        Top precedents: ${escapeHtml(profile.top_precedents || "none")}.
      </p>
    </div>
    <div class="detail-grid">
      <div class="detail-stat"><div class="small">Mean confidence</div><div class="value">${number(profile.mean_confidence, 4)}</div></div>
      <div class="detail-stat"><div class="small">Confident wrong</div><div class="value">${number(profile.high_confidence_wrong_n, 0)}</div></div>
      <div class="detail-stat"><div class="small">Label 1 rate</div><div class="value">${pct(profile.label_1_rate)}</div></div>
      <div class="detail-stat"><div class="small">Embedding noise</div><div class="value">${pct(split.noise_rate)}</div></div>
    </div>
    <div class="inspector-section">
      <h4>Enriched Evidence</h4>
      ${renderSimpleTable(data.features || [], [
        { key: "feature_rank", label: "Rank", num: true, format: "num", digits: 0 },
        { key: "feature_type", label: "Type" },
        { key: "feature_name", label: "Evidence" },
        { key: "case_count_in_community", label: "Cases in community", num: true, format: "num", digits: 0 },
        { key: "community_rate", label: "Community rate", num: true, format: "pct" },
        { key: "corpus_rate", label: "Corpus rate", num: true, format: "pct" },
        { key: "enrichment", label: "Enrichment", num: true, format: "num" },
      ], { barColumn: "enrichment" })}
    </div>
    <div class="inspector-section">
      <h4>Representative Cases</h4>
      ${renderSimpleTable(data.representative_cases || [], [
        { key: "case_index", label: "Index", num: true, format: "num", digits: 0 },
        { key: "case_id", label: "Case" },
        { key: "split", label: "Split" },
        { key: "target_label", label: "Target", format: "label" },
        { key: "pred_label", label: "Pred", format: "label" },
        { key: "confidence", label: "Confidence", num: true, format: "num" },
        { key: "pagerank", label: "PageRank", num: true, format: "num" },
      ], { clickable: true, barColumn: "pagerank" })}
    </div>`;
  document.querySelectorAll("#exp4Detail tr.clickable").forEach((row) => {
    row.addEventListener("click", () => {
      setView("cases");
      loadCaseDetail(row.dataset.caseIndex);
    });
  });
}

// ---------------------------------------------------------------------------
// EXP 5: Embedding Clusters
// ---------------------------------------------------------------------------

async function loadExp5() {
  const data = await api("/api/exp/embedding_clusters");
  renderFindings("exp5Findings", data.findings);

  const align = data.alignment_row || {};
  const pureClusters = (data.clusters || []).filter((row) => {
    const size = Number(row.size) || 0;
    const labelRate = Math.max(Number(row.label_1_rate) || 0, Number(row["label_-1_rate"]) || 0);
    return size >= 50 && labelRate >= 0.9 && !row.is_noise;
  });
  const mostSplit = (data.splits || [])[0] || {};
  renderResultStory("exp5ResultStory", [
    {
      kicker: "Topology alignment",
      valueText: number(align.normalized_mutual_info_all, 3) || "—",
      title: "Embedding clusters do not simply mirror structural communities.",
      body: `ARI ${number(align.adjusted_rand_all, 3) || "—"}; V-measure ${number(align.v_measure_all, 3) || "—"}.`,
      tone: Number(align.normalized_mutual_info_all) < 0.1 ? "good" : "neutral",
    },
    {
      kicker: "Embedding collapse",
      valueText: `${number(align.n_embedding_clusters_including_noise, 0) || "—"} vs ${number(align.n_structural_communities, 0) || "—"}`,
      title: "Few embedding clusters cover many structural communities.",
      body: `Noise rate ${pct(align.noise_rate) || "—"} shows how much of the space remains diffuse.`,
      tone: "neutral",
    },
    {
      kicker: "Outcome-pure pockets",
      valueText: number(pureClusters.length, 0) || "0",
      title: "Some embedding regions are already strongly label-separated.",
      body: "Large pure clusters are the cleanest examples of model-discovered outcome geometry.",
      tone: pureClusters.length ? "good" : "neutral",
    },
    {
      kicker: "Most split community",
      valueText: `#${mostSplit.community_id ?? "—"}`,
      title: `${number(mostSplit.n_embedding_clusters, 0) || "—"} embedding clusters inside one structural community`,
      body: "This is where HGT subdivides a legal neighborhood into finer model regions.",
      tone: "neutral",
    },
  ]);

  byId("exp5FlowGraph").innerHTML = renderCommunityEmbeddingFlow(data.flow);

  byId("exp5Alignment").innerHTML = renderSimpleTable([align], [
    { key: "n_cases", label: "Cases", num: true, format: "num", digits: 0 },
    { key: "n_structural_communities", label: "Communities", num: true, format: "num", digits: 0 },
    { key: "n_embedding_clusters_including_noise", label: "Embedding clusters", num: true, format: "num", digits: 0 },
    { key: "noise_rate", label: "Noise rate", num: true, format: "pct" },
    { key: "normalized_mutual_info_all", label: "NMI", num: true, format: "num", help: "0 = independent, 1 = identical." },
    { key: "v_measure_all", label: "V-measure", num: true, format: "num" },
    { key: "adjusted_rand_all", label: "ARI", num: true, format: "num" },
    { key: "homogeneity_all", label: "Homogeneity", num: true, format: "num" },
    { key: "completeness_all", label: "Completeness", num: true, format: "num" },
  ]);

  byId("exp5Clusters").innerHTML = renderSimpleTable(data.clusters, [
    { key: "embedding_cluster_id", label: "Cluster", num: true, format: "num", digits: 0 },
    { key: "is_noise", label: "Noise" },
    { key: "size", label: "Size", num: true, format: "num", digits: 0 },
    { key: "dominant_label", label: "Label", format: "label" },
    { key: "accuracy", label: "Accuracy", num: true, format: "pct" },
    { key: "label_1_rate", label: "Label 1 rate", num: true, format: "pct" },
    { key: "dominant_structural_community_id", label: "Dominant community" },
    { key: "top_domain_buckets", label: "Domains", format: "list", limit: 4 },
  ], { barColumn: "size" });

  byId("exp5Splits").innerHTML = renderSimpleTable(data.splits, [
    { key: "community_id", label: "Community", num: true, format: "num", digits: 0 },
    { key: "size", label: "Size", num: true, format: "num", digits: 0 },
    { key: "n_embedding_clusters", label: "Embedding clusters", num: true, format: "num", digits: 0 },
    { key: "noise_rate", label: "Noise rate", num: true, format: "pct" },
    { key: "dominant_embedding_cluster_id", label: "Dominant cluster" },
    { key: "dominant_embedding_cluster_rate", label: "Dominant rate", num: true, format: "pct" },
    { key: "label_distribution", label: "Labels", format: "list", limit: 2 },
  ], { clickableCommunity: true, barColumn: "n_embedding_clusters" });
  document.querySelectorAll("#exp5Splits tr.community-row").forEach((row) => {
    row.addEventListener("click", () => {
      setView("exp4");
      loadView("exp4").then(() => loadCommunityDetail(row.dataset.communityId));
    });
  });
}

// ---------------------------------------------------------------------------
// EXP 6: Opposite Cases
// ---------------------------------------------------------------------------

async function loadExp6() {
  const data = await api("/api/exp/neighborhoods");
  renderFindings("exp6Findings", data.findings);
  if (!data.available) {
    ["exp6CosineSummary", "exp6FeatureTypes", "exp6Pairs", "exp6ContrastGraph"].forEach((id) => byId(id).innerHTML = `<div class="detail-body empty">Neighborhood outputs not found.</div>`);
    renderResultStory("exp6ResultStory", []);
    return;
  }

  const c = data.cosine_summary || {};
  const topPair = (data.top_pairs || [])[0] || {};
  const queryTop = bestBy((data.feature_type_summary || []).filter((row) => row.side === "query_only"), (row) => row.n);
  const oppositeTop = bestBy((data.feature_type_summary || []).filter((row) => row.side === "opposite_only"), (row) => row.n);
  renderResultStory("exp6ResultStory", [
    {
      kicker: "Boundary proximity",
      valueText: number(c.mean, 3) || "—",
      title: "Most cases have a very close opposite-label neighbor.",
      body: `Median cosine ${number(c.median, 3) || "—"} across ${number(data.n_pairs, 0) || "—"} test cases.`,
      tone: "warn",
    },
    {
      kicker: "Closest pair",
      valueText: number(topPair.cosine_similarity, 4) || "—",
      title: compactText(topPair.case_id || "No pair found", 90),
      body: `Opposite case ${topPair.nearest_opposite_case_index ?? "—"}; use the table to inspect the boundary evidence.`,
      tone: "neutral",
    },
    {
      kicker: "Query-only signal",
      valueText: queryTop.feature_type || "—",
      title: `${number(queryTop.n, 0) || "—"} occurrences`,
      body: "Feature type most often present in the query case and absent from its opposite-label neighbor.",
      tone: queryTop.feature_type === "judge" || queryTop.feature_type === "court" ? "warn" : "neutral",
    },
    {
      kicker: "Opposite-only signal",
      valueText: oppositeTop.feature_type || "—",
      title: `${number(oppositeTop.n, 0) || "—"} occurrences`,
      body: "Feature type most often present in the opposite-label neighbor and absent from the query case.",
      tone: oppositeTop.feature_type === "precedent" || oppositeTop.feature_type === "provision" ? "good" : "neutral",
    },
  ]);
  state.exp6SelectedCase = topPair.case_index;
  if (byId("exp6CasePick")) byId("exp6CasePick").value = topPair.case_index ?? "";
  byId("exp6ContrastGraph").innerHTML = renderCaseContrastGraph(data.contrast_graph);

  byId("exp6CosineSummary").innerHTML = `
    <div class="kv-grid">
      <div class="kv"><span>Pairs analyzed</span><strong>${number(data.n_pairs, 0)}</strong></div>
      <div class="kv"><span>Mean cosine</span><strong>${number(c.mean, 4)}</strong></div>
      <div class="kv"><span>Median cosine</span><strong>${number(c.median, 4)}</strong></div>
      <div class="kv"><span>P95 cosine</span><strong>${number(c.p95, 4)}</strong></div>
    </div>
    <p class="panel-note-inline">High cosine + opposite label = the embedding sees the cases as nearly identical despite the actual outcome differing. These are decision-boundary cases worth manual review.</p>`;

  byId("exp6FeatureTypes").innerHTML = renderSimpleTable(data.feature_type_summary, [
    { key: "side", label: "Side" },
    { key: "feature_type", label: "Feature type" },
    { key: "n", label: "Occurrences", num: true, format: "num", digits: 0 },
  ], { barColumn: "n" });

  loadExp6Pairs(data.top_pairs || []);
}

async function loadOppositeGraphForCase(caseIndex) {
  const cleanIndex = String(caseIndex ?? "").trim();
  if (!cleanIndex) return;
  state.exp6SelectedCase = cleanIndex;
  if (byId("exp6CasePick")) byId("exp6CasePick").value = cleanIndex;
  document.querySelectorAll("#exp6Pairs tr.clickable").forEach((other) => {
    other.classList.toggle("selected", String(other.dataset.caseIndex) === cleanIndex);
  });
  byId("exp6ContrastGraph").innerHTML = `<div class="detail-body empty">Loading contrast graph for case ${escapeHtml(cleanIndex)}…</div>`;
  try {
    const graph = await api(`/api/opposite_graph?case_index=${encodeURIComponent(cleanIndex)}`);
    byId("exp6ContrastGraph").innerHTML = renderCaseContrastGraph(graph);
  } catch (err) {
    byId("exp6ContrastGraph").innerHTML = `<div class="detail-body empty">${escapeHtml(err.message)}</div>`;
  }
}

function loadExp6Pairs(pairs) {
  const q = (byId("exp6Search").value || "").trim().toLowerCase();
  const filtered = q ? pairs.filter((r) =>
    Object.values(r).some((v) => String(v ?? "").toLowerCase().includes(q))
  ) : pairs;
  byId("exp6Pairs").innerHTML = renderSimpleTable(filtered, [
    { key: "case_index", label: "Case", num: true, format: "num", digits: 0 },
    { key: "target_label", label: "Target", format: "label" },
    { key: "pred_label", label: "Pred", format: "label" },
    { key: "community_id", label: "Community", num: true, format: "num", digits: 0 },
    { key: "nearest_opposite_case_index", label: "Opposite case", num: true, format: "num", digits: 0 },
    { key: "nearest_opposite_target_label", label: "Opp. label", format: "label" },
    { key: "cosine_similarity", label: "Cosine", num: true, format: "num" },
    { key: "top_query_only_features", label: "Case-only evidence", format: "list", limit: 3 },
    { key: "top_opposite_only_features", label: "Opposite-only evidence", format: "list", limit: 3 },
  ], { clickable: true, barColumn: "cosine_similarity" });
  document.querySelectorAll("#exp6Pairs tr.clickable").forEach((row) => {
    row.classList.toggle("selected", String(row.dataset.caseIndex) === String(state.exp6SelectedCase));
    row.addEventListener("click", () => loadOppositeGraphForCase(row.dataset.caseIndex));
  });
  // Cache pairs so search re-uses without refetching
  state.exp6Pairs = pairs;
}

// ---------------------------------------------------------------------------
// Shared visual helpers for exp7/8/9
// ---------------------------------------------------------------------------

function heroTile({ kicker, big, sub, tone = "" }) {
  return `<div class="tile ${tone ? `tone-${tone}` : ""}">
    <div class="kicker">${escapeHtml(kicker)}</div>
    <div class="big">${escapeHtml(big)}</div>
    <div class="sub">${sub || ""}</div>
  </div>`;
}

function renderHero(targetId, tiles) {
  const target = byId(targetId);
  if (!target) return;
  target.innerHTML = (tiles || []).filter(Boolean).map(heroTile).join("");
}

function statBlock(label, val, tone = "") {
  return `<div class="stat"><span class="label">${escapeHtml(label)}</span><span class="val ${tone ? `tone-${tone}` : ""}">${val}</span></div>`;
}

function stackBar(label, segments) {
  // segments: [{ pct: 0.6, cls: "stack-seg-pos", label: "60%" }]
  const total = segments.reduce((s, x) => s + (x.pct || 0), 0);
  if (total <= 0) return "";
  const segs = segments
    .filter((s) => (s.pct || 0) > 0)
    .map((s) => `<div class="stack-seg ${s.cls}" style="width:${(100 * s.pct / total).toFixed(2)}%" title="${escapeHtml(s.title || s.label || "")}">${s.pct >= 0.08 ? escapeHtml(s.label || "") : ""}</div>`)
    .join("");
  return `<div class="stack-bar-row">
    <span class="stack-label">${escapeHtml(label)}</span>
    <div class="stack-bar">${segs}</div>
  </div>`;
}

function safeNum(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

// ---------------------------------------------------------------------------
// EXP 7: Full-Graph Communities — narrative + community cards
// ---------------------------------------------------------------------------

const SWEEP_COLS = [
  { key: "resolution", label: "Resolution γ", num: true, format: "num", digits: 2 },
  { key: "n_communities", label: "Communities", num: true, format: "num", digits: 0 },
  { key: "quality", label: "Quality", num: true, format: "num", digits: 4 },
  { key: "max_community_size", label: "Max size", num: true, format: "num", digits: 0 },
  { key: "median_community_size", label: "Median size", num: true, format: "num", digits: 1 },
  { key: "min_community_size", label: "Min size", num: true, format: "num", digits: 0 },
  { key: "elapsed_seconds", label: "Elapsed (s)", num: true, format: "num", digits: 1 },
];

const FG_BOUNDARY_COLS = [
  { key: "case_index", label: "Case", num: true, format: "num", digits: 0 },
  { key: "case_id", label: "Case id" },
  { key: "case_community", label: "Community", num: true, format: "num", digits: 0 },
  { key: "n_authorities", label: "Authorities", num: true, format: "num", digits: 0 },
  { key: "n_communities_touched", label: "Spread", num: true, format: "num", digits: 0 },
  { key: "neighborhood_normalized_entropy", label: "Norm entropy", num: true, format: "num" },
  { key: "split", label: "Split" },
  { key: "target_label", label: "Target", format: "label" },
  { key: "pred_label", label: "Pred", format: "label" },
  { key: "confidence", label: "Confidence", num: true, format: "num" },
  { key: "domain_bucket", label: "Domain" },
];

function exp7CommunityCard(profile, predictions, authorities) {
  const acc = safeNum(profile.accuracy);
  const accTest = safeNum(profile.accuracy_test);
  const conf = safeNum(profile.mean_confidence);
  const hcWrong = safeNum(profile.high_conf_wrong_n) || 0;
  const nCases = safeNum(profile.n_cases) || 0;
  const labelPosRate = safeNum(profile.label_1_rate) || 0;
  const labelNegRate = safeNum(profile["label_-1_rate"]) || 0;

  const labelClass = String(profile.dominant_label) === "1" ? "label-pos" : "label-neg";
  const labelText = `Label ${profile.dominant_label ?? "?"} dominant`;
  const accTone = acc === null ? "" : acc >= 0.8 ? "good" : acc < 0.55 ? "bad" : "warn";
  const cardTone = hcWrong >= Math.max(8, nCases * 0.03) ? "warn" : acc !== null && acc >= 0.85 ? "good" : "";

  const labelStack = stackBar("Label balance", [
    { pct: labelPosRate, cls: "stack-seg-pos", label: `+1 ${pct(labelPosRate)}`, title: `Label 1: ${pct(labelPosRate)}` },
    { pct: labelNegRate, cls: "stack-seg-neg", label: `−1 ${pct(labelNegRate)}`, title: `Label −1: ${pct(labelNegRate)}` },
  ]);

  let predStack = "";
  if (predictions) {
    const hcCorrect = safeNum(predictions.high_conf_correct_rate) || 0;
    const hcWrongRate = safeNum(predictions.high_conf_wrong_rate) || 0;
    const lcCorrect = safeNum(predictions.low_conf_correct_rate) || 0;
    const lcWrong = safeNum(predictions.low_conf_wrong_rate) || 0;
    predStack = stackBar("Prediction quality", [
      { pct: hcCorrect, cls: "stack-seg-good", label: `HC ✓ ${pct(hcCorrect)}`, title: `High-confidence correct ${pct(hcCorrect)}` },
      { pct: hcWrongRate, cls: "stack-seg-warn", label: `HC ✗ ${pct(hcWrongRate)}`, title: `High-confidence wrong ${pct(hcWrongRate)}` },
      { pct: lcCorrect, cls: "stack-seg-muted", label: `LC ✓ ${pct(lcCorrect)}`, title: `Low-confidence correct ${pct(lcCorrect)}` },
      { pct: lcWrong, cls: "stack-seg-caution", label: `LC ✗ ${pct(lcWrong)}`, title: `Low-confidence wrong ${pct(lcWrong)}` },
    ]);
  }

  const authsByType = {};
  (authorities || []).forEach((row) => {
    const t = row.feature_type;
    if (!t) return;
    if (!authsByType[t]) authsByType[t] = [];
    authsByType[t].push(row);
  });
  const authBlocks = ["statute", "provision", "precedent"]
    .filter((t) => authsByType[t] && authsByType[t].length)
    .map((t) => {
      const pills = authsByType[t]
        .slice(0, 5)
        .map((row) => {
          const name = compactText(String(row.feature_name || ""), 64);
          const cnt = number(row.case_count_in_community, 0);
          return `<span class="evidence-pill">${escapeHtml(name)}<span class="count">×${cnt}</span></span>`;
        })
        .join("");
      return `<div class="evidence-list"><div class="header">${escapeHtml(t)}s</div>${pills}</div>`;
    })
    .join("");

  return `<article class="community-card${cardTone ? ` tone-${cardTone}` : ""}" data-community-id="${escapeHtml(profile.community_id)}">
    <header class="card-header">
      <div class="card-id">#${escapeHtml(profile.community_id)} <small>${number(nCases, 0)} cases</small></div>
      <div class="card-pills">
        <span class="pill ${labelClass}">${escapeHtml(labelText)}</span>
        <span class="pill">${escapeHtml(profile.dominant_domain_bucket || "unknown")}</span>
      </div>
    </header>
    <div class="stat-row">
      ${statBlock("Accuracy", acc === null ? "—" : pct(acc), accTone)}
      ${statBlock("Test acc", accTest === null ? "—" : pct(accTest))}
      ${statBlock("Confidence", conf === null ? "—" : number(conf, 3))}
      ${statBlock("HC wrong", number(hcWrong, 0), hcWrong > 0 ? (hcWrong >= 10 ? "bad" : "warn") : "")}
    </div>
    <div>
      ${labelStack}
      ${predStack}
    </div>
    ${authBlocks ? `<div>${authBlocks}</div>` : ""}
  </article>`;
}

async function loadExp7(requestedResolution) {
  const qs = requestedResolution !== undefined && requestedResolution !== null
    ? `?resolution=${encodeURIComponent(requestedResolution)}`
    : "";
  const data = await api(`/api/exp/full_graph_communities${qs}`);
  const empty = `<div class="detail-body empty">Full-graph community outputs not found. Run <code>run_full_graph_communities.sh</code>, then refresh.</div>`;
  if (!data.available) {
    ["exp7Sweep", "exp7Cards", "exp7RiskyCards", "exp7Boundary"].forEach((id) => byId(id).innerHTML = empty);
    renderHero("exp7Hero", []);
    renderFindings("exp7Findings", []);
    const sel = byId("exp7ResolutionSelect");
    if (sel) sel.innerHTML = "";
    return;
  }

  renderFindings("exp7Findings", data.findings);

  const select = byId("exp7ResolutionSelect");
  if (select) {
    select.innerHTML = (data.resolutions || [])
      .map((r) => `<option value="${r}"${r === data.current_resolution ? " selected" : ""}>γ = ${Number(r).toFixed(2)}</option>`)
      .join("");
    select.onchange = (e) => loadExp7(parseFloat(e.target.value));
  }

  byId("exp7Sweep").innerHTML = renderSimpleTable(data.sweep_summary || [], SWEEP_COLS, { barColumn: "n_communities" });

  const top = data.top_profiles || [];
  const risky = data.risky_predictions || [];
  const boundary = data.boundary_cases || [];
  const predById = data.predictions_by_community || {};
  const authsById = data.authorities_by_community || {};

  const biggest = top[0] || {};
  const worstAccuracy = risky.find((r) => Number(r.high_conf_wrong_n) > 0) || risky[0] || {};
  const sumHCWrong = top.reduce((s, r) => s + (Number(r.high_conf_wrong_n) || 0), 0);

  renderHero("exp7Hero", [
    {
      kicker: `γ = ${Number(data.current_resolution).toFixed(2)}`,
      big: number(data.n_communities, 0) || "—",
      sub: `communities at this resolution<br>${number(data.n_cases_in_communities, 0) || "—"} cases assigned overall.`,
    },
    {
      kicker: "Largest community",
      big: `#${biggest.community_id ?? "—"}`,
      sub: `${number(biggest.n_cases, 0) || "—"} cases · ${escapeHtml(biggest.dominant_domain_bucket || "unknown")}<br>Accuracy ${pct(biggest.accuracy) || "—"}`,
    },
    {
      kicker: "Confident-wrong total",
      big: number(sumHCWrong, 0) || "0",
      sub: `cases the model got wrong with high confidence,<br>across the top ${top.length} communities shown.`,
      tone: sumHCWrong > 0 ? "warn" : "good",
    },
    {
      kicker: "Boundary cases",
      big: number(boundary.length, 0) || "0",
      sub: `cases whose authorities span <em>multiple</em> communities — naturally ambiguous.`,
      tone: boundary.length ? "warn" : "good",
    },
  ]);

  byId("exp7Cards").innerHTML = top.length
    ? top.map((p) => exp7CommunityCard(p, predById[p.community_id], authsById[p.community_id])).join("")
    : `<div class="detail-body empty">No community profiles yet.</div>`;

  byId("exp7RiskyCards").innerHTML = risky.length
    ? risky
        .map((p) => {
          // Build a synthetic "profile" object from prediction payload so the card layout works.
          const profile = top.find((t) => t.community_id === p.community_id) || {
            community_id: p.community_id,
            n_cases: p.n_cases,
            accuracy: p.accuracy,
            accuracy_test: p.accuracy_test,
            high_conf_wrong_n: p.high_conf_wrong_n,
            mean_confidence: p.mean_confidence_correct,
            dominant_label: "?",
            dominant_domain_bucket: "?",
            label_1_rate: 0,
            "label_-1_rate": 0,
          };
          return exp7CommunityCard(profile, p, authsById[p.community_id]);
        })
        .join("")
    : `<div class="detail-body empty">No confident-wrong communities at this resolution.</div>`;

  byId("exp7Boundary").innerHTML = renderSimpleTable(boundary, FG_BOUNDARY_COLS, {
    clickable: true,
    barColumn: "neighborhood_normalized_entropy",
  });

  document.querySelectorAll("#exp7Boundary tr.clickable").forEach((row) => {
    row.addEventListener("click", () => {
      setView("cases");
      loadCaseDetail(row.dataset.caseIndex);
    });
  });
}

// ---------------------------------------------------------------------------
// EXP 8: Community Hierarchy — visual lineage + interpretation
// ---------------------------------------------------------------------------

function exp8LineageRow(row, resolutions) {
  const leafGamma = safeNum(row.leaf_resolution);
  const leafSize = safeNum(row.leaf_size) || 0;
  const coarseRes = resolutions.filter((r) => r < leafGamma).sort((a, b) => a - b);
  const steps = coarseRes
    .map((g) => {
      const key = `parent_res_${g.toFixed(2)}`;
      const shareKey = `parent_share_res_${g.toFixed(2)}`;
      const pid = row[key];
      const share = safeNum(row[shareKey]);
      if (pid === undefined || pid === null) return "";
      const weak = share !== null && share < 0.5 ? " weak" : "";
      return `<span class="hier-step${weak}">
        <span class="gamma">γ ${g.toFixed(2)}</span>
        <span class="pid">#${escapeHtml(pid)}</span>
        <span class="share">${share !== null ? pct(share) : ""}</span>
      </span><span class="hier-arrow">→</span>`;
    })
    .join("");
  return `<div class="hier-row">
    <div class="hier-leaf">
      <span class="id">#${escapeHtml(row.leaf_community)}</span>
      <span class="size">leaf γ ${leafGamma.toFixed(2)} · ${number(leafSize, 0)} cases</span>
    </div>
    <div class="hier-chain">${steps}<span class="hier-step" style="background:#ecf6ee;border-color:#c5dec9"><span class="gamma">leaf γ ${leafGamma.toFixed(2)}</span><span class="pid">#${escapeHtml(row.leaf_community)}</span><span class="share">${number(leafSize, 0)} cases</span></span></div>
  </div>`;
}

function exp8HeatmapCell(value) {
  const v = safeNum(value);
  if (v === null) return "—";
  // 0 = white, 1 = teal
  const bg = `rgba(23, 107, 135, ${(v * 0.85).toFixed(2)})`;
  const fg = v > 0.55 ? "#ffffff" : "#1f2933";
  return `<span class="heatmap" style="background:${bg};color:${fg}">${v.toFixed(3)}</span>`;
}

async function loadExp8() {
  const data = await api("/api/exp/community_hierarchy");
  const empty = `<div class="detail-body empty">Community hierarchy outputs not found. Run <code>community_hierarchy_analysis.py</code>, then refresh.</div>`;
  if (!data.available) {
    ["exp8Lineage", "exp8Pairwise", "exp8Pairs"].forEach((id) => byId(id).innerHTML = empty);
    byId("exp8HierLegend").innerHTML = "";
    renderHero("exp8Hero", []);
    renderFindings("exp8Findings", []);
    return;
  }
  renderFindings("exp8Findings", data.findings);

  const chains = data.lineage_chains || [];
  const allRes = new Set();
  chains.forEach((row) => {
    const leafG = safeNum(row.leaf_resolution);
    if (leafG !== null) allRes.add(leafG);
    Object.keys(row).forEach((k) => {
      if (k.startsWith("parent_res_")) {
        const g = safeNum(k.replace("parent_res_", ""));
        if (g !== null) allRes.add(g);
      }
    });
  });
  const resolutions = Array.from(allRes).sort((a, b) => a - b);

  // Hero tally
  const totalChains = chains.length;
  const driftRows = (data.pairwise || []).slice().sort((a, b) => safeNum(a.normalized_mutual_info) - safeNum(b.normalized_mutual_info));
  const mostDrift = driftRows[0] || {};
  const cleanestSplit = chains.filter((r) => {
    const keys = Object.keys(r).filter((k) => k.startsWith("parent_share_res_"));
    return keys.length && keys.every((k) => safeNum(r[k]) !== null && safeNum(r[k]) >= 0.95);
  }).length;
  renderHero("exp8Hero", [
    {
      kicker: "Resolutions in sweep",
      big: number(resolutions.length, 0) || "—",
      sub: `γ values: ${resolutions.map((g) => g.toFixed(2)).join(", ")}`,
    },
    {
      kicker: "Leaf communities",
      big: number(totalChains, 0) || "—",
      sub: "communities at the finest γ that we trace back through every coarser γ.",
    },
    {
      kicker: "Cleanest splits",
      big: number(cleanestSplit, 0) || "—",
      sub: `leaves whose every ancestor share is ≥ 95%.<br>These nest cleanly — no reshuffling.`,
      tone: cleanestSplit ? "good" : "",
    },
    {
      kicker: "Most reshuffling",
      big: `γ${safeNum(mostDrift.coarse_resolution)?.toFixed(2) ?? "—"} → γ${safeNum(mostDrift.fine_resolution)?.toFixed(2) ?? "—"}`,
      sub: `NMI ${number(mostDrift.normalized_mutual_info, 3) || "—"}, ARI ${number(mostDrift.adjusted_rand_index, 3) || "—"}`,
      tone: "warn",
    },
  ]);

  // Hierarchy legend
  byId("exp8HierLegend").innerHTML = `<span>Resolutions in this sweep:</span>` +
    resolutions.map((g) => `<span class="col-name">γ = ${g.toFixed(2)}</span>`).join("");

  // Lineage rows (limit to 30 largest)
  const sortedChains = chains.slice().sort((a, b) => (safeNum(b.leaf_size) || 0) - (safeNum(a.leaf_size) || 0)).slice(0, 30);
  byId("exp8Lineage").innerHTML = sortedChains.length
    ? sortedChains.map((r) => exp8LineageRow(r, resolutions)).join("")
    : `<div class="detail-body empty">No lineage chains available — at least two resolutions are required.</div>`;

  // Pairwise as a heatmap-ish table
  const pw = (data.pairwise || []).slice().sort((a, b) => safeNum(a.coarse_resolution) - safeNum(b.coarse_resolution) || safeNum(a.fine_resolution) - safeNum(b.fine_resolution));
  if (pw.length) {
    const rows = pw
      .map(
        (r) => `<tr>
          <td>γ ${safeNum(r.coarse_resolution)?.toFixed(2)} → γ ${safeNum(r.fine_resolution)?.toFixed(2)}</td>
          <td class="num">${number(r.n_overlap_nodes, 0)}</td>
          <td class="num">${number(r.n_coarse_communities, 0)} → ${number(r.n_fine_communities, 0)}</td>
          <td class="num">${exp8HeatmapCell(r.normalized_mutual_info)}</td>
          <td class="num">${exp8HeatmapCell(r.adjusted_rand_index)}</td>
        </tr>`
      )
      .join("");
    byId("exp8Pairwise").innerHTML = `<table>
      <thead><tr><th>Resolution pair</th><th class="num">Cases</th><th class="num">Comm count</th><th class="num" title="0 = independent, 1 = identical">NMI</th><th class="num">ARI</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  } else {
    byId("exp8Pairwise").innerHTML = `<div class="detail-body empty">No pairwise alignment data.</div>`;
  }

  const pairCols = [
    { key: "coarse_resolution", label: "Coarse γ", num: true, format: "num", digits: 2 },
    { key: "fine_resolution", label: "Fine γ", num: true, format: "num", digits: 2 },
    { key: "parent_community", label: "Parent id", num: true, format: "num", digits: 0 },
    { key: "child_community", label: "Child id", num: true, format: "num", digits: 0 },
    { key: "child_size", label: "Child size", num: true, format: "num", digits: 0 },
    { key: "overlap_n", label: "Shared cases", num: true, format: "num", digits: 0 },
    { key: "overlap_share", label: "Share of child", num: true, format: "pct" },
  ];
  byId("exp8Pairs").innerHTML = renderSimpleTable(data.lineage_pairs || [], pairCols, { barColumn: "overlap_n" });
}

// ---------------------------------------------------------------------------
// EXP 9: Bridge / Hub / Core Authorities — role tally + spread cards
// ---------------------------------------------------------------------------

function parseTopCommunities(text) {
  if (!text) return [];
  return String(text)
    .split("|")
    .map((part) => {
      const [cid, cnt] = part.split(":");
      return { community: cid, count: Number(cnt) || 0 };
    })
    .filter((x) => x.community !== undefined && x.count > 0);
}

function exp9AuthorityCard(row, role) {
  const roleClass = `role-${role}`;
  const acc = safeNum(row.model_accuracy_on_citing_cases);
  const conf = safeNum(row.mean_confidence_on_citing_cases);
  const accTone = acc === null ? "" : acc >= 0.8 ? "good" : acc < 0.55 ? "bad" : "warn";
  const topComms = parseTopCommunities(row.top_communities);
  const totalCases = safeNum(row.n_cases) || 1;
  const spreadRows = topComms
    .slice(0, 5)
    .map((c) => {
      const share = c.count / totalCases;
      return `<div class="spread-row">
        <span class="lbl">#${escapeHtml(c.community)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(share * 100).toFixed(1)}%"></div></div>
        <span class="pctv">${pct(share)}</span>
      </div>`;
    })
    .join("");

  const home = row.home_community !== undefined && row.home_community !== null
    ? `home #${row.home_community}`
    : "";
  const entropy = safeNum(row.normalized_entropy);
  const entropyText = entropy !== null ? `entropy ${entropy.toFixed(2)}` : "";
  const homeShare = safeNum(row.max_community_share);
  const homeShareText = homeShare !== null ? `top share ${pct(homeShare)}` : "";

  return `<article class="authority-card">
    <header class="card-header">
      <div>
        <span class="role-badge ${roleClass}">${role}</span>
        <span class="pill" style="margin-left:6px">${escapeHtml(row.feature_type || "?")}</span>
      </div>
      <div class="card-pills">
        ${row.n_communities_touched !== undefined ? `<span class="pill">spans ${escapeHtml(row.n_communities_touched)} communities</span>` : ""}
        ${home ? `<span class="pill">${escapeHtml(home)}</span>` : ""}
      </div>
    </header>
    <div style="font-size:14px; font-weight:600; line-height:1.35; color:#1f2933">
      ${escapeHtml(compactText(String(row.feature_name || ""), 140))}
    </div>
    <div class="stat-row three">
      ${statBlock("Citing cases", number(row.n_cases, 0))}
      ${statBlock("Model acc", acc === null ? "—" : pct(acc), accTone)}
      ${statBlock("Mean conf", conf === null ? "—" : number(conf, 3))}
    </div>
    ${spreadRows ? `<div>
      <div class="evidence-list"><div class="header">Community spread (${entropyText}${entropyText && homeShareText ? " · " : ""}${homeShareText})</div></div>
      ${spreadRows}
    </div>` : ""}
  </article>`;
}

function exp9BridgePairView(pair) {
  return `<div class="bridge-pair-row">
    <span class="bridge-pair-comm">Community #${escapeHtml(pair.community_a)}</span>
    <span class="bridge-pair-arrow">⇄</span>
    <span class="bridge-pair-comm">Community #${escapeHtml(pair.community_b)}</span>
    <span class="bridge-pair-count">${number(pair.n_bridge_authorities, 0)}<small>bridge authorities</small></span>
  </div>`;
}

async function loadExp9(requestedResolution) {
  const qs = requestedResolution !== undefined && requestedResolution !== null
    ? `?resolution=${encodeURIComponent(requestedResolution)}`
    : "";
  const data = await api(`/api/exp/bridge_hub${qs}`);
  const empty = `<div class="detail-body empty">Bridge/hub authority outputs not found. Run <code>bridge_hub_authority_analysis.py</code>, then refresh.</div>`;
  if (!data.available) {
    ["exp9RoleSummary", "exp9BridgePairs", "exp9Hubs", "exp9Bridges", "exp9Cores", "exp9HubSensitivity", "exp9MaskedHubs"].forEach((id) => byId(id).innerHTML = empty);
    renderHero("exp9Hero", []);
    renderFindings("exp9Findings", []);
    const sel = byId("exp9ResolutionSelect");
    if (sel) sel.innerHTML = "";
    return;
  }
  renderFindings("exp9Findings", data.findings);

  const select = byId("exp9ResolutionSelect");
  if (select) {
    select.innerHTML = (data.resolutions || [])
      .map((r) => `<option value="${r}"${r === data.current_resolution ? " selected" : ""}>γ = ${Number(r).toFixed(2)}</option>`)
      .join("");
    select.onchange = (e) => loadExp9(parseFloat(e.target.value));
  }

  // Aggregate role counts across all types
  const roleCounts = { core: 0, bridge: 0, hub: 0, rare: 0 };
  (data.role_summary || []).forEach((row) => {
    if (row.role && row.n_authorities) {
      roleCounts[row.role] = (roleCounts[row.role] || 0) + Number(row.n_authorities);
    }
  });
  const hubSensitivity = data.hub_sensitivity || [];
  const topHubMask = bestBy(hubSensitivity, (row) => Number(row.top_k_hubs) || 0);
  renderHero("exp9Hero", [
    { kicker: "Hub authorities", big: number(roleCounts.hub, 0) || "0", sub: "diffuse across many communities — corpus backbone.", tone: "good" },
    { kicker: "Bridge authorities", big: number(roleCounts.bridge, 0) || "0", sub: "concentrated in a small set of communities — transfer paths.", tone: "warn" },
    { kicker: "Core authorities", big: number(roleCounts.core, 0) || "0", sub: "almost all citing cases live in one community — niche anchors." },
    { kicker: "Hub stress drop", big: pct(topHubMask.accuracy_drop) || "—", sub: `top ${number(topHubMask.top_k_hubs, 0) || "—"} hubs removed; flip ${pct(topHubMask.flip_rate) || "—"}.` },
  ]);

  byId("exp9HubSensitivity").innerHTML = renderSimpleTable(hubSensitivity, [
    { key: "mask_name", label: "Mask" },
    { key: "top_k_hubs", label: "Top-k", num: true, format: "num", digits: 0 },
    { key: "masked_edge_share", label: "Edges masked", num: true, format: "pct" },
    { key: "baseline_accuracy", label: "Base acc", num: true, format: "pct" },
    { key: "masked_accuracy", label: "Masked acc", num: true, format: "pct" },
    { key: "accuracy_drop", label: "Acc drop", num: true, format: "pct" },
    { key: "macro_f1_drop", label: "Macro-F1 drop", num: true, format: "pct" },
    { key: "confidence_drop", label: "Orig-conf drop", num: true, format: "num" },
    { key: "flip_rate", label: "Flip rate", num: true, format: "pct" },
  ], { barColumn: "accuracy_drop" });

  byId("exp9MaskedHubs").innerHTML = renderSimpleTable(data.masked_hub_authorities || [], [
    { key: "hub_rank", label: "Rank", num: true, format: "num", digits: 0 },
    { key: "feature_type", label: "Type" },
    { key: "feature_name", label: "Authority" },
    { key: "n_cases", label: "Citing cases", num: true, format: "num", digits: 0 },
    { key: "n_unique_cases", label: "Unique cases", num: true, format: "num", digits: 0 },
    { key: "n_communities_touched", label: "Communities", num: true, format: "num", digits: 0 },
    { key: "max_community_share", label: "Top share", num: true, format: "pct" },
  ], { barColumn: "n_cases" });

  const summaryCols = [
    { key: "feature_type", label: "Type" },
    { key: "role", label: "Role" },
    { key: "n_authorities", label: "Authorities", num: true, format: "num", digits: 0 },
  ];
  byId("exp9RoleSummary").innerHTML = renderSimpleTable(data.role_summary || [], summaryCols, { barColumn: "n_authorities" });

  const examples = data.examples_by_role || {};
  const renderRoleCards = (role, container) => {
    const list = examples[role] || [];
    container.innerHTML = list.length
      ? list.map((row) => exp9AuthorityCard(row, role)).join("")
      : `<div class="detail-body empty">No ${role} authorities at this resolution.</div>`;
  };
  renderRoleCards("hub", byId("exp9Hubs"));
  renderRoleCards("bridge", byId("exp9Bridges"));
  renderRoleCards("core", byId("exp9Cores"));

  const pairs = data.bridge_pairs || [];
  byId("exp9BridgePairs").innerHTML = pairs.length
    ? pairs.map(exp9BridgePairView).join("")
    : `<div class="detail-body empty">No bridge pairs at this resolution — bridge authorities are scarce or rare.</div>`;
}

// ---------------------------------------------------------------------------
// EXP MASK — Mask Sensitivity Audit
// ---------------------------------------------------------------------------

async function loadExpMask() {
  const data = await api("/api/exp/mask_sensitivity");
  renderFindings("expMaskFindings", data.findings);

  if (!data.available) {
    ["expMaskIdentity", "expMaskHubs", "expMaskDomains", "expMaskHubList"].forEach(
      (id) => { byId(id).innerHTML = `<p class="empty">${(data.findings || ["No data."])[0]}</p>`; }
    );
    renderResultStory("expMaskResultStory", []);
    return;
  }

  const identityRows = data.identity_masks || [];
  const hubMaskRows  = data.hub_masks || [];

  // Summary cards
  const allIdentity = identityRows.find((r) => r.mask_name === "no_all_identities") || {};
  const strongestId = bestBy(
    identityRows.filter((r) => r.mask_name !== "no_all_identities"),
    (r) => r.accuracy_drop
  );
  const smallestHub = hubMaskRows.length ? hubMaskRows[0] : {};
  const largestHub  = hubMaskRows.length ? hubMaskRows[hubMaskRows.length - 1] : {};

  renderResultStory("expMaskResultStory", [
    {
      kicker: "All identities removed",
      valueText: pct(allIdentity.accuracy_drop) || "—",
      title: `${pct(allIdentity.flip_rate) || "—"} of predictions flip`,
      body: `Mean original-class confidence drop ${number(allIdentity.confidence_drop, 3) || "—"}. A large drop signals the model learnt identity shortcuts.`,
      tone: Number(allIdentity.accuracy_drop) >= 0.02 ? "warn" : "neutral",
    },
    {
      kicker: "Strongest single identity mask",
      valueText: strongestId ? (pct(strongestId.accuracy_drop) || "—") : "—",
      title: strongestId ? strongestId.mask_name : "N/A",
      body: strongestId
        ? `Flip rate ${pct(strongestId.flip_rate) || "—"}. Compare with identity-only AUC from the Audit tab to separate dataset correlation from model reliance.`
        : "No individual identity masks found.",
      tone: Number(strongestId && strongestId.accuracy_drop) >= 0.02 ? "warn" : "neutral",
    },
    {
      kicker: `Top-${smallestHub.n_masked_authorities || "?"} hub mask`,
      valueText: pct(smallestHub.accuracy_drop) || "—",
      title: `Flip rate ${pct(smallestHub.flip_rate) || "—"}`,
      body: `Top-${largestHub.n_masked_authorities || "?"} hub mask drops accuracy by ${pct(largestHub.accuracy_drop) || "—"}. Hub fragility shows how much the model relies on cross-domain legal anchors.`,
      tone: "neutral",
    },
  ]);

  const MASK_COLS = [
    { key: "mask_name",            label: "Mask" },
    { key: "n_masked_authorities", label: "Nodes masked", num: true, format: "int" },
    { key: "masked_edge_share",    label: "Edges masked",  num: true, format: "pct" },
    { key: "baseline_accuracy",    label: "Baseline acc",  num: true, format: "pct" },
    { key: "masked_accuracy",      label: "Masked acc",    num: true, format: "pct" },
    { key: "accuracy_drop",        label: "Acc drop",      num: true, format: "pct" },
    { key: "macro_f1_drop",        label: "F1 drop",       num: true, format: "num", help: "Macro-F1 drop after masking." },
    { key: "confidence_drop",      label: "Orig-conf drop",num: true, format: "num", help: "Mean drop in probability assigned to the original predicted class." },
    { key: "flip_rate",            label: "Flip rate",     num: true, format: "pct", help: "Share of cases whose predicted class changes." },
  ];

  byId("expMaskIdentity").innerHTML = renderSimpleTable(identityRows, MASK_COLS);
  byId("expMaskHubs").innerHTML     = renderSimpleTable(
    hubMaskRows.map((r) => ({ ...r, mask_name: `top-${r.n_masked_authorities || r.top_k_hubs} hubs` })),
    MASK_COLS
  );

  byId("expMaskDomains").innerHTML = renderSimpleTable(data.domain_drops || [], [
    { key: "mask_name",    label: "Mask" },
    { key: "mask_family",  label: "Family" },
    { key: "domain_bucket",label: "Domain" },
    { key: "n_cases",      label: "Cases",    num: true, format: "int" },
    { key: "accuracy_drop",label: "Acc drop", num: true, format: "pct" },
    { key: "macro_f1_drop",label: "F1 drop",  num: true, format: "num" },
    { key: "confidence_drop", label: "Conf drop", num: true, format: "num" },
    { key: "flip_rate",    label: "Flip rate",num: true, format: "pct" },
  ]);

  byId("expMaskHubList").innerHTML = renderSimpleTable(data.hub_authorities || [], [
    { key: "hub_rank",    label: "Rank",   num: true, format: "int" },
    { key: "feature_type",label: "Type" },
    { key: "feature_name",label: "Authority" },
    { key: "n_unique_cases", label: "Citing cases", num: true, format: "int",
      help: "Number of unique cases that cite this authority." },
    { key: "role",        label: "Role" },
  ]);
}

// ---------------------------------------------------------------------------
// AGGREGATE
// ---------------------------------------------------------------------------

async function loadAggregate() {
  const [agg, headlineWrap] = await Promise.all([
    api("/api/exp/aggregate"),
    api("/api/exp_overview"),
  ]);
  renderFindings("aggregateTrends", agg.trends);
  const h = headlineWrap.headline || {};
  renderResultStory("aggregateResultStory", [
    {
      kicker: "Core thesis signal",
      valueText: `${number(h.embedding_nmi, 3) || "—"} NMI`,
      title: "The model representation is not just the structural graph partition.",
      body: `Accuracy ${pct(h.accuracy) || "—"} with weak topology alignment supports the claim that HGT learned outcome-relevant signal beyond shared authorities.`,
      tone: "good",
    },
    {
      kicker: "Explanation validity",
      valueText: h.comp_lift_vs_random ? `${h.comp_lift_vs_random.toFixed(1)}x` : "—",
      title: "Counterfactual explanations beat random removal.",
      body: `Comprehensiveness AUC ${number(h.cf_comprehensiveness_auc, 3) || "—"} vs random baseline; this is the strongest faithfulness readout.`,
      tone: "good",
    },
    {
      kicker: "Audit risk",
      valueText: number(h.identity_shortcut_auc, 3) || pct(h.identity_evidence_share) || "—",
      title: "Identity evidence still matters enough to report.",
      body: `${h.identity_shortcut_scope || "Identity"} shortcut signal should be discussed with the counterfactual flip-rate results, not hidden.`,
      tone: "warn",
    },
    {
      kicker: "Case-level drilldown",
      valueText: number(h.n_cases, 0) || "—",
      title: "Every aggregate claim can be traced back to cases.",
      body: "Use the Case Explorer after reading the aggregate to inspect top evidence, community context, and opposite-case boundaries.",
      tone: "neutral",
    },
  ]);

  byId("aggregateCalibration").innerHTML = `
    <div class="kv-grid">
      <div class="kv"><span>Saved-prediction accuracy</span><strong>${pct(h.accuracy)}</strong></div>
      <div class="kv"><span>Cases with mask flips</span><strong>${pct(h.flip_rate)}</strong></div>
      <div class="kv"><span>Counterfactual sufficiency AUC</span><strong>${number(h.cf_sufficiency_auc, 3)}</strong></div>
      <div class="kv"><span>Counterfactual comp. AUC</span><strong>${number(h.cf_comprehensiveness_auc, 3)}</strong></div>
    </div>
    <p class="panel-note-inline">A model that is both accurate and explanation-faithful should show high accuracy <em>and</em> high comprehensiveness AUC. If accuracy &gt; AUC, the explanation pipeline is the weak link, not the model.</p>`;

  byId("aggregateIdentity").innerHTML = `
    <div class="kv-grid">
      <div class="kv"><span>Identity-evidence importance share</span><strong>${pct(h.identity_evidence_share)}</strong></div>
      <div class="kv"><span>Top identity-only AUC</span><strong>${number(h.identity_shortcut_auc, 3)}</strong></div>
      <div class="kv"><span>All-identity mask acc drop</span><strong>${pct(h.identity_mask_accuracy_drop)}</strong></div>
      <div class="kv"><span>All-identity mask flip rate</span><strong>${pct(h.identity_mask_flip_rate)}</strong></div>
      <div class="kv"><span>Top identity scope</span><strong>${escapeHtml(h.identity_shortcut_scope || "—")}</strong></div>
      <div class="kv"><span>Mean attention/CF overlap</span><strong>${pct(h.attention_overlap)}</strong></div>
    </div>
    <p class="panel-note-inline">If identity (judge/court/parties) carries a large share of mean importance, has strong identity-only AUC, or causes a material post-hoc mask drop, the model may be leaning on <em>who</em> rather than <em>what law</em>. Confirm with the Identity Shortcut tab and identity-heavy cases in the Case Explorer.</p>`;

  byId("aggregateTopology").innerHTML = `
    <div class="kv-grid kv-grid-3">
      <div class="kv"><span>Embedding NMI vs structure</span><strong>${number(h.embedding_nmi, 3)}</strong></div>
      <div class="kv"><span>Embedding ARI</span><strong>${number(h.embedding_ari, 3)}</strong></div>
      <div class="kv"><span>HDBSCAN noise rate</span><strong>${pct(h.embedding_noise_rate)}</strong></div>
      <div class="kv"><span>Communities found</span><strong>${number(h.n_communities, 0)}</strong></div>
      <div class="kv"><span>Top-50 hub mask acc drop</span><strong>${pct(h.hub_top50_accuracy_drop)}</strong></div>
      <div class="kv"><span>Comprehensiveness lift over random</span><strong>${h.comp_lift_vs_random ? `${h.comp_lift_vs_random.toFixed(1)}×` : "—"}</strong></div>
      <div class="kv"><span>Cases analyzed</span><strong>${number(h.n_cases, 0)}</strong></div>
    </div>
    <p class="panel-note-inline">Low NMI + high accuracy + high comprehensiveness lift = the HGT learned outcome-relevant signal beyond pure citation topology. That is the core thesis claim — confirm here, then drill into specific embedding clusters in Exp 5.</p>`;
}

// ---------------------------------------------------------------------------
// CASE EXPLORER (preserved from prior version, lightly polished)
// ---------------------------------------------------------------------------

function caseQuery() {
  const params = new URLSearchParams();
  params.set("page", state.casePage);
  params.set("limit", state.caseLimit);
  params.set("q", byId("caseSearch").value);
  params.set("target", byId("targetFilter").value);
  params.set("pred", byId("predFilter").value);
  params.set("correct", byId("correctFilter").value);
  params.set("sort", byId("caseSort").value);
  return params;
}

async function loadCases() {
  const data = await api(`/api/cases?${caseQuery().toString()}`);
  byId("casePage").textContent = `Page ${data.page} / ${Math.max(1, Math.ceil(data.total / data.limit))} (${number(data.total, 0)} cases)`;
  byId("caseList").innerHTML = renderSimpleTable(data.rows, [
    { key: "case_index", label: "Index", num: true, format: "num", digits: 0 },
    { key: "case_id", label: "Case" },
    { key: "target_label", label: "Target", format: "label" },
    { key: "baseline_pred_label", label: "Pred", format: "label" },
    { key: "baseline_pred_proba", label: "Conf.", num: true, format: "num", help: "Probability assigned to the predicted class." },
    { key: "max_abs_delta_pred_proba", label: "Top |Δp|", num: true, format: "num" },
    { key: "top_evidence_type", label: "Top type" },
    { key: "top_evidence_name", label: "Top evidence" },
    { key: "n_prediction_flips", label: "Flips", num: true, format: "num", digits: 0 },
  ], { clickable: true, barColumn: "max_abs_delta_pred_proba" });
  document.querySelectorAll("#caseList tr.clickable").forEach((row) => {
    row.classList.toggle("selected", String(row.dataset.caseIndex) === String(state.selectedCase));
    row.addEventListener("click", () => {
      document.querySelectorAll("#caseList tr.clickable").forEach((other) => other.classList.remove("selected"));
      row.classList.add("selected");
      loadCaseDetail(row.dataset.caseIndex);
    });
  });
  if (!state.selectedCase) {
    byId("caseDetail").className = "detail-body empty";
    byId("caseDetail").innerHTML = "Select a case row.";
    byId("caseEvidenceInspector").className = "detail-body empty";
    byId("caseEvidenceInspector").innerHTML = "Select an explanation row.";
  }
}

async function loadCaseDetail(caseIndex) {
  state.selectedCase = caseIndex;
  const target = byId("caseDetail");
  target.className = "detail-body empty";
  target.innerHTML = `Loading case ${escapeHtml(caseIndex)}…`;
  byId("caseEvidenceInspector").className = "detail-body empty";
  byId("caseEvidenceInspector").innerHTML = "Loading evidence inspector…";
  const data = await api(`/api/case?case_index=${encodeURIComponent(caseIndex)}`);
  const summary = data.summary;
  if (!summary) {
    target.className = "detail-body empty";
    target.innerHTML = "No case selected";
    byId("caseEvidenceInspector").className = "detail-body empty";
    byId("caseEvidenceInspector").innerHTML = "Select an explanation row above.";
    return;
  }
  target.className = "detail-body";
  const attention = data.attention || {};
  const pattern = data.pattern || {};
  const community = pattern.community || {};
  const nearest = pattern.nearest_opposite || {};
  const cluster = pattern.embedding_cluster || {};
  const top = (data.top_explanations || [])[0] || {};
  const topDelta = Number(top.delta_pred_proba || 0);
  const direction = topDelta >= 0
    ? "Removing the top group reduced confidence in the original prediction."
    : "Removing the top group increased confidence in the original prediction.";
  target.innerHTML = `
    <div class="case-title">
      <h3>${escapeHtml(summary.case_id)}</h3>
      <div class="case-meta">
        <span class="pill">Index ${escapeHtml(summary.case_index)}</span>
        <span class="pill">Target ${escapeHtml(summary.target_label)}</span>
        <span class="pill">Pred ${escapeHtml(summary.baseline_pred_label)}</span>
        <span class="pill">Top |Δp| ${number(summary.max_abs_delta_pred_proba, 4)}</span>
      </div>
      <p class="case-reading">${escapeHtml(direction)} Top evidence: <strong>${escapeHtml(top.evidence_name || summary.top_evidence_name)}</strong> via ${displayPath(top.path_family || summary.top_path_family)}.</p>
    </div>
    <div class="detail-grid">
      <div class="detail-stat"><div class="small">Prediction confidence</div><div class="value">${number(summary.baseline_pred_proba, 4)}</div></div>
      <div class="detail-stat"><div class="small">Label 1 probability</div><div class="value">${number(summary.baseline_positive_proba, 4)}</div></div>
      <div class="detail-stat"><div class="small">Local graph edges</div><div class="value">${number(summary.rf_edge_count, 0)}</div></div>
      <div class="detail-stat"><div class="small">Attention agreement</div><div class="value">${pct(attention.counterfactual_attention_overlap)}</div></div>
    </div>
    ${community.community_id !== undefined ? `
      <div class="pattern-card">
        <div class="pattern-reading">
          <strong>Pattern context.</strong>
          Community ${escapeHtml(community.community_id)} (n=${number(community.community_size, 0)}, ${escapeHtml(community.community_dominant_domain_bucket)}, dominant label ${escapeHtml(community.community_dominant_label)}, accuracy ${pct(community.community_accuracy)}).
        </div>
        <div class="detail-grid compact-grid">
          <div class="detail-stat"><div class="small">Embedding cluster</div><div class="value">${escapeHtml(cluster.embedding_cluster_id ?? "")}</div></div>
          <div class="detail-stat"><div class="small">Nearest-opposite cosine</div><div class="value">${number(nearest.cosine_similarity, 4)}</div></div>
        </div>
        ${nearest.nearest_opposite_case_index !== undefined ? `
          <p class="case-reading">
            Closest opposite-label training case:
            <strong>${escapeHtml(nearest.nearest_opposite_case_index)}</strong> (label ${escapeHtml(nearest.nearest_opposite_target_label)}).<br>
            Case-only evidence: ${escapeHtml(nearest.top_query_only_features || "none")}.<br>
            Opposite-only evidence: ${escapeHtml(nearest.top_opposite_only_features || "none")}.
          </p>` : ""}
      </div>` : ""}
    ${renderLocalExplanationSubgraph(summary, data.local_graph || data.top_explanations || [])}
    ${renderSimpleTable(data.top_explanations, [
      { key: "group_rank_abs", label: "Rank", num: true, format: "num", digits: 0 },
      { key: "evidence_type", label: "Type" },
      { key: "evidence_name", label: "Evidence" },
      { key: "path_family", label: "Path", format: "path" },
      { key: "delta_pred_proba", label: "Signed Δp", num: true, format: "num" },
      { key: "abs_delta_pred_proba", label: "|Δp|", num: true, format: "num" },
      { key: "support_train_n", label: "Train support", num: true, format: "num", digits: 0 },
      { key: "support_positive_rate", label: "Label 1 rate", num: true, format: "pct" },
      { key: "attention_score", label: "Attn", num: true, format: "num" },
    ], { barColumn: "abs_delta_pred_proba", clickableEvidence: true })}`;

  document.querySelectorAll("#caseDetail tr.evidence-row").forEach((row) => {
    row.addEventListener("click", () => {
      document.querySelectorAll("#caseDetail tr.evidence-row").forEach((o) => o.classList.remove("selected"));
      row.classList.add("selected");
      loadEvidenceDetail(data.top_explanations[Number(row.dataset.rowIndex)], "caseEvidenceInspector");
    });
  });
  if (data.top_explanations && data.top_explanations.length) {
    const firstRow = document.querySelector("#caseDetail tr.evidence-row");
    if (firstRow) firstRow.classList.add("selected");
    loadEvidenceDetail(data.top_explanations[0], "caseEvidenceInspector");
  }
}

function evidenceParams(row) {
  const params = new URLSearchParams();
  ["evidence_type", "evidence_global_index", "evidence_id", "evidence_name", "relation_types", "path_family"].forEach((key) => {
    const raw = row ? row[key] : "";
    if (raw !== null && raw !== undefined && raw !== "" && raw !== "nan") params.set(key, raw);
  });
  return params;
}

function searchUrl(name, type) {
  const text = String(value(name)).trim();
  if (!text || type === "arguments" || type === "relation_type") return "";
  const suffix = type === "precedent" ? " case judgment" : " Indian law";
  return `https://www.google.com/search?q=${encodeURIComponent(text + suffix)}`;
}

async function loadEvidenceDetail(row, targetId) {
  const target = byId(targetId);
  if (!row || !target) return;
  target.className = "detail-body";
  target.innerHTML = `<div class="inspector-loading">Loading evidence detail…</div>`;
  try {
    const data = await api(`/api/evidence_detail?${evidenceParams(row).toString()}`);
    renderEvidenceDetail(data, targetId);
  } catch (e) {
    target.className = "detail-body empty";
    target.innerHTML = escapeHtml(e.message);
  }
}

function renderEvidenceDetail(data, targetId) {
  const target = byId(targetId);
  const support = data.support || {};
  const evidenceName = data.evidence_name || support.evidence_name || "";
  const evidenceType = data.evidence_type || support.evidence_type || "";
  const rawId = data.evidence_id || support.evidence_id || "";
  const globalIndex = data.evidence_global_index || support.evidence_global_index || "";
  const supportN = support.support_train_n;
  const positiveRate = support.support_positive_rate;
  const negativeRate = support.support_negative_rate;
  const link = searchUrl(evidenceName, evidenceType);
  const notes = data.relation_notes || [];
  const supportText = supportN
    ? `${number(supportN, 0)} connected training cases: ${pct(positiveRate)} label 1, ${pct(negativeRate)} label -1.`
    : evidenceType === "relation_type"
      ? "Relation-type rows have no single training-neighbourhood support node."
      : "No connected training-neighbourhood support row was found.";

  target.innerHTML = `
    <div class="inspector">
      <div class="case-title compact">
        <h3>${escapeHtml(evidenceName || data.relation_types || "Evidence")}</h3>
        <div class="case-meta">
          <span class="pill">${escapeHtml(evidenceType || "unknown type")}</span>
          ${globalIndex ? `<span class="pill">Graph index ${escapeHtml(globalIndex)}</span>` : ""}
          ${link ? `<a class="external-link" href="${escapeHtml(link)}" target="_blank" rel="noreferrer">Search source</a>` : ""}
        </div>
      </div>
      <div class="inspector-section">
        <h4>What This Row Is</h4>
        <p>${escapeHtml(supportText)}</p>
        ${data.path_family ? `<p><strong>Path:</strong> ${displayPath(data.path_family)}</p>` : ""}
        ${data.relation_types ? `<p><strong>Typed edges:</strong> <code>${escapeHtml(data.relation_types)}</code></p>` : ""}
      </div>
      <div class="detail-grid inspector-grid">
        <div class="detail-stat"><div class="small">Train support</div><div class="value">${number(supportN, 0)}</div></div>
        <div class="detail-stat"><div class="small">Label 1 rate</div><div class="value">${pct(positiveRate)}</div></div>
        <div class="detail-stat"><div class="small">Label -1 rate</div><div class="value">${pct(negativeRate)}</div></div>
        <div class="detail-stat"><div class="small">Cases shown</div><div class="value">${number((data.cases || []).length, 0)}</div></div>
      </div>
      <div class="inspector-section">
        <h4>Raw Graph Node</h4>
        <div class="raw-id"><span>evidence_id</span><code>${escapeHtml(rawId || "n/a for relation-type rows")}</code></div>
        <div class="raw-id"><span>evidence_name</span><code>${escapeHtml(evidenceName || "n/a")}</code></div>
      </div>
      ${notes.length ? `
        <div class="inspector-section relation-notes">
          <h4>Relation Notes</h4>
          ${notes.map((n) => `<p>${escapeHtml(n)}</p>`).join("")}
        </div>` : ""}
      <div class="inspector-section">
        <h4>Test Cases Where This Evidence Mattered</h4>
        <p class="small">Sorted by importance.</p>
      </div>
      ${renderSimpleTable(data.cases || [], [
        { key: "case_index", label: "Case index", num: true, format: "num", digits: 0 },
        { key: "case_id", label: "Case" },
        { key: "target_label", label: "Target", format: "label" },
        { key: "baseline_pred_label", label: "Pred", format: "label" },
        { key: "path_family", label: "Path", format: "path" },
        { key: "delta_pred_proba", label: "Signed Δp", num: true, format: "num" },
        { key: "abs_delta_pred_proba", label: "|Δp|", num: true, format: "num" },
      ], { barColumn: "abs_delta_pred_proba" })}
    </div>`;
}

// ---------------------------------------------------------------------------
// RAW TABLES
// ---------------------------------------------------------------------------

async function loadGenericTable() {
  const params = new URLSearchParams();
  params.set("name", state.tableName);
  params.set("q", byId("tableSearch").value);
  params.set("limit", "200");
  const data = await api(`/api/table?${params.toString()}`);
  const rows = data.rows;
  if (!rows.length) {
    byId("genericTable").innerHTML = `<div class="detail-body empty">No rows for ${escapeHtml(state.tableName)}.</div>`;
    return;
  }
  const preferred = Object.keys(rows[0]).slice(0, 12).map((key) => ({
    key,
    label: COLUMN_LABELS[key] || key,
    num: typeof rows[0][key] === "number",
    format: key === "path_family" ? "path"
      : key === "relation" || key === "relation_type" ? "relation"
      : (key.endsWith("_rate") || key.endsWith("_share") || key.includes("overlap") || key === "flip_rate" || key.endsWith("_flip_rate")) ? "pct"
      : typeof rows[0][key] === "number" ? "num" : undefined,
  }));
  byId("genericTable").innerHTML = renderSimpleTable(rows, preferred, {
    barColumn: rows[0].sum_abs_delta_pred_proba !== undefined ? "sum_abs_delta_pred_proba" : undefined,
  });
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function wireEvents() {
  document.querySelectorAll(".tab").forEach((b) => {
    b.addEventListener("click", () => {
      const target = setView(b.dataset.view);
      loadView(target);
    });
  });
  document.querySelectorAll("#experimentMap a[data-jump]").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const target = a.dataset.jump;
      const viewName = setView(target);
      loadView(viewName);
    });
  });
  document.body.addEventListener("click", async (e) => {
    const openCase = e.target.closest("[data-open-case]");
    if (openCase) {
      e.preventDefault();
      const targetCase = openCase.dataset.openCase;
      const viewName = setView("cases");
      await loadView(viewName);
      await loadCaseDetail(targetCase);
      return;
    }
    const link = e.target.closest("[data-jump]");
    if (!link) return;
    e.preventDefault();
    const target = link.dataset.jump;
    const viewName = setView(target);
    loadView(viewName);
  });

  window.addEventListener("hashchange", () => {
    const target = cleanViewName(location.hash.slice(1));
    setView(target, { updateHash: false });
    loadView(target);
  });

  ["caseSearch", "targetFilter", "predFilter", "correctFilter", "caseSort"].forEach((id) => {
    byId(id).addEventListener("input", () => {
      state.casePage = 1;
      state.selectedCase = null;
      loadCases();
    });
  });
  byId("prevCases").addEventListener("click", () => { state.casePage = Math.max(1, state.casePage - 1); loadCases(); });
  byId("nextCases").addEventListener("click", () => { state.casePage += 1; loadCases(); });

  byId("exp6Search").addEventListener("input", () => {
    if (state.exp6Pairs) loadExp6Pairs(state.exp6Pairs);
  });
  byId("exp6LoadCase").addEventListener("click", () => loadOppositeGraphForCase(byId("exp6CasePick").value));
  byId("exp6CasePick").addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadOppositeGraphForCase(byId("exp6CasePick").value);
  });

  byId("tableSelect").addEventListener("input", (e) => {
    state.tableName = e.target.value;
    loadGenericTable();
  });
  byId("tableSearch").addEventListener("input", loadGenericTable);
}

async function loadView(name) {
  if (state.loaded[name] && name !== "cases" && name !== "tables") return;
  try {
    if (name === "overview") await loadOverview();
    else if (name === "exp1") await loadExp1();
    else if (name === "exp2") await loadExp2();
    else if (name === "exp10") await loadExp10();
    else if (name === "exp3") await loadExp3();
    else if (name === "exp4") await loadExp4();
    else if (name === "exp5") await loadExp5();
    else if (name === "exp6") await loadExp6();
    else if (name === "exp7") await loadExp7();
    else if (name === "exp8") await loadExp8();
    else if (name === "exp9") await loadExp9();
    else if (name === "expMask") await loadExpMask();
    else if (name === "aggregate") await loadAggregate();
    else if (name === "cases") await loadCases();
    else if (name === "tables") await loadGenericTable();
    state.loaded[name] = true;
  } catch (err) {
    const target = document.querySelector(`#${name}`) || document.body;
    const note = document.createElement("div");
    note.className = "detail-body empty";
    note.textContent = `Failed to load ${name}: ${err.message}`;
    target.prepend(note);
  }
}

async function boot() {
  wireEvents();
  const initialView = cleanViewName(location.hash.slice(1));
  setView(initialView, { updateHash: false, scroll: false });
  await loadView(initialView);
}

boot().catch((err) => {
  document.body.innerHTML = `<main><section class="panel"><div class="detail-body empty">${escapeHtml(err.message)}</div></section></main>`;
});
