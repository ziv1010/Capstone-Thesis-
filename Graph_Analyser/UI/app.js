document.addEventListener('DOMContentLoaded', () => {
    const caseListEl = document.getElementById('case-list');
    const searchInput = document.getElementById('search-input');
    const analyticsButton = document.getElementById('analytics-button');

    const welcomeState = document.getElementById('welcome-state');
    const loadingState = document.getElementById('loading-state');
    const detailState = document.getElementById('detail-state');
    const analyticsState = document.getElementById('analytics-state');
    const loadingIndicatorEl = document.getElementById('loading-indicator');
    const loadingCopyEl = document.getElementById('loading-copy');
    const analyticsContentEl = document.getElementById('analytics-content');

    const detailCaseId = document.getElementById('detail-case-id');
    const badgePred = document.getElementById('badge-prediction');
    const badgeTarget = document.getElementById('badge-target');
    const badgeConf = document.getElementById('badge-confidence');
    const badgeCorrect = document.getElementById('badge-correctness');

    const pipelineStagesEl = document.getElementById('pipeline-stages');
    const summaryExplanationEl = document.getElementById('summary-explanation');
    const precedentsListEl = document.getElementById('precedents-list');
    const similarCasesListEl = document.getElementById('similar-cases-list');

    const detailTabs = Array.from(document.querySelectorAll('.detail-tab'));
    const detailPanels = {
        stages: document.getElementById('detail-panel-stages'),
        summary: document.getElementById('detail-panel-summary'),
        graph: document.getElementById('detail-panel-graph')
    };

    const graphDescriptionEl = document.getElementById('graph-description');
    const graphLegendEl = document.getElementById('graph-legend');
    const graphVisualizationEl = document.getElementById('graph-visualization');
    const graphTooltipEl = document.getElementById('graph-tooltip');
    const graphStatsEl = document.getElementById('graph-stats');
    const graphBreakdownEl = document.getElementById('graph-breakdown');
    const graphInsightsEl = document.getElementById('graph-insights');

    const GROUP_COLORS = {
        precedent: '#f6c667',
        statute: '#63c8ff',
        provision: '#7ce38b',
        arguments: '#ff9575',
        petitioner_arguments: '#c295ff',
        respondent_arguments: '#ff89c2',
        other_lawyer_arguments: '#9aa7b8',
        similar_training_cases: '#59e1d5'
    };

    const GROUP_LABELS = {
        precedent: 'Precedent',
        statute: 'Statute',
        provision: 'Provision',
        arguments: 'Argument',
        petitioner_arguments: 'Petitioner Arg.',
        respondent_arguments: 'Respondent Arg.',
        other_lawyer_arguments: 'Other Counsel Arg.',
        similar_training_cases: 'Similar Case'
    };

    let allCases = [];
    let activeTab = 'stages';
    let analyticsLoaded = false;
    let analyticsPromise = null;

    detailTabs.forEach(button => {
        button.addEventListener('click', () => setActiveTab(button.dataset.tab));
    });

    analyticsButton.addEventListener('click', () => showAnalytics());

    searchInput.addEventListener('input', (event) => {
        const query = event.target.value.toLowerCase();
        const filtered = allCases.filter(caseItem =>
            caseItem.case_id.toLowerCase().includes(query) ||
            caseItem.case_node_index.toString().includes(query)
        );
        renderCaseList(filtered);
    });

    async function init() {
        try {
            const response = await fetch('../outputs/phase5_llm_reasoning/index.json');
            if (!response.ok) throw new Error('Failed to load index.json');
            allCases = await response.json();
            renderCaseList(allCases);
        } catch (error) {
            caseListEl.innerHTML = `<div class="loading error">Error loading cases: ${escapeHtml(error.message)}.<br>Make sure you are running via a web server.</div>`;
            console.error(error);
        }
    }

    function setActiveTab(tabName) {
        if (!detailPanels[tabName]) return;
        activeTab = tabName;
        detailTabs.forEach(button => {
            const isActive = button.dataset.tab === tabName;
            button.classList.toggle('active', isActive);
            button.setAttribute('aria-selected', String(isActive));
        });
        Object.entries(detailPanels).forEach(([name, panel]) => {
            panel.classList.toggle('hidden', name !== tabName);
        });
    }

    function renderCaseList(cases) {
        caseListEl.innerHTML = '';
        if (cases.length === 0) {
            caseListEl.innerHTML = '<div class="loading">No cases found.</div>';
            return;
        }
        cases.forEach(caseItem => {
            const li = document.createElement('li');
            li.className = 'case-item';
            li.dataset.id = caseItem.case_node_index;
            const isWin = caseItem.predicted_label === '1';
            const isCorrect = caseItem.predicted_label === caseItem.target_label;
            li.innerHTML = `
                <div class="case-item-title">${escapeHtml(caseItem.case_id)}</div>
                <div class="case-item-meta">
                    <span class="label-dot ${isWin ? 'win' : 'loss'}"></span>
                    <span>Pred: ${escapeHtml(caseItem.predicted_label)} | Target: ${escapeHtml(caseItem.target_label)}</span>
                    ${isCorrect ? '✅' : '❌'}
                </div>
            `;
            li.addEventListener('click', () => loadCaseDetails(caseItem.case_node_index, li));
            caseListEl.appendChild(li);
        });
    }

    async function loadCaseDetails(nodeIndex, listItemElement) {
        document.querySelectorAll('.case-item').forEach(el => el.classList.remove('active'));
        if (listItemElement) listItemElement.classList.add('active');

        welcomeState.classList.add('hidden');
        analyticsState.classList.add('hidden');
        detailState.classList.add('hidden');
        showLoading('Analyzing data...');

        try {
            const [explRes, caseRes] = await Promise.all([
                fetch(`../outputs/phase5_llm_reasoning/explanation_${nodeIndex}.json`),
                fetch(`../outputs/phase4_explanations/cases/case_${nodeIndex}.json`).catch(() => null)
            ]);

            if (!explRes.ok) throw new Error('Explanation not found');
            const explData = await explRes.json();
            const caseData = (caseRes && caseRes.ok) ? await caseRes.json() : null;

            renderDetails(explData, caseData);
            loadingState.classList.add('hidden');
            detailState.classList.remove('hidden');
            setActiveTab(activeTab);
        } catch (error) {
            console.error(error);
            showLoading(`Error loading case details: ${error.message}`, true);
        }
    }

    function renderDetails(explData, caseData) {
        detailCaseId.textContent = explData.case_id;

        setupBadge(badgePred, `Pred: ${formatOutcomeLabel(explData.predicted_label)}`,
            explData.predicted_label === '1' ? 'success' : 'danger');
        setupBadge(badgeTarget, `Target: ${formatOutcomeLabel(explData.target_label)}`,
            explData.target_label === '1' ? 'success' : 'danger');
        setupBadge(badgeConf, `Conf: ${formatPercent(explData.confidence, 2)}`, 'info');

        const isCorrect = explData.predicted_label === explData.target_label;
        setupBadge(badgeCorrect, isCorrect ? '✓ Correct' : '✗ Incorrect', isCorrect ? 'success' : 'danger');

        renderPipelineStages(explData, caseData);
        renderSummaryPanel(explData, caseData);
        renderGraphView(explData, caseData);
    }

    // ============ PIPELINE STAGES ============
    function renderPipelineStages(explData, caseData) {
        const stages = [
            buildStage1Inference(explData, caseData),
            buildStage2CaseInput(caseData),
            buildStage3GraphExplainer(caseData),
            buildStage4Retrieval(caseData),
            buildStage5LLMReasoning(explData)
        ];
        pipelineStagesEl.innerHTML = stages.join('');
    }

    function buildStage1Inference(explData, caseData) {
        const probs = caseData?.class_probabilities || {};
        const winProb = probs['1'] ?? (explData.predicted_label === '1' ? explData.confidence : 1 - (explData.confidence || 0));
        const lossProb = probs['-1'] ?? (1 - winProb);
        const isCorrect = explData.predicted_label === explData.target_label;

        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">1</span>
                    <div>
                        <h3>Inference (Phase 1–2)</h3>
                        <p class="stage-subtitle">GNN forward pass over the heterogeneous case graph.</p>
                    </div>
                </div>
                <div class="stage-body">
                    <div class="prob-bars">
                        ${buildProbBar('Win (label 1)', winProb, 'win')}
                        ${buildProbBar('Loss (label -1)', lossProb, 'loss')}
                    </div>
                    <div class="stage-row">
                        <div class="stage-metric"><span>Prediction</span><strong class="${explData.predicted_label === '1' ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(explData.predicted_label))}</strong></div>
                        <div class="stage-metric"><span>Ground Truth</span><strong class="${explData.target_label === '1' ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(explData.target_label))}</strong></div>
                        <div class="stage-metric"><span>Confidence</span><strong>${escapeHtml(formatPercent(explData.confidence, 2))}</strong></div>
                        <div class="stage-metric"><span>Outcome</span><strong class="${isCorrect ? 'pos' : 'neg'}">${isCorrect ? 'Correct' : 'Incorrect'}</strong></div>
                    </div>
                </div>
            </article>
        `;
    }

    function buildStage2CaseInput(caseData) {
        const text = caseData?.target_case_text || {};
        const snippets = [
            { label: 'Preamble', body: text.preamble },
            { label: 'Facts', body: text.facts },
            { label: 'Arguments', body: text.arguments },
            { label: 'Other-Counsel Arguments', body: text.other_lawyer_arguments }
        ].filter(item => item.body && item.body.trim());

        const content = snippets.length
            ? snippets.map(item => `
                <div class="text-snippet">
                    <div class="snippet-label">${escapeHtml(item.label)}</div>
                    <div class="snippet-body">${escapeHtml(truncate(item.body, 600))}</div>
                </div>
            `).join('')
            : '<p class="empty-copy">No case text was persisted for this record.</p>';

        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">2</span>
                    <div>
                        <h3>Case Input (Phase 4 — Source Text)</h3>
                        <p class="stage-subtitle">Raw text segments that anchor the case node in the graph.</p>
                    </div>
                </div>
                <div class="stage-body">
                    ${content}
                </div>
            </article>
        `;
    }

    function buildStage3GraphExplainer(caseData) {
        const top = caseData?.top_nodes || {};
        const raw = caseData?._raw_argument_top_nodes || {};
        const categories = [
            { key: 'precedent', items: top.precedent },
            { key: 'statute', items: top.statute },
            { key: 'provision', items: top.provision },
            { key: 'arguments', items: raw.arguments },
            { key: 'petitioner_arguments', items: raw.petitioner_arguments },
            { key: 'respondent_arguments', items: raw.respondent_arguments },
            { key: 'other_lawyer_arguments', items: raw.other_lawyer_arguments }
        ].filter(cat => Array.isArray(cat.items) && cat.items.length > 0);

        if (!categories.length) {
            return `
                <article class="stage-card">
                    <div class="stage-header">
                        <span class="stage-number">3</span>
                        <div>
                            <h3>Graph Explainer (Phase 3–4)</h3>
                            <p class="stage-subtitle">Subgraph importance scores from the trained PGExplainer.</p>
                        </div>
                    </div>
                    <div class="stage-body">
                        <p class="empty-copy">No top-scored nodes were returned by the explainer for this case.</p>
                    </div>
                </article>
            `;
        }

        const blocks = categories.map(cat => `
            <div class="explainer-category">
                <div class="explainer-category-head">
                    <span class="explainer-swatch" style="background:${GROUP_COLORS[cat.key]}"></span>
                    <span class="explainer-category-label">${escapeHtml(GROUP_LABELS[cat.key])}</span>
                    <span class="explainer-category-count">${cat.items.length}</span>
                </div>
                <div class="explainer-items">
                    ${cat.items.map(item => buildExplainerItem(item, cat.key)).join('')}
                </div>
            </div>
        `).join('');

        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">3</span>
                    <div>
                        <h3>Graph Explainer (Phase 3–4)</h3>
                        <p class="stage-subtitle">Nodes scored by the trained PGExplainer as most influential to the prediction.</p>
                    </div>
                </div>
                <div class="stage-body explainer-grid">
                    ${blocks}
                </div>
            </article>
        `;
    }

    function buildExplainerItem(item, groupKey) {
        const text = normalizeNodeText(item.text);
        const imp = Number.isFinite(item.importance) ? item.importance : 0;
        const width = Math.min(100, Math.max(6, imp * 220));
        const edge = item.edge_type ? `<span class="edge-tag">${escapeHtml(item.edge_type)}</span>` : '';
        return `
            <div class="explainer-item">
                <div class="explainer-item-title">${escapeHtml(text)}</div>
                <div class="explainer-item-meta">
                    <span>Node #${escapeHtml(String(item.node_index ?? '-'))}</span>
                    ${edge}
                </div>
                <div class="importance-wrapper">
                    <div class="importance-bar-container">
                        <div class="importance-bar" style="width:${width}%; background:${GROUP_COLORS[groupKey] || '#58a6ff'}"></div>
                    </div>
                    <div class="importance-val">${formatPercent(imp, 1)}</div>
                </div>
            </div>
        `;
    }

    function buildStage4Retrieval(caseData) {
        const argCtx = caseData?.similar_case_argument_context || [];
        const similar = caseData?.similar_training_cases || [];

        const argCtxHtml = argCtx.length
            ? argCtx.map(ctx => `
                <div class="arg-ctx-item">
                    <div class="arg-ctx-head">
                        <strong>${escapeHtml(ctx.owning_case)}</strong>
                        <span class="pill info">${escapeHtml(formatPercent(ctx.importance, 1))}</span>
                    </div>
                    <div class="arg-ctx-meta">${escapeHtml(ctx.argument_section || '')}</div>
                    <div class="arg-ctx-snippet">${escapeHtml(truncate(ctx.snippet || '', 320))}</div>
                </div>
            `).join('')
            : '<p class="empty-copy">No argument context from cited cases.</p>';

        const similarHtml = similar.length
            ? similar.map(sc => {
                const isWin = sc.target_label === '1';
                return `
                    <div class="retrieval-item">
                        <div class="retrieval-rank">#${escapeHtml(String(sc.rank))}</div>
                        <div class="retrieval-body">
                            <div class="retrieval-title">${escapeHtml(sc.case_id)}</div>
                            <div class="retrieval-meta">
                                <span class="pill ${isWin ? 'success' : 'danger'}">Actual: ${escapeHtml(formatOutcomeLabel(sc.target_label))}</span>
                                <span class="pill ${sc.predicted_label === '1' ? 'success' : 'danger'}">Pred: ${escapeHtml(formatOutcomeLabel(sc.predicted_label))}</span>
                                <span class="pill info">Sim ${escapeHtml(formatPercent(sc.similarity, 2))}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('')
            : '<p class="empty-copy">No similar training cases.</p>';

        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">4</span>
                    <div>
                        <h3>Retrieval & Argument Context</h3>
                        <p class="stage-subtitle">Nearest training analogues (FAISS cosine) and attributed argument snippets.</p>
                    </div>
                </div>
                <div class="stage-body stage-split">
                    <div class="stage-col">
                        <h4 class="stage-col-title">Similar Training Cases</h4>
                        ${similarHtml}
                    </div>
                    <div class="stage-col">
                        <h4 class="stage-col-title">Argument Snippets From Retrieved Cases</h4>
                        ${argCtxHtml}
                    </div>
                </div>
            </article>
        `;
    }

    function buildStage5LLMReasoning(explData) {
        const body = typeof marked !== 'undefined'
            ? marked.parse(explData.explanation || '*No explanation provided.*')
            : escapeHtml(explData.explanation || 'No explanation provided.');
        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">5</span>
                    <div>
                        <h3>LLM Reasoning (Phase 5)</h3>
                        <p class="stage-subtitle">Natural-language justification grounded in the explainer evidence.</p>
                    </div>
                </div>
                <div class="stage-body markdown-body">${body}</div>
            </article>
        `;
    }

    function buildProbBar(label, prob, tone) {
        const pct = Math.max(0, Math.min(1, prob || 0)) * 100;
        return `
            <div class="prob-row">
                <div class="prob-label">${escapeHtml(label)}</div>
                <div class="prob-track">
                    <div class="prob-fill ${tone}" style="width:${pct.toFixed(1)}%"></div>
                </div>
                <div class="prob-val">${pct.toFixed(2)}%</div>
            </div>
        `;
    }

    // ============ SUMMARY PANEL ============
    function renderSummaryPanel(explData, caseData) {
        if (typeof marked !== 'undefined') {
            summaryExplanationEl.innerHTML = marked.parse(explData.explanation || '*No explanation provided.*');
        } else {
            summaryExplanationEl.textContent = explData.explanation || 'No explanation provided.';
        }
        renderPrecedents(caseData);
        renderSimilarCases(caseData);
    }

    function renderPrecedents(caseData) {
        precedentsListEl.innerHTML = '';
        const items = [
            ...(caseData?.top_nodes?.precedent || []),
            ...(caseData?._raw_argument_top_nodes?.arguments || [])
        ];
        if (items.length === 0) {
            precedentsListEl.innerHTML = '<p class="empty-copy">No top precedents or argument clusters extracted.</p>';
            return;
        }
        items.forEach(item => {
            const title = normalizeNodeText(item.text);
            const metaPrefix = item.text && item.text.includes('::') ? 'Argument Node' : 'Node';
            precedentsListEl.appendChild(createNodeCard(title, item.importance, `${metaPrefix} ${item.node_index}`));
        });
    }

    function renderSimilarCases(caseData) {
        similarCasesListEl.innerHTML = '';
        if (!caseData?.similar_training_cases?.length) {
            similarCasesListEl.innerHTML = '<p class="empty-copy">No similar cases found.</p>';
            return;
        }
        caseData.similar_training_cases.forEach(sc => {
            const isWin = sc.target_label === '1';
            const simPct = formatPercent(sc.similarity, 1);
            const card = document.createElement('div');
            card.className = 'node-item';
            card.innerHTML = `
                <div class="node-item-title">${escapeHtml(sc.case_id)}</div>
                <div class="node-item-meta" style="margin-bottom:8px;">
                    <span style="display:flex;align-items:center;gap:6px;">
                        <span class="label-dot ${isWin ? 'win' : 'loss'}"></span>
                        Outcome: ${escapeHtml(formatOutcomeLabel(sc.target_label))}
                    </span>
                    <span>Sim: ${simPct}</span>
                </div>
                <div class="node-item-meta">
                    <span>Model: ${escapeHtml(formatOutcomeLabel(sc.predicted_label))}</span>
                </div>
            `;
            similarCasesListEl.appendChild(card);
        });
    }

    // ============ SVG GRAPH ============
    function renderGraphView(explData, caseData) {
        const nodes = collectGraphNodes(caseData);
        graphDescriptionEl.textContent = nodes.length
            ? 'Node-link view. Edge thickness ≈ importance; dashed edges = embedding-retrieval links.'
            : 'No structured graph links available for this case.';
        graphLegendEl.innerHTML = buildGraphLegend(nodes);
        graphVisualizationEl.innerHTML = buildGraphSVG(explData, nodes);
        attachGraphInteractions();

        renderGraphSide(explData, nodes);
    }

    function collectGraphNodes(caseData) {
        const out = [];
        const top = caseData?.top_nodes || {};
        const raw = caseData?._raw_argument_top_nodes || {};

        const legalSources = [
            { key: 'precedent', items: (top.precedent || []).slice(0, 4), section: 'legal' },
            { key: 'statute', items: (top.statute || []).slice(0, 3), section: 'legal' },
            { key: 'provision', items: (top.provision || []).slice(0, 3), section: 'legal' }
        ];
        const argSources = [
            { key: 'arguments', items: (raw.arguments || []).slice(0, 4), section: 'arguments' },
            { key: 'petitioner_arguments', items: (raw.petitioner_arguments || []).slice(0, 2), section: 'arguments' },
            { key: 'respondent_arguments', items: (raw.respondent_arguments || []).slice(0, 2), section: 'arguments' },
            { key: 'other_lawyer_arguments', items: (raw.other_lawyer_arguments || []).slice(0, 2), section: 'arguments' }
        ];

        [...legalSources, ...argSources].forEach(cat => {
            cat.items.forEach(item => {
                const imp = Number.isFinite(item.importance) ? item.importance : 0;
                out.push({
                    section: cat.section,
                    groupKey: cat.key,
                    groupLabel: GROUP_LABELS[cat.key],
                    color: GROUP_COLORS[cat.key],
                    title: normalizeNodeText(item.text),
                    score: imp,
                    scoreLabel: 'Importance',
                    edgeStyle: 'solid',
                    extra: `Node #${item.node_index}`
                });
            });
        });

        (caseData?.similar_training_cases || []).slice(0, 4).forEach(sc => {
            out.push({
                section: 'similar',
                groupKey: 'similar_training_cases',
                groupLabel: GROUP_LABELS.similar_training_cases,
                color: GROUP_COLORS.similar_training_cases,
                title: sc.case_id,
                score: sc.similarity || 0,
                scoreLabel: 'Similarity',
                edgeStyle: 'dashed',
                extra: `Rank #${sc.rank} · ${formatOutcomeLabel(sc.target_label)}`,
                outcomeClass: sc.target_label === '1' ? 'win' : 'loss'
            });
        });

        return out;
    }

    function buildGraphLegend(nodes) {
        const groups = Array.from(new Set(nodes.map(n => n.groupKey)));
        if (!groups.length) return '';
        const chips = groups.map(k => `
            <span class="legend-chip"><span class="legend-swatch" style="background:${GROUP_COLORS[k]}"></span>${escapeHtml(GROUP_LABELS[k])}</span>
        `).join('');
        return `
            ${chips}
            <span class="legend-chip"><span class="legend-line solid"></span>Importance</span>
            <span class="legend-chip"><span class="legend-line dashed"></span>Similarity</span>
        `;
    }

    function buildGraphSVG(explData, nodes) {
        if (!nodes.length) {
            return `<div class="graph-empty">No structured links were produced for this case — only the narrative explanation is available.</div>`;
        }

        const W = 760, H = 560;
        const cx = W / 2, cy = H / 2;
        const R = Math.min(W, H) * 0.38;

        const sections = {
            legal: { start: -145, end: -35, nodes: [] },
            arguments: { start: 35, end: 145, nodes: [] },
            similar: { start: 160, end: 200, nodes: [] }
        };
        nodes.forEach(n => { (sections[n.section] || sections.legal).nodes.push(n); });

        const positioned = [];
        Object.values(sections).forEach(sec => {
            if (!sec.nodes.length) return;
            sec.nodes.sort((a, b) => b.score - a.score);
            sec.nodes.forEach((n, i) => {
                const t = sec.nodes.length === 1 ? 0.5 : i / (sec.nodes.length - 1);
                const angleDeg = sec.start + t * (sec.end - sec.start);
                positionNode(n, angleDeg, R, cx, cy, positioned);
            });
        });

        const maxScore = Math.max(0.01, ...positioned.map(p => p.score));

        const edges = positioned.map(p => {
            const strokeW = 1 + (p.score / maxScore) * 3.5;
            const midX = (cx + p.x) / 2;
            const midY = (cy + p.y) / 2;
            const dx = p.x - cx, dy = p.y - cy;
            const nx = -dy, ny = dx;
            const norm = Math.hypot(nx, ny) || 1;
            const curveStrength = 26;
            const ctrlX = midX + (nx / norm) * curveStrength;
            const ctrlY = midY + (ny / norm) * curveStrength;
            const dash = p.edgeStyle === 'dashed' ? 'stroke-dasharray="6 5"' : '';
            return `<path class="graph-edge" d="M ${cx} ${cy} Q ${ctrlX} ${ctrlY} ${p.x} ${p.y}" stroke="${p.color}" stroke-width="${strokeW.toFixed(2)}" fill="none" ${dash} data-node-idx="${p.idx}"/>`;
        }).join('');

        const satellites = positioned.map(p => {
            const label = truncate(p.title, 28);
            const r = 22;
            const angleRad = (p.angleDeg * Math.PI) / 180;
            const cosA = Math.cos(angleRad);
            const sinA = Math.sin(angleRad);
            const labelOffset = r + 10;
            const lx = cosA * labelOffset;
            const ly = sinA * labelOffset;
            let anchor = 'middle';
            if (cosA > 0.25) anchor = 'start';
            else if (cosA < -0.25) anchor = 'end';
            const dyAdjust = sinA > 0.25 ? 12 : (sinA < -0.25 ? -2 : 4);
            return `
                <g class="graph-node-wrapper" transform="translate(${p.x.toFixed(1)}, ${p.y.toFixed(1)})">
                    <g class="graph-node-group" data-node-idx="${p.idx}">
                        <circle class="graph-node-halo" r="${r + 6}" fill="${p.color}" opacity="0.12"/>
                        <circle class="graph-node" r="${r}" fill="${p.color}" stroke="#0d1117" stroke-width="2"/>
                        <text class="graph-node-kind" y="4" text-anchor="middle">${escapeHtml(shortKind(p.groupKey))}</text>
                    </g>
                    <text class="graph-node-label" x="${lx.toFixed(1)}" y="${(ly + dyAdjust).toFixed(1)}" text-anchor="${anchor}">${escapeHtml(label)}</text>
                </g>
            `;
        }).join('');

        const centerR = 46;
        const center = `
            <g class="graph-center">
                <circle r="${centerR + 10}" cx="${cx}" cy="${cy}" fill="#58a6ff" opacity="0.1"/>
                <circle r="${centerR}" cx="${cx}" cy="${cy}" fill="#0d1117" stroke="#58a6ff" stroke-width="2.5"/>
                <text x="${cx}" y="${cy - 4}" text-anchor="middle" class="graph-center-kicker">CASE</text>
                <text x="${cx}" y="${cy + 12}" text-anchor="middle" class="graph-center-label">${escapeHtml(formatOutcomeLabel(explData.predicted_label))}</text>
                <text x="${cx}" y="${cy + centerR + 22}" text-anchor="middle" class="graph-center-name">${escapeHtml(truncate(explData.case_id, 56))}</text>
            </g>
        `;

        const dataPayload = encodeURIComponent(JSON.stringify(positioned.map(p => ({
            idx: p.idx, title: p.title, kind: p.groupLabel, score: p.score,
            scoreLabel: p.scoreLabel, extra: p.extra
        }))));

        return `
            <svg class="graph-svg" viewBox="0 0 ${W} ${H}" data-nodes="${dataPayload}">
                <defs>
                    <filter id="graph-glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="3" result="b"/>
                        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                </defs>
                <g class="graph-edges">${edges}</g>
                ${center}
                <g class="graph-nodes">${satellites}</g>
            </svg>
        `;
    }

    function positionNode(node, angleDeg, R, cx, cy, accumulator) {
        const angleRad = (angleDeg * Math.PI) / 180;
        const x = cx + Math.cos(angleRad) * R;
        const y = cy + Math.sin(angleRad) * R;
        accumulator.push({ ...node, x, y, angleDeg, idx: accumulator.length });
    }

    function attachGraphInteractions() {
        const svg = graphVisualizationEl.querySelector('svg.graph-svg');
        if (!svg) return;
        const payloadAttr = svg.getAttribute('data-nodes');
        if (!payloadAttr) return;
        let data = [];
        try { data = JSON.parse(decodeURIComponent(payloadAttr)); } catch (_) { data = []; }

        svg.querySelectorAll('.graph-node-group').forEach(groupEl => {
            const idx = Number(groupEl.getAttribute('data-node-idx'));
            const meta = data[idx];
            if (!meta) return;
            groupEl.addEventListener('mouseenter', (evt) => showTooltip(evt, meta));
            groupEl.addEventListener('mousemove', (evt) => moveTooltip(evt));
            groupEl.addEventListener('mouseleave', hideTooltip);
            groupEl.addEventListener('focus', (evt) => showTooltip(evt, meta));
            groupEl.addEventListener('blur', hideTooltip);
            groupEl.setAttribute('tabindex', '0');
        });
    }

    function showTooltip(evt, meta) {
        graphTooltipEl.innerHTML = `
            <div class="tt-kind">${escapeHtml(meta.kind)}</div>
            <div class="tt-title">${escapeHtml(meta.title)}</div>
            <div class="tt-row"><span>${escapeHtml(meta.scoreLabel)}</span><strong>${escapeHtml(formatPercent(meta.score, 2))}</strong></div>
            ${meta.extra ? `<div class="tt-extra">${escapeHtml(meta.extra)}</div>` : ''}
        `;
        graphTooltipEl.classList.remove('hidden');
        moveTooltip(evt);
    }

    function moveTooltip(evt) {
        const rect = graphVisualizationEl.getBoundingClientRect();
        const x = evt.clientX - rect.left + 14;
        const y = evt.clientY - rect.top + 14;
        graphTooltipEl.style.transform = `translate(${x}px, ${y}px)`;
    }

    function hideTooltip() {
        graphTooltipEl.classList.add('hidden');
    }

    function renderGraphSide(explData, nodes) {
        const legalCount = nodes.filter(n => n.section === 'legal').length;
        const argCount = nodes.filter(n => n.section === 'arguments').length;
        const simNodes = nodes.filter(n => n.section === 'similar');
        const simWins = simNodes.filter(n => n.outcomeClass === 'win').length;
        const simLosses = simNodes.filter(n => n.outcomeClass === 'loss').length;

        graphStatsEl.innerHTML = [
            { label: 'Satellite nodes', value: String(nodes.length) },
            { label: 'Legal sources', value: String(legalCount) },
            { label: 'Argument nodes', value: String(argCount) },
            { label: 'Retrieved outcomes', value: `${simWins}W / ${simLosses}L` },
            { label: 'Prediction conf.', value: formatPercent(explData.confidence, 2) }
        ].map(s => `<div class="stat-card"><div class="stat-value">${escapeHtml(s.value)}</div><div class="stat-label">${escapeHtml(s.label)}</div></div>`).join('');

        const groupStats = new Map();
        nodes.forEach(n => {
            if (!groupStats.has(n.groupKey)) {
                groupStats.set(n.groupKey, { key: n.groupKey, items: [] });
            }
            groupStats.get(n.groupKey).items.push(n);
        });
        const groups = Array.from(groupStats.values());
        const maxCount = Math.max(1, ...groups.map(g => g.items.length));
        graphBreakdownEl.innerHTML = groups.length
            ? groups
                .sort((a, b) => b.items.length - a.items.length)
                .map(g => {
                    const peak = Math.max(...g.items.map(i => i.score));
                    const color = GROUP_COLORS[g.key];
                    return `
                        <div class="breakdown-item">
                            <div class="breakdown-header">
                                <div class="breakdown-label">
                                    <span class="swatch" style="background:${color}"></span>
                                    ${escapeHtml(GROUP_LABELS[g.key])}
                                </div>
                                <span>${g.items.length}</span>
                            </div>
                            <div class="breakdown-bar-track">
                                <div class="breakdown-bar-fill" style="width:${(g.items.length / maxCount) * 100}%; background:${color}"></div>
                            </div>
                            <div class="breakdown-detail">Peak ${formatPercent(peak, 1)}</div>
                        </div>
                    `;
                }).join('')
            : '<p class="empty-copy">No structured graph links for this case.</p>';

        const strongest = nodes.slice().sort((a, b) => b.score - a.score)[0];
        const insights = [];
        insights.push({
            title: 'Prediction',
            body: `Model outcome ${formatOutcomeLabel(explData.predicted_label)} at ${formatPercent(explData.confidence, 2)}.`
        });
        if (strongest) {
            insights.push({
                title: `Strongest ${strongest.groupLabel}`,
                body: `${strongest.title} scored ${formatPercent(strongest.score, 2)}.`
            });
        }
        if (simNodes.length) {
            const agree = explData.predicted_label === '1'
                ? simWins
                : simLosses;
            insights.push({
                title: 'Retrieval Agreement',
                body: `${agree} of ${simNodes.length} retrieved analogues share the predicted outcome.`
            });
        }
        graphInsightsEl.innerHTML = insights.map(i =>
            `<div class="insight-item"><div class="insight-title">${escapeHtml(i.title)}</div><div class="insight-body">${escapeHtml(i.body)}</div></div>`
        ).join('');
    }

    function createNodeCard(title, importance, meta) {
        const card = document.createElement('div');
        card.className = 'node-item';
        const safeImp = Number.isFinite(importance) ? importance : 0;
        const pct = safeImp * 100;
        const width = Math.min(100, Math.max(10, safeImp * 220));
        card.innerHTML = `
            <div class="node-item-title">${escapeHtml(title)}</div>
            <div class="node-item-meta"><span>${escapeHtml(meta)}</span></div>
            <div class="importance-wrapper">
                <div class="importance-bar-container"><div class="importance-bar" style="width:${width}%;"></div></div>
                <div class="importance-val">${pct.toFixed(1)}%</div>
            </div>
        `;
        return card;
    }

    // ============ DATASET ANALYTICS ============
    async function showAnalytics() {
        welcomeState.classList.add('hidden');
        detailState.classList.add('hidden');
        loadingState.classList.add('hidden');
        analyticsState.classList.remove('hidden');
        document.querySelectorAll('.case-item').forEach(el => el.classList.remove('active'));

        if (analyticsLoaded) return;
        if (!analyticsPromise) {
            analyticsContentEl.innerHTML = '<div class="loading">Loading analytics...</div>';
            analyticsPromise = loadAnalyticsData()
                .then(data => {
                    analyticsLoaded = true;
                    renderAnalytics(data);
                })
                .catch(err => {
                    console.error(err);
                    analyticsContentEl.innerHTML = `<div class="loading error">Failed to load analytics: ${escapeHtml(err.message)}</div>`;
                });
        }
        await analyticsPromise;
    }

    async function loadAnalyticsData() {
        const [summaryRes, manifestRes, trainHistRes, predRes] = await Promise.all([
            fetch('../outputs/phase1_2_inference/summary.json'),
            fetch('../outputs/phase4_explanations/manifest.json'),
            fetch('../outputs/phase3_explainer/training_history.json').catch(() => null),
            fetch('../outputs/phase1_2_inference/predictions.csv').catch(() => null)
        ]);

        const summary = summaryRes.ok ? await summaryRes.json() : {};
        const manifest = manifestRes.ok ? await manifestRes.json() : { case_indices: [] };
        const trainHist = (trainHistRes && trainHistRes.ok) ? await trainHistRes.json() : null;
        const predCsvText = (predRes && predRes.ok) ? await predRes.text() : null;

        const caseIndices = manifest.case_indices || [];
        const caseFiles = await Promise.all(caseIndices.map(async idx => {
            try {
                const r = await fetch(`../outputs/phase4_explanations/cases/case_${idx}.json`);
                return r.ok ? await r.json() : null;
            } catch (_) { return null; }
        }));

        const predictions = predCsvText ? parsePredictionsCsv(predCsvText) : null;

        return { summary, manifest, trainHist, predictions, cases: caseFiles.filter(Boolean) };
    }

    function parsePredictionsCsv(text) {
        const lines = text.split(/\r?\n/).filter(l => l.length);
        if (lines.length < 2) return null;
        const header = splitCsvLine(lines[0]);
        const idx = {
            split: header.indexOf('split'),
            target: header.indexOf('target_label'),
            pred: header.indexOf('pred_label'),
            conf: header.indexOf('confidence')
        };
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const cells = splitCsvLine(lines[i]);
            if (cells.length < header.length) continue;
            rows.push({
                split: cells[idx.split],
                target: cells[idx.target],
                pred: cells[idx.pred],
                confidence: parseFloat(cells[idx.conf])
            });
        }
        return rows;
    }

    function splitCsvLine(line) {
        const out = [];
        let cur = '';
        let inQuote = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"') {
                if (inQuote && line[i + 1] === '"') { cur += '"'; i++; }
                else inQuote = !inQuote;
            } else if (ch === ',' && !inQuote) {
                out.push(cur); cur = '';
            } else {
                cur += ch;
            }
        }
        out.push(cur);
        return out;
    }

    function renderAnalytics(data) {
        const { summary, predictions, trainHist, cases } = data;

        const headline = buildAnalyticsHeadline(summary, predictions, cases);
        const splitAccuracy = buildSplitAccuracy(predictions);
        const confidenceHist = buildConfidenceHistogram(predictions);
        const topPrecedents = buildTopPrecedents(cases);
        const retrievalAgreement = buildRetrievalAgreement(cases);
        const lossChart = buildTrainingLossChart(trainHist);
        const explainedDist = buildExplainedDistribution(cases);

        analyticsContentEl.innerHTML = `
            ${headline}
            <div class="analytics-row">
                ${splitAccuracy}
                ${explainedDist}
            </div>
            <div class="analytics-row">
                ${confidenceHist}
                ${retrievalAgreement}
            </div>
            ${topPrecedents}
            ${lossChart}
        `;
    }

    function buildAnalyticsHeadline(summary, predictions, cases) {
        const totalCases = summary.n_cases || (predictions ? predictions.length : cases.length);
        const testPreds = predictions ? predictions.filter(p => p.split === 'test') : [];
        const testAcc = testPreds.length
            ? testPreds.filter(p => p.pred === p.target).length / testPreds.length
            : null;
        const overallAcc = predictions && predictions.length
            ? predictions.filter(p => p.pred === p.target).length / predictions.length
            : null;
        const avgConf = predictions && predictions.length
            ? predictions.reduce((s, p) => s + (p.confidence || 0), 0) / predictions.length
            : null;

        const cards = [
            { label: 'Total cases in graph', value: String(totalCases) },
            { label: 'Cases with full explanation', value: String(cases.length) },
            { label: 'Architecture', value: (summary.effective_model_cfg?.architecture || 'hgt').toUpperCase() },
            { label: 'Embedding dim', value: String(summary.embedding_dim || '-') },
            { label: 'Overall accuracy', value: overallAcc != null ? formatPercent(overallAcc, 2) : '—' },
            { label: 'Test accuracy', value: testAcc != null ? formatPercent(testAcc, 2) : '—' },
            { label: 'Mean confidence', value: avgConf != null ? formatPercent(avgConf, 2) : '—' },
            { label: 'Train / Val / Test', value: summary.split_counts
                ? `${summary.split_counts.train} / ${summary.split_counts.val} / ${summary.split_counts.test}`
                : '—' }
        ];

        return `
            <section class="card analytics-headline">
                <div class="card-header"><h3>🧾 Model & Dataset Summary</h3></div>
                <div class="stat-grid wide">
                    ${cards.map(c => `<div class="stat-card"><div class="stat-value">${escapeHtml(c.value)}</div><div class="stat-label">${escapeHtml(c.label)}</div></div>`).join('')}
                </div>
            </section>
        `;
    }

    function buildSplitAccuracy(predictions) {
        if (!predictions || !predictions.length) {
            return `<section class="card"><div class="card-header"><h3>🎯 Accuracy By Split</h3></div><p class="empty-copy">Predictions CSV unavailable.</p></section>`;
        }
        const splits = ['train', 'val', 'test'];
        const rows = splits.map(split => {
            const subset = predictions.filter(p => p.split === split);
            if (!subset.length) return null;
            const correct = subset.filter(p => p.pred === p.target).length;
            const acc = correct / subset.length;
            const wins = subset.filter(p => p.pred === '1').length;
            const losses = subset.length - wins;
            return { split, n: subset.length, acc, wins, losses };
        }).filter(Boolean);

        return `
            <section class="card">
                <div class="card-header"><h3>🎯 Accuracy By Split</h3></div>
                <div class="acc-list">
                    ${rows.map(r => `
                        <div class="acc-item">
                            <div class="acc-head">
                                <strong>${escapeHtml(r.split.toUpperCase())}</strong>
                                <span>${r.n.toLocaleString()} cases</span>
                            </div>
                            <div class="acc-bar-track">
                                <div class="acc-bar-fill" style="width:${(r.acc * 100).toFixed(1)}%"></div>
                                <div class="acc-bar-value">${formatPercent(r.acc, 2)}</div>
                            </div>
                            <div class="acc-meta">
                                <span class="pill success">Pred Win ${r.wins.toLocaleString()}</span>
                                <span class="pill danger">Pred Loss ${r.losses.toLocaleString()}</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </section>
        `;
    }

    function buildExplainedDistribution(cases) {
        if (!cases.length) {
            return `<section class="card"><div class="card-header"><h3>🧮 Explained Cases</h3></div><p class="empty-copy">No case files loaded.</p></section>`;
        }
        const correct = cases.filter(c => c.predicted_label === c.target_label).length;
        const wins = cases.filter(c => c.predicted_label === '1').length;
        const confs = cases.map(c => c.confidence).filter(Number.isFinite);
        const highConf = confs.filter(v => v >= 0.95).length;
        const lowConf = confs.filter(v => v < 0.6).length;
        const avgTopPrec = averageTopPrecedentImportance(cases);
        const avgSim = averageTopSimilarity(cases);

        return `
            <section class="card">
                <div class="card-header"><h3>🧮 Explained Cases (${cases.length})</h3></div>
                <div class="stat-grid">
                    <div class="stat-card"><div class="stat-value">${formatPercent(correct / cases.length, 1)}</div><div class="stat-label">Correct predictions</div></div>
                    <div class="stat-card"><div class="stat-value">${wins} / ${cases.length - wins}</div><div class="stat-label">Pred Win / Loss</div></div>
                    <div class="stat-card"><div class="stat-value">${highConf}</div><div class="stat-label">Very confident (≥95%)</div></div>
                    <div class="stat-card"><div class="stat-value">${lowConf}</div><div class="stat-label">Uncertain (&lt;60%)</div></div>
                    <div class="stat-card"><div class="stat-value">${formatPercent(avgTopPrec, 1)}</div><div class="stat-label">Avg top-precedent score</div></div>
                    <div class="stat-card"><div class="stat-value">${formatPercent(avgSim, 2)}</div><div class="stat-label">Avg top-1 similarity</div></div>
                </div>
            </section>
        `;
    }

    function averageTopPrecedentImportance(cases) {
        const vals = cases
            .map(c => c?.top_nodes?.precedent?.[0]?.importance)
            .filter(Number.isFinite);
        if (!vals.length) return 0;
        return vals.reduce((s, v) => s + v, 0) / vals.length;
    }

    function averageTopSimilarity(cases) {
        const vals = cases
            .map(c => c?.similar_training_cases?.[0]?.similarity)
            .filter(Number.isFinite);
        if (!vals.length) return 0;
        return vals.reduce((s, v) => s + v, 0) / vals.length;
    }

    function buildConfidenceHistogram(predictions) {
        if (!predictions || !predictions.length) {
            return `<section class="card"><div class="card-header"><h3>📈 Confidence Distribution</h3></div><p class="empty-copy">Predictions CSV unavailable.</p></section>`;
        }
        const bins = new Array(10).fill(0);
        predictions.forEach(p => {
            const v = p.confidence;
            if (!Number.isFinite(v)) return;
            const b = Math.min(9, Math.floor(v * 10));
            bins[b]++;
        });
        const maxBin = Math.max(...bins);
        const bars = bins.map((count, i) => {
            const lo = (i / 10).toFixed(1);
            const hi = ((i + 1) / 10).toFixed(1);
            const h = maxBin ? (count / maxBin) * 100 : 0;
            return `
                <div class="hist-bar-col" title="${count.toLocaleString()} cases (${lo}-${hi})">
                    <div class="hist-bar-count">${count.toLocaleString()}</div>
                    <div class="hist-bar-wrap"><div class="hist-bar" style="height:${h}%"></div></div>
                    <div class="hist-bar-label">${lo}</div>
                </div>
            `;
        }).join('');

        return `
            <section class="card">
                <div class="card-header"><h3>📈 Confidence Distribution (all ${predictions.length.toLocaleString()} predictions)</h3></div>
                <div class="hist">${bars}</div>
                <p class="card-subtext">Each column is a 0.1 confidence band. Most predictions pile up at the high end — typical of a saturated binary classifier.</p>
            </section>
        `;
    }

    function buildTopPrecedents(cases) {
        const counter = new Map();
        cases.forEach(c => {
            (c?.top_nodes?.precedent || []).forEach(p => {
                const key = normalizeNodeText(p.text);
                if (!key) return;
                const cur = counter.get(key) || { count: 0, sumImp: 0, maxImp: 0, appearsIn: new Set() };
                cur.count += 1;
                cur.sumImp += p.importance || 0;
                cur.maxImp = Math.max(cur.maxImp, p.importance || 0);
                cur.appearsIn.add(c.case_id);
                counter.set(key, cur);
            });
        });
        const ranked = Array.from(counter.entries())
            .map(([text, stat]) => ({
                text,
                count: stat.count,
                avgImp: stat.sumImp / stat.count,
                maxImp: stat.maxImp
            }))
            .sort((a, b) => b.count - a.count || b.avgImp - a.avgImp)
            .slice(0, 12);

        if (!ranked.length) {
            return `<section class="card"><div class="card-header"><h3>⚖️ Most Influential Precedents</h3></div><p class="empty-copy">No precedent nodes found.</p></section>`;
        }

        const maxCount = Math.max(...ranked.map(r => r.count));
        const rows = ranked.map((r, i) => `
            <div class="rank-row">
                <div class="rank-index">#${i + 1}</div>
                <div class="rank-body">
                    <div class="rank-title">${escapeHtml(r.text)}</div>
                    <div class="rank-meta">
                        <span class="pill info">${r.count}× appearances</span>
                        <span class="pill">avg ${formatPercent(r.avgImp, 1)}</span>
                        <span class="pill">peak ${formatPercent(r.maxImp, 1)}</span>
                    </div>
                    <div class="rank-bar-track">
                        <div class="rank-bar-fill" style="width:${(r.count / maxCount) * 100}%"></div>
                    </div>
                </div>
            </div>
        `).join('');

        return `
            <section class="card">
                <div class="card-header"><h3>⚖️ Most Influential Precedents (across ${cases.length} explained cases)</h3></div>
                <div class="rank-list">${rows}</div>
            </section>
        `;
    }

    function buildRetrievalAgreement(cases) {
        let withSim = 0, sameOutcome = 0, samePred = 0, totalSims = 0, simOut = 0;
        cases.forEach(c => {
            const sims = c?.similar_training_cases || [];
            if (!sims.length) return;
            withSim++;
            const top = sims[0];
            if (top.target_label === c.target_label) sameOutcome++;
            if (top.predicted_label === c.predicted_label) samePred++;
            sims.forEach(s => {
                totalSims++;
                if (s.target_label === c.predicted_label) simOut++;
            });
        });

        if (!withSim) {
            return `<section class="card"><div class="card-header"><h3>🔗 Retrieval Agreement</h3></div><p class="empty-copy">No similar-case retrievals.</p></section>`;
        }

        const topMatchPct = sameOutcome / withSim;
        const topPredPct = samePred / withSim;
        const allMatchPct = totalSims ? simOut / totalSims : 0;

        return `
            <section class="card">
                <div class="card-header"><h3>🔗 Retrieval ↔ Case Agreement</h3></div>
                <div class="stat-grid">
                    <div class="stat-card"><div class="stat-value">${formatPercent(topMatchPct, 1)}</div><div class="stat-label">Top-1 shares query outcome</div></div>
                    <div class="stat-card"><div class="stat-value">${formatPercent(topPredPct, 1)}</div><div class="stat-label">Top-1 shares query prediction</div></div>
                    <div class="stat-card"><div class="stat-value">${formatPercent(allMatchPct, 1)}</div><div class="stat-label">All retrievals match prediction</div></div>
                    <div class="stat-card"><div class="stat-value">${withSim}</div><div class="stat-label">Cases with ≥1 neighbor</div></div>
                </div>
                <p class="card-subtext">High agreement suggests the embedding space clusters outcome-compatible cases, which is exactly what drives the retrieval-augmented reasoning in phase 5.</p>
            </section>
        `;
    }

    function buildTrainingLossChart(trainHist) {
        if (!trainHist || !Array.isArray(trainHist.history) || !trainHist.history.length) {
            return '';
        }
        const hist = trainHist.history;
        const w = 720, h = 220, padL = 36, padR = 16, padT = 12, padB = 26;
        const series = [
            { key: 'pred_loss', color: '#58a6ff', label: 'pred' },
            { key: 'size_loss', color: '#f6c667', label: 'size' },
            { key: 'ent_loss', color: '#c295ff', label: 'ent' },
            { key: 'total_loss', color: '#ff9575', label: 'total' }
        ];
        const xMax = hist.length - 1 || 1;
        const yMax = Math.max(...hist.flatMap(e => series.map(s => e[s.key] || 0)));
        const yMin = Math.min(...hist.flatMap(e => series.map(s => e[s.key] || 0)));
        const scaleX = v => padL + (v / xMax) * (w - padL - padR);
        const scaleY = v => padT + (1 - (v - yMin) / ((yMax - yMin) || 1)) * (h - padT - padB);

        const paths = series.map(s => {
            const d = hist.map((e, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(2)} ${scaleY(e[s.key] || 0).toFixed(2)}`).join(' ');
            return `<path d="${d}" stroke="${s.color}" fill="none" stroke-width="1.8"/>`;
        }).join('');

        const xTicks = [0, Math.floor(xMax / 2), xMax].map(i => `
            <text x="${scaleX(i)}" y="${h - 6}" text-anchor="middle" class="axis-label">${i}</text>
        `).join('');
        const yTicks = [yMin, (yMin + yMax) / 2, yMax].map(v => `
            <text x="${padL - 6}" y="${scaleY(v) + 4}" text-anchor="end" class="axis-label">${v.toFixed(2)}</text>
            <line x1="${padL}" x2="${w - padR}" y1="${scaleY(v)}" y2="${scaleY(v)}" class="axis-grid"/>
        `).join('');

        const legend = series.map(s => `<span class="legend-chip"><span class="legend-swatch" style="background:${s.color}"></span>${escapeHtml(s.label)}</span>`).join('');

        return `
            <section class="card">
                <div class="card-header"><h3>📉 Explainer Training Loss</h3></div>
                <div class="loss-legend">${legend}</div>
                <svg viewBox="0 0 ${w} ${h}" class="loss-chart">
                    ${yTicks}
                    ${paths}
                    ${xTicks}
                </svg>
                <p class="card-subtext">Per-epoch loss components from phase-3 PGExplainer training — all curves converge as the explainer tightens around the minimal sufficient subgraph.</p>
            </section>
        `;
    }

    // ============ UTILS ============
    function setupBadge(el, text, className) {
        el.textContent = text;
        el.className = `badge ${className}`;
    }

    function showLoading(message, isError = false) {
        loadingState.classList.remove('hidden');
        loadingIndicatorEl.classList.toggle('hidden', isError);
        loadingCopyEl.textContent = message;
        loadingCopyEl.classList.toggle('error', isError);
    }

    function normalizeNodeText(text) {
        if (!text) return 'Unnamed node';
        return text
            .replace(/^case::/i, '')
            .replace(/::petitioner_arguments$/i, ' (petitioner)')
            .replace(/::respondent_arguments$/i, ' (respondent)')
            .replace(/::other_lawyer_arguments$/i, ' (other counsel)')
            .replace(/::arguments$/i, ' (arguments)')
            .trim();
    }

    function shortKind(key) {
        const map = {
            precedent: 'PREC', statute: 'STAT', provision: 'PROV',
            arguments: 'ARG', petitioner_arguments: 'PET',
            respondent_arguments: 'RESP', other_lawyer_arguments: 'OTH',
            similar_training_cases: 'SIM'
        };
        return map[key] || '•';
    }

    function truncate(text, max) {
        const s = String(text || '').trim();
        return s.length > max ? s.slice(0, max - 1).trimEnd() + '…' : s;
    }

    function formatPercent(value, digits = 1) {
        if (!Number.isFinite(value)) return 'N/A';
        return `${(value * 100).toFixed(digits)}%`;
    }

    function formatOutcomeLabel(value) {
        if (value === '1' || value === 1) return 'Win';
        if (value === '-1' || value === -1 || value === '0' || value === 0) return 'Loss';
        return String(value ?? 'Unknown');
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    init();
});
