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
    const argumentContextListEl = document.getElementById('argument-context-list');

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

    const params = new URLSearchParams(window.location.search);
    const OUTPUT_BASE = (params.get('output') || '../outputs/fin_fraud_party_args_preamble_lr_decay_fold_00').replace(/\/$/, '');

    function outputUrl(relativePath) {
        return `${OUTPUT_BASE}/${relativePath.replace(/^\//, '')}`;
    }

    const GROUP_COLORS = {
        case: '#f0f6fc',
        preamble: '#e6edf3',
        facts: '#d2a8ff',
        precedent: '#f6c667',
        statute: '#63c8ff',
        provision: '#7ce38b',
        arguments: '#ff9575',
        petitioner_arguments: '#c295ff',
        respondent_arguments: '#ff89c2',
        other_lawyer_arguments: '#9aa7b8',
        court: '#a5d6ff',
        judge: '#79c0ff',
        lawyer: '#ffa657',
        defence_lawyer: '#ffb77c',
        petitioner_lawyer: '#ffd33d',
        petitioner: '#56d364',
        respondent: '#f778ba'
    };

    const GROUP_LABELS = {
        case: 'Case',
        preamble: 'Preamble',
        facts: 'Facts',
        precedent: 'Precedent',
        statute: 'Statute',
        provision: 'Provision',
        arguments: 'Argument',
        petitioner_arguments: 'Petitioner Arg.',
        respondent_arguments: 'Respondent Arg.',
        other_lawyer_arguments: 'Other Counsel Arg.',
        court: 'Court',
        judge: 'Judge',
        lawyer: 'Lawyer',
        defence_lawyer: 'Defence Lawyer',
        petitioner_lawyer: 'Petitioner Lawyer',
        petitioner: 'Petitioner',
        respondent: 'Respondent'
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
            allCases = await loadCaseIndex();
            renderCaseList(allCases);
        } catch (error) {
            caseListEl.innerHTML = `<div class="loading error">Error loading cases: ${escapeHtml(error.message)}.<br>Make sure you are running via a web server.</div>`;
            console.error(error);
        }
    }

    async function loadCaseIndex() {
        const phase6Summary = await fetchJsonIfExists(outputUrl('phase6_misclass_diagnostic/summary.json'));
        if (Array.isArray(phase6Summary) && phase6Summary.length) {
            return phase6Summary.map(row => ({
                case_node_index: row.case_node_index,
                case_id: row.case_id || `case_${row.case_node_index}`,
                predicted_label: row.predicted_label,
                target_label: row.target_label,
                confidence: row.confidence
            }));
        }

        const manifest = await fetchJsonIfExists(outputUrl('phase4_explanations/manifest.json'));
        const indices = Array.isArray(manifest?.case_indices) ? manifest.case_indices : [];
        if (indices.length) {
            return indices.map(idx => ({
                case_node_index: idx,
                case_id: `case_${idx}`,
                predicted_label: '?',
                target_label: '?',
                confidence: null
            }));
        }

        throw new Error('No Phase 4/5/6 case index found under outputs/');
    }

    async function fetchJsonIfExists(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) return null;
            return await response.json();
        } catch (_error) {
            return null;
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
            const [caseRes, diagRes, phase7Res] = await Promise.all([
                fetch(outputUrl(`phase4_explanations/cases/case_${nodeIndex}.json`)).catch(() => null),
                fetch(outputUrl(`phase6_misclass_diagnostic/case_${nodeIndex}.json`)).catch(() => null),
                fetchPhase7Report(nodeIndex)
            ]);

            const caseData = (caseRes && caseRes.ok) ? await caseRes.json() : null;
            const diagData = (diagRes && diagRes.ok) ? await diagRes.json() : null;
            const phase7Data = (phase7Res && phase7Res.ok) ? await phase7Res.json() : null;
            const explData = buildCaseOverview(nodeIndex, caseData, diagData, phase7Data);

            if (!caseData && !diagData && !phase7Data) {
                throw new Error('Case details not found');
            }

            renderDetails(explData, caseData, diagData, phase7Data);
            loadingState.classList.add('hidden');
            detailState.classList.remove('hidden');
            setActiveTab(activeTab);
        } catch (error) {
            console.error(error);
            showLoading(`Error loading case details: ${error.message}`, true);
        }
    }

    function buildCaseOverview(nodeIndex, caseData, diagData, phase7Data) {
        const source = diagData || caseData || phase7Data || {};
        return {
            case_node_index: Number(nodeIndex),
            case_id: source.case_id || `case_${nodeIndex}`,
            predicted_label: source.predicted_label || '?',
            target_label: source.target_label || '?',
            confidence: source.confidence ?? 0,
            class_probabilities: source.class_probabilities || {},
            explanation: ''
        };
    }

    async function fetchPhase7Report(nodeIndex) {
        const current = await fetchJsonIfExists(outputUrl(`phase7_topk_embedding/case_${nodeIndex}.json`));
        if (current) return { ok: true, json: async () => current };
        const legacy = await fetchJsonIfExists(outputUrl(`phase7_counterfactual_embedding/case_${nodeIndex}.json`));
        if (legacy) return { ok: true, json: async () => legacy };
        return null;
    }

    function renderDetails(explData, caseData, diagData, phase7Data) {
        detailCaseId.textContent = explData.case_id;

        setupBadge(badgePred, `Pred: ${formatOutcomeLabel(explData.predicted_label)}`,
            explData.predicted_label === '1' ? 'success' : 'danger');
        setupBadge(badgeTarget, `Target: ${formatOutcomeLabel(explData.target_label)}`,
            explData.target_label === '1' ? 'success' : 'danger');
        setupBadge(badgeConf, `Conf: ${formatPercent(explData.confidence, 2)}`, 'info');

        const isCorrect = explData.predicted_label === explData.target_label;
        setupBadge(badgeCorrect, isCorrect ? '✓ Correct' : '✗ Incorrect', isCorrect ? 'success' : 'danger');

        renderPipelineStages(explData, caseData, diagData, phase7Data);
        renderSummaryPanel(explData, caseData, diagData);
        renderGraphView(explData, caseData);
    }

    // ============ PIPELINE STAGES ============
    function renderPipelineStages(explData, caseData, diagData, phase7Data) {
        const stages = [
            buildStage1Inference(explData, caseData),
            buildStage2CaseInput(caseData),
            buildStage3GraphExplainer(caseData),
            buildStage4ArgumentContext(caseData),
            buildStage6Diagnostic(explData, diagData),
            buildStage7Counterfactual(explData, caseData, diagData, phase7Data)
        ];
        pipelineStagesEl.innerHTML = stages.join('');
        bindDiagnosticControls(diagData);
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
        const legalCategories = [
            { key: 'precedent', items: top.precedent },
            { key: 'statute', items: top.statute },
            { key: 'provision', items: top.provision },
            { key: 'arguments', items: raw.arguments },
            { key: 'petitioner_arguments', items: raw.petitioner_arguments },
            { key: 'respondent_arguments', items: raw.respondent_arguments },
            { key: 'other_lawyer_arguments', items: raw.other_lawyer_arguments }
        ].filter(cat => Array.isArray(cat.items) && cat.items.length > 0);
        const graphCategories = getTopGraphCategories(caseData)
            .filter(cat => Array.isArray(cat.items) && cat.items.length > 0);

        if (!legalCategories.length && !graphCategories.length) {
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

        const legalBlocks = legalCategories.map(cat => buildExplainerCategory(cat)).join('');
        const graphBlocks = graphCategories.map(cat => buildExplainerCategory(cat)).join('');

        const legalSection = legalBlocks
            ? `
                <div class="stage-section">
                    <h4 class="stage-col-title">Legal Evidence</h4>
                    <div class="explainer-grid">${legalBlocks}</div>
                </div>
            `
            : '';
        const graphSection = graphBlocks
            ? `
                <div class="stage-section">
                    <h4 class="stage-col-title">Full Graph Evidence</h4>
                    <div class="explainer-grid">${graphBlocks}</div>
                </div>
            `
            : '';

        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">3</span>
                    <div>
                        <h3>Graph Explainer (Phase 3–4)</h3>
                        <p class="stage-subtitle">PGExplainer node scores across legal evidence and broader graph structure.</p>
                    </div>
                </div>
                <div class="stage-body">
                    ${legalSection}
                    ${graphSection}
                </div>
            </article>
        `;
    }

    function buildExplainerCategory(cat) {
        return `
            <div class="explainer-category">
                <div class="explainer-category-head">
                    <span class="explainer-swatch" style="background:${GROUP_COLORS[cat.key] || '#58a6ff'}"></span>
                    <span class="explainer-category-label">${escapeHtml(GROUP_LABELS[cat.key] || cat.key)}</span>
                    <span class="explainer-category-count">${cat.items.length}</span>
                </div>
                <div class="explainer-items">
                    ${cat.items.map(item => buildExplainerItem(item, cat.key)).join('')}
                </div>
            </div>
        `;
    }

    function getTopGraphCategories(caseData) {
        const graph = caseData?.top_graph_nodes || {};
        const preferred = [
            'case', 'preamble', 'facts', 'arguments',
            'petitioner_arguments', 'respondent_arguments', 'other_lawyer_arguments',
            'court', 'judge', 'lawyer', 'defence_lawyer', 'petitioner_lawyer',
            'petitioner', 'respondent', 'precedent', 'statute', 'provision'
        ];
        const seen = new Set(preferred);
        const rest = Object.keys(graph).filter(key => !seen.has(key)).sort();
        return [...preferred, ...rest].map(key => ({ key, items: graph[key] || [] }));
    }

    function buildExplainerItem(item, groupKey) {
        const text = normalizeNodeText(item.text);
        const imp = Number.isFinite(item.importance) ? item.importance : 0;
        const width = Math.min(100, Math.max(6, imp * 220));
        const edge = item.edge_type ? `<span class="edge-tag">${escapeHtml(item.edge_type)}</span>` : '';
        const scope = item.connection_scope
            ? `<span class="scope-tag ${item.target_direct ? 'direct' : 'shared'}">${escapeHtml(item.connection_scope)}</span>`
            : '';
        return `
            <div class="explainer-item">
                <div class="explainer-item-title">${escapeHtml(text)}</div>
                <div class="explainer-item-meta">
                    <span>Node #${escapeHtml(String(item.node_index ?? '-'))}</span>
                    ${scope}
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

    function buildStage4ArgumentContext(caseData) {
        const rawGroups = caseData?._raw_argument_top_nodes || {};
        const argCtx = Object.entries(rawGroups).flatMap(([group, items]) =>
            (items || []).map(item => ({ ...item, group }))
        ).sort((a, b) => (b.importance || 0) - (a.importance || 0)).slice(0, 12);

        const argCtxHtml = argCtx.length
            ? argCtx.map(item => `
                <div class="arg-ctx-item">
                    <div class="arg-ctx-head">
                        <strong>${escapeHtml(truncate(normalizeNodeText(item.text || ''), 120))}</strong>
                        <span class="pill info">${escapeHtml(formatPercent(item.importance, 1))}</span>
                    </div>
                    <div class="arg-ctx-meta">
                        ${escapeHtml(GROUP_LABELS[item.group] || item.group || 'Argument')}
                        · Node #${escapeHtml(String(item.node_index ?? '-'))}
                    </div>
                    <div class="arg-ctx-snippet">${escapeHtml(truncate(item.edge_type || '', 180))}</div>
                </div>
            `).join('')
            : '<p class="empty-copy">No bucket-local argument nodes were surfaced for this case.</p>';

        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">4</span>
                    <div>
                        <h3>Bucket-Local Argument Nodes</h3>
                        <p class="stage-subtitle">Argument-role graph nodes surfaced by the explainer for this selected bucket.</p>
                    </div>
                </div>
                <div class="stage-body">
                    ${argCtxHtml}
                </div>
            </article>
        `;
    }

    function buildStage6Diagnostic(explData, diagData) {
        if (!diagData) {
            return `
                <article class="stage-card">
                    <div class="stage-header">
                        <span class="stage-number">6</span>
                        <div>
                            <h3>Evidence Diagnostic (Phase 6)</h3>
                            <p class="stage-subtitle">Training-set label distribution behind the surfaced evidence.</p>
                        </div>
                    </div>
                    <div class="stage-body">
                        <p class="empty-copy">Phase 6 diagnostic output is not available for this case.</p>
                    </div>
                </article>
            `;
        }

        const predicted = String(diagData.predicted_label ?? explData.predicted_label ?? '?');
        const target = String(diagData.target_label ?? explData.target_label ?? '?');
        const weighted = diagData.weighted_evidence || {};
        const majority = String(weighted.majority_class ?? 'untraceable');
        const strength = toFiniteNumber(weighted.strength);
        const perNode = Array.isArray(diagData.per_node) ? diagData.per_node : [];
        const traceableNodes = toFiniteNumber(weighted.n_traceable_nodes) ?? 0;
        const totalNodes = toFiniteNumber(weighted.n_nodes) ?? perNode.length;
        const scopeLabel = diagData.diagnostic_scope === 'full_graph' ? 'Full graph' : 'Legal';
        const isMisclassified = Boolean(diagData.misclassified);
        const supportsPrediction = majority !== 'tie' && majority !== 'untraceable' && majority === predicted;
        const scopeOptions = availableTopKScopes(diagData);
        const cutoffs = topKCutoffs(diagData);
        const initialScope = scopeOptions[0]?.key || 'full_graph';
        const initialK = cutoffs[0] || 3;

        const supportLabel = majority === 'tie' || majority === 'untraceable'
            ? formatDiagnosticMajority(majority)
            : (supportsPrediction ? 'Yes' : 'No');
        const supportTone = majority === 'tie' || majority === 'untraceable'
            ? ''
            : (supportsPrediction ? 'pos' : 'neg');

        return `
            <article class="stage-card" data-diagnostic-card>
                <div class="stage-header">
                    <span class="stage-number">6</span>
                    <div>
                        <h3>Evidence Diagnostic (Phase 6)</h3>
                        <p class="stage-subtitle">Top-k sweep over PGExplainer nodes, with traceable training-neighbour evidence separated from case-local factors.</p>
                    </div>
                </div>
                <div class="stage-body">
                    <div class="stage-row">
                        <div class="stage-metric"><span>Status</span><strong class="${isMisclassified ? 'neg' : 'pos'}">${isMisclassified ? 'Misclassified' : 'Correct'}</strong></div>
                        <div class="stage-metric"><span>Scope</span><strong>${escapeHtml(scopeLabel)}</strong></div>
                        <div class="stage-metric"><span>Prediction</span><strong class="${predicted === '1' ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(predicted))}</strong></div>
                        <div class="stage-metric"><span>Target</span><strong class="${target === '1' ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(target))}</strong></div>
                        <div class="stage-metric"><span>Full-Scope Majority</span><strong>${escapeHtml(formatDiagnosticMajority(majority))}</strong></div>
                        <div class="stage-metric"><span>Full-Scope Skew</span><strong>${strength == null ? 'N/A' : escapeHtml(formatPercent(strength, 1))}</strong></div>
                        <div class="stage-metric"><span>Traceable Nodes</span><strong>${escapeHtml(`${traceableNodes} / ${totalNodes}`)}</strong></div>
                        <div class="stage-metric"><span>Traceable Weight</span><strong>${escapeHtml(formatPercent(weighted.traceable_importance_share || 0, 1))}</strong></div>
                        <div class="stage-metric"><span>Full-Scope Support</span><strong class="${supportTone}">${escapeHtml(supportLabel)}</strong></div>
                    </div>
                    ${buildDiagnosticEvidenceBarForWeighted(weighted, 'Full-scope traceable evidence vote')}
                    ${scopeOptions.length ? `
                        <div class="diagnostic-controls">
                            <label class="diagnostic-control">
                                <span>Top-k cutoff</span>
                                <input type="range" min="0" max="${Math.max(cutoffs.length - 1, 0)}" value="0" step="1" data-topk-slider>
                                <strong data-topk-label>Top ${escapeHtml(String(initialK))}</strong>
                            </label>
                            <label class="diagnostic-control">
                                <span>Evidence scope</span>
                                <select data-topk-scope>
                                    ${scopeOptions.map(opt => `<option value="${escapeHtml(opt.key)}">${escapeHtml(opt.label)}</option>`).join('')}
                                </select>
                            </label>
                        </div>
                        <div class="diagnostic-topk-body" data-topk-body>
                            ${buildTopKDiagnosticPanel(diagData, initialScope, initialK)}
                        </div>
                    ` : '<p class="empty-copy">No top-k diagnostic sweep is available for this case.</p>'}
                    ${buildLegalNeighbourhoodPanel(diagData)}
                    <div class="diagnostic-section">
                        <h4 class="stage-col-title">All Node Training-Neighbour Breakdown</h4>
                        ${buildDiagnosticNodeRows(perNode)}
                    </div>
                </div>
            </article>
        `;
    }

    function buildStage7Counterfactual(explData, caseData, diagData, phase7Data) {
        if (!phase7Data) {
            return `
                <article class="stage-card muted-stage">
                    <div class="stage-header">
                        <span class="stage-number">7</span>
                        <div>
                            <h3>Embedding Nearest Neighbours (Phase 7)</h3>
                            <p class="stage-subtitle">Run Phase 7 to find closest training cases in GNN embedding space.</p>
                        </div>
                    </div>
                    <div class="stage-body">
                        <p class="empty-copy">Phase 7 output is not available for this case yet.</p>
                    </div>
                </article>
            `;
        }

        const predicted = String(phase7Data.predicted_label ?? explData.predicted_label ?? '?');
        const target = String(phase7Data.target_label ?? explData.target_label ?? '?');
        const confidence = phase7Data.confidence ?? null;
        const neighbours = phase7Data.embedding_neighbours || {};
        const nnCounts = neighbours.target_label_counts || {};
        const nnMajority = String(neighbours.majority_target_label ?? '?');
        const nnSupports = nnMajority === predicted;

        const isMisclassified = predicted !== target;
        const isUntraceable = diagData
            ? (diagData.weighted_evidence?.majority_class === 'untraceable' ||
               Number(diagData.weighted_evidence?.n_traceable_nodes ?? 1) === 0)
            : false;

        return `
            <article class="stage-card">
                <div class="stage-header">
                    <span class="stage-number">7</span>
                    <div>
                        <h3>Embedding Nearest Neighbours (Phase 7)</h3>
                        <p class="stage-subtitle">Closest training cases by cosine similarity in the frozen GNN embedding space. Fallback when graph evidence is untraceable.</p>
                    </div>
                </div>
                <div class="stage-body">
                    <div class="stage-row">
                        <div class="stage-metric"><span>Prediction</span><strong class="${predicted === '1' ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(predicted))}</strong></div>
                        <div class="stage-metric"><span>Target</span><strong class="${target === '1' ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(target))}</strong></div>
                        ${confidence != null ? `<div class="stage-metric" title="GNN softmax probability assigned to the predicted class"><span>Confidence (?)</span><strong>${escapeHtml(formatPercent(confidence, 2))}</strong></div>` : ''}
                        <div class="stage-metric"><span>Neighbour Majority</span><strong class="${nnSupports ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(nnMajority))}</strong></div>
                        <div class="stage-metric"><span>Supports Prediction</span><strong class="${nnSupports ? 'pos' : 'neg'}">${nnSupports ? 'Yes' : 'No'}</strong></div>
                    </div>
                    ${isMisclassified && isUntraceable ? buildDrivingSignalPanel(caseData, predicted, target) : ''}
                    ${buildPhase7NeighbourRows(neighbours, nnCounts)}
                </div>
            </article>
        `;
    }

    function buildDrivingSignalPanel(caseData, predicted, target) {
        if (!caseData) return '';

        // Collect surfaced nodes sorted by importance
        const topGraphNodes = caseData.top_graph_nodes || {};
        const allNodes = [];
        for (const [nodeType, items] of Object.entries(topGraphNodes)) {
            for (const node of (items || [])) {
                allNodes.push({ nodeType, ...node });
            }
        }
        allNodes.sort((a, b) => (b.importance || 0) - (a.importance || 0));

        // Count which node types have any coverage
        const legalTypes = ['statute', 'provision', 'precedent'];
        const presentTypes = new Set(Object.keys(topGraphNodes).filter(t => (topGraphNodes[t] || []).length > 0));
        const missingLegal = legalTypes.filter(t => !presentTypes.has(t));
        const hasNoLegal = missingLegal.length === legalTypes.length;

        // Get the case text for the top-importance nodes
        const caseText = caseData.target_case_text || {};
        const TEXT_ORDER = ['respondent_arguments', 'petitioner_arguments', 'arguments', 'facts', 'preamble'];
        const textSnippets = TEXT_ORDER
            .filter(k => caseText[k] && caseText[k].trim())
            .slice(0, 3)
            .map(k => ({ section: k, text: caseText[k].trim() }));

        const top3Nodes = allNodes.slice(0, 3);

        return `
            <div class="diagnostic-evidence" style="margin-bottom:14px; border-color: rgba(248,81,73,0.3)">
                <div class="diagnostic-evidence-head" style="margin-bottom:8px">
                    <strong style="color:#f85149">Misclassified + Untraceable — Driving Signal Analysis</strong>
                    <span>Neighbour majority says ${escapeHtml(formatOutcomeLabel(target))} but model predicted ${escapeHtml(formatOutcomeLabel(predicted))}</span>
                </div>

                ${hasNoLegal ? `
                <p style="font-size:0.82rem; color: var(--text-secondary); margin:0 0 10px">
                    <strong style="color:#e3b341">⚠ No legal citation nodes</strong> — zero statutes, provisions, or precedents were found in
                    this case's graph subgraph. The model has no shared legal evidence and is relying
                    entirely on case-local text features.
                </p>` : ''}

                ${top3Nodes.length ? `
                <div style="margin-bottom:10px">
                    <div class="stage-col-title" style="margin-bottom:6px">Top surfaced nodes by importance</div>
                    ${top3Nodes.map(n => `
                        <div style="display:flex; justify-content:space-between; align-items:baseline; gap:8px; padding:4px 0; border-bottom:1px solid rgba(139,148,158,0.1); font-size:0.82rem">
                            <span class="edge-tag">${escapeHtml(n.nodeType)}</span>
                            <span style="color:var(--text-secondary); font-variant-numeric:tabular-nums">${formatPercent(n.importance, 2)} importance</span>
                        </div>
                    `).join('')}
                </div>` : ''}

                ${textSnippets.length ? `
                <div>
                    <div class="stage-col-title" style="margin-bottom:6px">Case text the model was reading</div>
                    ${textSnippets.map(({ section, text }) => `
                        <div style="margin-bottom:10px">
                            <div style="font-size:0.74rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--text-secondary); margin-bottom:3px">${escapeHtml(section.replace(/_/g, ' '))}</div>
                            <div style="font-size:0.83rem; color:#cdd9e5; line-height:1.5; padding:8px 10px; background:rgba(13,17,23,0.7); border-radius:6px; border-left:2px solid rgba(139,148,158,0.3)">${escapeHtml(text.slice(0, 400))}${text.length > 400 ? '…' : ''}</div>
                        </div>
                    `).join('')}
                </div>` : ''}
            </div>
        `;
    }

    function buildPhase7NeighbourRows(neighbours, counts) {
        const rows = Array.isArray(neighbours?.neighbours) ? neighbours.neighbours : [];
        if (!rows.length) {
            return '<p class="empty-copy">No embedding-neighbour records were generated for this case.</p>';
        }

        const totalCount = rows.length;
        const winCount = rows.filter(r => String(r.target_label) === '1').length;
        const lossCount = totalCount - winCount;
        const winRate = totalCount > 0 ? winCount / totalCount : 0;
        const lossRate = 1 - winRate;

        // similarity-weighted win/loss
        const totalSim = rows.reduce((s, r) => s + (r.cosine_similarity || 0), 0);
        const winSimSum = rows.filter(r => String(r.target_label) === '1').reduce((s, r) => s + (r.cosine_similarity || 0), 0);
        const weightedWinRate = totalSim > 0 ? winSimSum / totalSim : 0;
        const weightedLossRate = 1 - weightedWinRate;

        // avg similarity by label
        const winRows = rows.filter(r => String(r.target_label) === '1');
        const lossRows = rows.filter(r => String(r.target_label) === '-1');
        const avgSimWin = winRows.length ? winRows.reduce((s, r) => s + (r.cosine_similarity || 0), 0) / winRows.length : null;
        const avgSimLoss = lossRows.length ? lossRows.reduce((s, r) => s + (r.cosine_similarity || 0), 0) / lossRows.length : null;

        const winPct = (winRate * 100).toFixed(0);
        const lossPct = (lossRate * 100).toFixed(0);
        const wWinPct = (weightedWinRate * 100).toFixed(0);
        const wLossPct = (weightedLossRate * 100).toFixed(0);

        return `
            <div class="diagnostic-section">
                <h4 class="stage-col-title">Nearest Training Cases in GNN Embedding Space</h4>

                <div class="diagnostic-evidence" style="margin-bottom:12px">
                    <div class="diagnostic-evidence-head">
                        <span>Raw label mix — ${winCount} Win / ${lossCount} Loss of ${totalCount} neighbours</span>
                        <span>${winPct}% Win · ${lossPct}% Loss</span>
                    </div>
                    <div class="diagnostic-stack">
                        <div class="diagnostic-segment win" style="width:${winPct}%" title="Win: ${winPct}%">${winPct >= 15 ? winPct + '%' : ''}</div>
                        <div class="diagnostic-segment loss" style="width:${lossPct}%" title="Loss: ${lossPct}%">${lossPct >= 15 ? lossPct + '%' : ''}</div>
                    </div>
                </div>

                <div class="diagnostic-evidence" style="margin-bottom:12px">
                    <div class="diagnostic-evidence-head">
                        <span title="Each neighbour weighted by its cosine similarity — closer neighbours count more">Similarity-weighted label mix</span>
                        <span>${wWinPct}% Win · ${wLossPct}% Loss</span>
                    </div>
                    <div class="diagnostic-stack">
                        <div class="diagnostic-segment win" style="width:${wWinPct}%" title="Weighted Win: ${wWinPct}%">${wWinPct >= 15 ? wWinPct + '%' : ''}</div>
                        <div class="diagnostic-segment loss" style="width:${wLossPct}%" title="Weighted Loss: ${wLossPct}%">${wLossPct >= 15 ? wLossPct + '%' : ''}</div>
                    </div>
                </div>

                <div class="stage-row" style="margin-bottom:14px">
                    ${avgSimWin != null ? `<div class="stage-metric" title="Mean cosine similarity among Win neighbours"><span>Avg sim (Win)</span><strong class="pos">${avgSimWin.toFixed(4)}</strong></div>` : ''}
                    ${avgSimLoss != null ? `<div class="stage-metric" title="Mean cosine similarity among Loss neighbours"><span>Avg sim (Loss)</span><strong class="neg">${avgSimLoss.toFixed(4)}</strong></div>` : ''}
                    <div class="stage-metric" title="Similarity-weighted outcome vote"><span>Weighted vote</span><strong class="${weightedWinRate >= 0.5 ? 'pos' : 'neg'}">${weightedWinRate >= 0.5 ? 'Win' : 'Loss'} (${(Math.max(weightedWinRate, weightedLossRate) * 100).toFixed(0)}%)</strong></div>
                </div>

                <div class="diagnostic-node-list">
                    ${rows.slice(0, 8).map(row => {
                        const label = String(row.target_label);
                        const simPct = Math.min(100, ((row.cosine_similarity || 0) * 100)).toFixed(1);
                        return `
                            <div class="diagnostic-node-row">
                                <div class="diagnostic-node-head">
                                    <strong>#${escapeHtml(String(row.rank))} <span class="${label === '1' ? 'pos' : 'neg'}">${escapeHtml(formatOutcomeLabel(label))}</span></strong>
                                    <span>similarity ${escapeHtml(Number(row.cosine_similarity || 0).toFixed(4))} · pred ${escapeHtml(formatOutcomeLabel(row.pred_label))}</span>
                                </div>
                                <div class="diagnostic-mini-stack" style="margin:6px 0 6px">
                                    <div class="diagnostic-mini-segment ${label === '1' ? 'win' : 'loss'}" style="width:${simPct}%"></div>
                                </div>
                                <div class="diagnostic-node-text">${escapeHtml(truncate(row.case_id || '', 190))}</div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    function formatDelta(value) {
        const n = toFiniteNumber(value);
        if (n == null) return 'N/A';
        const sign = n > 0 ? '+' : '';
        return `${sign}${formatPercent(n, 2)}`;
    }

    function buildDiagnosticEvidenceBar(diagData) {
        return buildDiagnosticEvidenceBarForWeighted(diagData.weighted_evidence, 'Weighted Evidence Distribution');
    }

    function buildDiagnosticEvidenceBarForWeighted(weighted, title) {
        const winShare = getDiagnosticPct(weighted, '1');
        const lossShare = getDiagnosticPct(weighted, '-1');
        const hasEvidence = (winShare || 0) + (lossShare || 0) > 0;
        if (!hasEvidence) {
            return '<p class="empty-copy">No weighted training-neighbour evidence was found for the surfaced nodes.</p>';
        }
        return `
            <div class="diagnostic-evidence">
                <div class="diagnostic-evidence-head">
                    <span>${escapeHtml(title)}</span>
                    <span>Loss ${escapeHtml(formatPercent(lossShare || 0, 1))} / Win ${escapeHtml(formatPercent(winShare || 0, 1))}</span>
                </div>
                <div class="diagnostic-stack" aria-label="Weighted evidence distribution">
                    <div class="diagnostic-segment loss" style="width:${((lossShare || 0) * 100).toFixed(2)}%">${lossShare >= 0.08 ? `Loss ${formatPercent(lossShare, 0)}` : ''}</div>
                    <div class="diagnostic-segment win" style="width:${((winShare || 0) * 100).toFixed(2)}%">${winShare >= 0.08 ? `Win ${formatPercent(winShare, 0)}` : ''}</div>
                </div>
            </div>
        `;
    }

    function buildLegalNeighbourhoodPanel(diagData) {
        const legalRows = (Array.isArray(diagData?.legal_per_node) ? diagData.legal_per_node : [])
            .filter(row => ['statute', 'provision', 'precedent'].includes(row.node_type));
        if (!legalRows.length) return '';

        const hasCaseLists = legalRows.some(row => Array.isArray(row.connected_train_cases));
        const rowsHtml = legalRows.map((row, index) => {
            const total = toFiniteNumber(row.n_train_neighbours) ?? 0;
            const allConnected = toFiniteNumber(row.n_connected_cases) ?? total;
            const lossCount = toFiniteNumber(row['label_-1']) ?? 0;
            const winCount = toFiniteNumber(row.label_1) ?? 0;
            const cases = Array.isArray(row.connected_train_cases) ? row.connected_train_cases : [];
            const visibleCount = cases.length > 80 ? 40 : cases.length;
            const visibleCases = cases.slice(0, visibleCount);
            const majority = formatDiagnosticMajority(row.majority_class);
            const importance = toFiniteNumber(row.importance) ?? 0;
            const groupLabel = GROUP_LABELS[row.node_type] || row.node_type || 'Node';
            const note = cases.length
                ? `showing ${visibleCases.length} of ${total} training cases`
                : (total ? 'rerun Phase 6 to materialize connected case rows' : 'no connected training cases');
            return `
                <details class="legal-neighbour-row" ${index < 2 ? 'open' : ''}>
                    <summary>
                        <span class="legal-neighbour-summary">
                            <strong>${escapeHtml(groupLabel)}</strong>
                            <span>${escapeHtml(truncate(row.text || '', 150))}</span>
                        </span>
                        <span class="legal-neighbour-meta">
                            ${escapeHtml(formatPercent(importance, 1))} importance · ${escapeHtml(String(total))} train / ${escapeHtml(String(allConnected))} connected · ${escapeHtml(majority)}
                        </span>
                    </summary>
                    <div class="diagnostic-mini-stack legal-neighbour-stack">
                        <div class="diagnostic-mini-segment loss" style="width:${total ? ((lossCount / total) * 100).toFixed(2) : 0}%"></div>
                        <div class="diagnostic-mini-segment win" style="width:${total ? ((winCount / total) * 100).toFixed(2) : 0}%"></div>
                    </div>
                    <div class="legal-neighbour-counts">
                        <span>Loss ${escapeHtml(String(Math.round(lossCount)))}</span>
                        <span>Win ${escapeHtml(String(Math.round(winCount)))}</span>
                        <span>${escapeHtml(note)}</span>
                    </div>
                    ${visibleCases.length ? `
                        <div class="legal-neighbour-case-list">
                            ${visibleCases.map(item => {
                                const label = String(item.label ?? '?');
                                const tone = label === '1' ? 'pos' : (label === '-1' ? 'neg' : '');
                                return `
                                    <div class="legal-neighbour-case">
                                        <span class="${tone}">${escapeHtml(formatOutcomeLabel(label))}</span>
                                        <span>${escapeHtml(truncate(item.case_id || '', 190))}</span>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    ` : ''}
                </details>
            `;
        }).join('');

        return `
            <div class="diagnostic-section">
                <h4 class="stage-col-title">Legal Shared-Case Neighbours</h4>
                ${hasCaseLists ? '' : '<p class="empty-copy">Existing Phase 6 JSON has counts only. Rerun Phase 6 to include connected training-case rows.</p>'}
                <div class="legal-neighbour-list">${rowsHtml}</div>
            </div>
        `;
    }

    function bindDiagnosticControls(diagData) {
        if (!diagData?.topk_diagnostics) return;
        const card = pipelineStagesEl.querySelector('[data-diagnostic-card]');
        if (!card) return;
        const slider = card.querySelector('[data-topk-slider]');
        const scopeSelect = card.querySelector('[data-topk-scope]');
        const label = card.querySelector('[data-topk-label]');
        const body = card.querySelector('[data-topk-body]');
        if (!slider || !scopeSelect || !label || !body) return;
        const cutoffs = topKCutoffs(diagData);
        const update = () => {
            const k = cutoffs[Number(slider.value)] || cutoffs[0] || 3;
            label.textContent = `Top ${k}`;
            body.innerHTML = buildTopKDiagnosticPanel(diagData, scopeSelect.value, k);
        };
        slider.addEventListener('input', update);
        scopeSelect.addEventListener('change', update);
    }

    function buildTopKDiagnosticPanel(diagData, scopeKey, k) {
        const sweep = diagData?.topk_diagnostics?.[scopeKey] || [];
        const selected = sweep.find(item => Number(item.k) === Number(k)) || sweep[0];
        if (!selected) {
            return '<p class="empty-copy">No top-k records were produced for this scope.</p>';
        }
        const weighted = selected.weighted_evidence || {};
        const majority = String(selected.evidence_majority ?? weighted.majority_class ?? 'untraceable');
        const supports = Boolean(selected.supports_prediction);
        const supportTone = majority === 'tie' || majority === 'untraceable' ? '' : (supports ? 'pos' : 'neg');
        const supportText = majority === 'tie' || majority === 'untraceable'
            ? formatDiagnosticMajority(majority)
            : (supports ? 'Supports prediction' : 'Does not support prediction');
        const traceable = toFiniteNumber(weighted.n_traceable_nodes) ?? 0;
        const total = toFiniteNumber(weighted.n_nodes) ?? selected.n_nodes_used ?? 0;
        const strength = toFiniteNumber(weighted.strength);
        return `
            <div class="diagnostic-section">
                <div class="stage-row compact">
                    <div class="stage-metric"><span>k</span><strong>${escapeHtml(String(selected.k))}</strong></div>
                    <div class="stage-metric"><span>Top-k Majority</span><strong>${escapeHtml(formatDiagnosticMajority(majority))}</strong></div>
                    <div class="stage-metric"><span>Skew</span><strong>${strength == null ? 'N/A' : escapeHtml(formatPercent(strength, 1))}</strong></div>
                    <div class="stage-metric"><span>Traceable</span><strong>${escapeHtml(`${traceable} / ${total}`)}</strong></div>
                    <div class="stage-metric"><span>Traceable Weight</span><strong>${escapeHtml(formatPercent(weighted.traceable_importance_share || 0, 1))}</strong></div>
                    <div class="stage-metric"><span>Verdict</span><strong class="${supportTone}">${escapeHtml(supportText)}</strong></div>
                </div>
                ${buildDiagnosticEvidenceBarForWeighted(weighted, `Top ${selected.k} traceable evidence vote`)}
                ${buildDiagnosticNodeRows(Array.isArray(selected.per_node) ? selected.per_node : [])}
            </div>
        `;
    }

    function topKCutoffs(diagData) {
        const explicit = Array.isArray(diagData?.top_k_cutoffs) ? diagData.top_k_cutoffs : [];
        const fromSweep = Object.values(diagData?.topk_diagnostics || {})
            .flatMap(items => Array.isArray(items) ? items.map(item => Number(item.k)) : []);
        const values = [...explicit, ...fromSweep]
            .map(Number)
            .filter(Number.isFinite);
        return Array.from(new Set(values)).sort((a, b) => a - b);
    }

    function availableTopKScopes(diagData) {
        const labels = {
            full_graph: 'Full graph',
            legal: 'Legal only',
            traceable_full_graph: 'Traceable full graph',
            traceable_legal: 'Traceable legal'
        };
        return Object.entries(diagData?.topk_diagnostics || {})
            .filter(([, items]) => Array.isArray(items) && items.length)
            .map(([key]) => ({ key, label: labels[key] || key }));
    }

    function buildDiagnosticNodeRows(rows) {
        if (!rows.length) {
            return '<p class="empty-copy">No per-node training-neighbour rows were produced for this case.</p>';
        }
        return `
            <div class="diagnostic-node-list">
                ${rows.slice(0, 8).map(row => {
                    const lossShare = getDiagnosticPct(row, '-1');
                    const winShare = getDiagnosticPct(row, '1');
                    const trainNeighbours = toFiniteNumber(row.n_train_neighbours) ?? 0;
                    const importance = toFiniteNumber(row.importance) ?? 0;
                    const majority = formatDiagnosticMajority(row.majority_class);
                    const groupLabel = GROUP_LABELS[row.node_type] || row.node_type || 'Node';
                    const scope = row.connection_scope
                        ? `<span class="scope-tag ${row.target_direct ? 'direct' : 'shared'}">${escapeHtml(row.connection_scope)}</span>`
                        : '';
                    const traceStatus = row.trace_status
                        ? `<span class="scope-tag trace">${escapeHtml(formatTraceStatus(row.trace_status))}</span>`
                        : '';
                    return `
                        <div class="diagnostic-node-row">
                            <div class="diagnostic-node-head">
                                <strong>${escapeHtml(groupLabel)}</strong>
                                <span>${escapeHtml(formatPercent(importance, 1))} importance · ${escapeHtml(String(trainNeighbours))} train cases · ${escapeHtml(majority)}</span>
                            </div>
                            <div class="explainer-item-meta">${scope}${traceStatus}<span class="edge-tag">${escapeHtml(row.edge_type || '')}</span></div>
                            <div class="diagnostic-node-text">${escapeHtml(truncate(row.text || '', 190))}</div>
                            <div class="diagnostic-mini-stack">
                                <div class="diagnostic-mini-segment loss" style="width:${((lossShare || 0) * 100).toFixed(2)}%"></div>
                                <div class="diagnostic-mini-segment win" style="width:${((winShare || 0) * 100).toFixed(2)}%"></div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
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
    function renderSummaryPanel(explData, caseData, diagData) {
        const weighted = diagData?.weighted_evidence || {};
        if (!diagData) {
            summaryExplanationEl.innerHTML = '<p class="empty-copy">Run Phase 6 to see the bucket-local evidence diagnostic for this prediction.</p>';
        } else {
            const majority = weighted.majority_class || 'untraceable';
            const strength = formatPercent(weighted.strength || 0, 1);
            const traceable = `${weighted.n_traceable_nodes || 0} / ${weighted.n_nodes || 0}`;
            summaryExplanationEl.innerHTML = `
                <p>The model predicted <strong>${escapeHtml(formatOutcomeLabel(explData.predicted_label))}</strong>
                with ${escapeHtml(formatPercent(explData.confidence, 2))} confidence for this bucket-local graph.</p>
                <p>Among surfaced evidence nodes that trace back to training cases, the importance-weighted majority is
                <strong>${escapeHtml(formatOutcomeLabel(majority))}</strong> with ${escapeHtml(strength)} skew.
                Traceable nodes: ${escapeHtml(traceable)}.</p>
            `;
        }
        renderPrecedents(caseData);
        renderArgumentContext(caseData);
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

    function renderArgumentContext(caseData) {
        if (!argumentContextListEl) {
            return;
        }
        argumentContextListEl.innerHTML = '';
        const rawGroups = caseData?._raw_argument_top_nodes || {};
        const contexts = Object.entries(rawGroups).flatMap(([group, items]) =>
            (items || []).map(item => ({ ...item, group }))
        ).sort((a, b) => (b.importance || 0) - (a.importance || 0)).slice(0, 12);
        if (!contexts.length) {
            argumentContextListEl.innerHTML = '<p class="empty-copy">No bucket-local argument nodes surfaced.</p>';
            return;
        }
        contexts.forEach(item => {
            const card = document.createElement('div');
            card.className = 'node-item';
            card.innerHTML = `
                <div class="node-item-title">${escapeHtml(truncate(normalizeNodeText(item.text || ''), 160))}</div>
                <div class="node-item-meta" style="margin-bottom:8px;">
                    <span>${escapeHtml(GROUP_LABELS[item.group] || item.group || 'Argument')}</span>
                    <span>${escapeHtml(formatPercent(item.importance || 0, 1))} importance</span>
                </div>
                <div class="node-item-meta">${escapeHtml(truncate(item.edge_type || '', 220))}</div>
            `;
            argumentContextListEl.appendChild(card);
        });
    }

    // ============ SVG GRAPH ============
    function renderGraphView(explData, caseData) {
        const nodes = collectGraphNodes(caseData);
        graphDescriptionEl.textContent = nodes.length
            ? 'Node-link view. Edge thickness ≈ PGExplainer importance.'
            : 'No structured graph links available for this case.';
        graphLegendEl.innerHTML = buildGraphLegend(nodes);
        graphVisualizationEl.innerHTML = buildGraphSVG(explData, nodes);
        attachGraphInteractions();

        renderGraphSide(explData, nodes);
    }

    function collectGraphNodes(caseData) {
        const out = [];
        const graphCategories = getTopGraphCategories(caseData)
            .flatMap(cat => (cat.items || []).map(item => ({ cat, item })))
            .sort((a, b) => (b.item.importance || 0) - (a.item.importance || 0))
            .slice(0, 22);

        if (graphCategories.length) {
            graphCategories.forEach(({ cat, item }) => {
                const imp = Number.isFinite(item.importance) ? item.importance : 0;
                out.push({
                    section: graphNodeSection(cat.key),
                    groupKey: cat.key,
                    groupLabel: GROUP_LABELS[cat.key] || cat.key,
                    color: GROUP_COLORS[cat.key] || '#58a6ff',
                    title: normalizeNodeText(item.text),
                    score: imp,
                    scoreLabel: 'Importance',
                    edgeStyle: 'solid',
                    extra: [
                        `Node #${item.node_index}`,
                        item.connection_scope || ''
                    ].filter(Boolean).join(' · ')
                });
            });
        } else {
            const top = caseData?.top_nodes || {};
            const raw = caseData?._raw_argument_top_nodes || {};
            const fallbackSources = [
                { key: 'precedent', items: (top.precedent || []).slice(0, 4), section: 'legal' },
                { key: 'statute', items: (top.statute || []).slice(0, 3), section: 'legal' },
                { key: 'provision', items: (top.provision || []).slice(0, 3), section: 'legal' },
                { key: 'arguments', items: (raw.arguments || []).slice(0, 4), section: 'text' },
                { key: 'petitioner_arguments', items: (raw.petitioner_arguments || []).slice(0, 2), section: 'text' },
                { key: 'respondent_arguments', items: (raw.respondent_arguments || []).slice(0, 2), section: 'text' },
                { key: 'other_lawyer_arguments', items: (raw.other_lawyer_arguments || []).slice(0, 2), section: 'text' }
            ];
            fallbackSources.forEach(cat => {
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
        }

        return out;
    }

    function graphNodeSection(key) {
        if (['statute', 'provision', 'precedent'].includes(key)) return 'legal';
        if (['preamble', 'facts', 'arguments', 'petitioner_arguments', 'respondent_arguments', 'other_lawyer_arguments'].includes(key)) return 'text';
        if (['court', 'judge', 'lawyer', 'defence_lawyer', 'petitioner_lawyer', 'petitioner', 'respondent'].includes(key)) return 'actors';
        if (key === 'case') return 'cases';
        return 'other';
    }

    function buildGraphLegend(nodes) {
        const groups = Array.from(new Set(nodes.map(n => n.groupKey)));
        if (!groups.length) return '';
        const chips = groups.map(k => `
            <span class="legend-chip"><span class="legend-swatch" style="background:${GROUP_COLORS[k] || '#58a6ff'}"></span>${escapeHtml(GROUP_LABELS[k] || k)}</span>
        `).join('');
        return `
            ${chips}
            <span class="legend-chip"><span class="legend-line solid"></span>Importance</span>
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
            legal: { start: -150, end: -70, nodes: [] },
            text: { start: -45, end: 35, nodes: [] },
            actors: { start: 60, end: 140, nodes: [] },
            cases: { start: 165, end: 210, nodes: [] },
            other: { start: 225, end: 305, nodes: [] }
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
            return `<path class="graph-edge" d="M ${cx} ${cy} Q ${ctrlX} ${ctrlY} ${p.x} ${p.y}" stroke="${p.color}" stroke-width="${strokeW.toFixed(2)}" fill="none" data-node-idx="${p.idx}"/>`;
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
        const textCount = nodes.filter(n => n.section === 'text').length;
        const actorCount = nodes.filter(n => n.section === 'actors').length;
        const caseCount = nodes.filter(n => n.section === 'cases').length;

        graphStatsEl.innerHTML = [
            { label: 'Satellite nodes', value: String(nodes.length) },
            { label: 'Legal sources', value: String(legalCount) },
            { label: 'Text sections', value: String(textCount) },
            { label: 'Actor nodes', value: String(actorCount) },
            { label: 'Case nodes', value: String(caseCount) },
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
                    const color = GROUP_COLORS[g.key] || '#58a6ff';
                    return `
                        <div class="breakdown-item">
                            <div class="breakdown-header">
                                <div class="breakdown-label">
                                    <span class="swatch" style="background:${color}"></span>
                                    ${escapeHtml(GROUP_LABELS[g.key] || g.key)}
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
            fetch(outputUrl('phase1_2_inference/summary.json')),
            fetch(outputUrl('phase4_explanations/manifest.json')),
            fetch(outputUrl('phase3_explainer/training_history.json')).catch(() => null),
            fetch(outputUrl('phase1_2_inference/predictions.csv')).catch(() => null)
        ]);

        const summary = summaryRes.ok ? await summaryRes.json() : {};
        const manifest = manifestRes.ok ? await manifestRes.json() : { case_indices: [] };
        const trainHist = (trainHistRes && trainHistRes.ok) ? await trainHistRes.json() : null;
        const predCsvText = (predRes && predRes.ok) ? await predRes.text() : null;

        const caseIndices = manifest.case_indices || [];
        const caseFiles = await Promise.all(caseIndices.map(async idx => {
            try {
                const r = await fetch(outputUrl(`phase4_explanations/cases/case_${idx}.json`));
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

        return `
            <section class="card">
                <div class="card-header"><h3>🧮 Explained Cases (${cases.length})</h3></div>
                <div class="stat-grid">
                    <div class="stat-card"><div class="stat-value">${formatPercent(correct / cases.length, 1)}</div><div class="stat-label">Correct predictions</div></div>
                    <div class="stat-card"><div class="stat-value">${wins} / ${cases.length - wins}</div><div class="stat-label">Pred Win / Loss</div></div>
                    <div class="stat-card"><div class="stat-value">${highConf}</div><div class="stat-label">Very confident (≥95%)</div></div>
                    <div class="stat-card"><div class="stat-value">${lowConf}</div><div class="stat-label">Uncertain (&lt;60%)</div></div>
                    <div class="stat-card"><div class="stat-value">${formatPercent(avgTopPrec, 1)}</div><div class="stat-label">Avg top-precedent score</div></div>
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
            case: 'CASE', preamble: 'PRE', facts: 'FACT',
            precedent: 'PREC', statute: 'STAT', provision: 'PROV',
            arguments: 'ARG', petitioner_arguments: 'PET',
            respondent_arguments: 'RESP', other_lawyer_arguments: 'OTH',
            court: 'CRT', judge: 'JDG', lawyer: 'LAW',
            defence_lawyer: 'DEF', petitioner_lawyer: 'PLAW',
            petitioner: 'PTR', respondent: 'RSP'
        };
        return map[key] || '•';
    }

    function truncate(text, max) {
        const s = String(text || '').trim();
        return s.length > max ? s.slice(0, max - 1).trimEnd() + '…' : s;
    }

    function toFiniteNumber(value) {
        const num = typeof value === 'number' ? value : Number.parseFloat(value);
        return Number.isFinite(num) ? num : null;
    }

    function getDiagnosticPct(source, label) {
        if (!source) return null;
        const value = toFiniteNumber(source[`pct_${label}`]);
        if (value == null) return null;
        return Math.max(0, Math.min(1, value));
    }

    function formatDiagnosticMajority(value) {
        if (value === 'tie') return 'Tie';
        if (value === 'untraceable') return 'Untraceable';
        return formatOutcomeLabel(value);
    }

    function formatTraceStatus(value) {
        const map = {
            traceable: 'Traceable',
            case_local_untraceable: 'Case-local',
            shared_untraceable: 'No train neighbours'
        };
        return map[value] || String(value || '');
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
