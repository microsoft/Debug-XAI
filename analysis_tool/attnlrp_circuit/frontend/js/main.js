const API_BASE = "/api";

// Loading Spinner Helpers
const elLoadingOverlay = document.getElementById('loading-overlay');
const elLoadingMessage = document.getElementById('loading-message');

function showLoading(msg="Processing...") {
    if(elLoadingMessage) elLoadingMessage.textContent = msg;
    if(elLoadingOverlay) {
        elLoadingOverlay.classList.remove('hidden');
        // Force browser repaint
        void elLoadingOverlay.offsetWidth;
    }
}

function hideLoading() {
    if(elLoadingOverlay) elLoadingOverlay.classList.add('hidden');
}

// State
let appState = {
    tokens: [],
    selectedTargetToken: null,
    selectedContrastToken: null,
    circuitData: null,
    mouseX: -1,
    mouseY: -1,
    extraTokens: [], // Stores objects: { type: 'id'|'str', value: ... }
    lastRelevance: null // Cache for attribution map re-rendering
};

// Selection State for Graph
let selectedNode = null;
let lastRenderedNodes = [];

// DOM Elements
const elModelSeries = document.getElementById('model-series');
const elModelSelect = document.getElementById('model-select');
const elModelRevision = document.getElementById('model-revision');
const elModelPath = document.getElementById('model-path');
const elModelDtype = document.getElementById('model-dtype');
const elQuant = document.getElementById('quant-4bit');
const elBtnLoad = document.getElementById('btn-load-model');
const elStatus = document.getElementById('model-status');

const elTraceDataset = document.getElementById('trace-dataset');
const elTraceFile = document.getElementById('trace-file');
const elSectionTraceExplore = document.getElementById('section-trace-explore');
const elContainerExploreOriginal = document.getElementById('container-explore-original');
const elContainerExplore4b = document.getElementById('container-explore-4b');
const elHeaderExploreOther = document.getElementById('header-explore-other');

const elPrompt = document.getElementById('prompt-input');
const elPromptOrig = document.getElementById('prompt-orig-completion');
const elAppendBos = document.getElementById('append-bos');
const elBtnLogits = document.getElementById('btn-compute-logits');

const elSectionLogits = document.getElementById('section-logits');
const elSectionInputAttr = document.getElementById('section-input-attr');
const elLogitsTableBody = document.querySelector('#logits-table tbody');
const elTopTokenDisplay = document.getElementById('top-token-display');
const elContrastTokenDisplay = document.getElementById('contrast-token-display');

const elSectionCircuit = document.getElementById('section-circuit');
const elBpMode = document.getElementById('bp-mode');
const elBpStrategy = document.getElementById('bp-strategy');
const elBpK = document.getElementById('bp-k');
const elBpRefId = document.getElementById('bp-ref-id');
const elOptTopk = document.getElementById('opt-topk');
const elOptRef = document.getElementById('opt-ref');

const elTargetLayer = document.getElementById('target-layer');
const elSourceLayer = document.getElementById('source-layer');
const elLayersList = document.getElementById('layers-list'); // NEW INPUT
const elBtnCircuit = document.getElementById('btn-compute-circuit');

const elVisStrength = document.getElementById('vis-strength');
const elValVisStrength = document.getElementById('val-vis-strength');

// New Controls
const elPruningMode = document.getElementById('pruning-mode');
const elVisTopP = document.getElementById('vis-top-p');
const elValVisTopP = document.getElementById('val-vis-top-p');
const elValGlobalThresh = document.getElementById('val-global-thresh');
const elCtrlTopP = document.getElementById('ctrl-top-p');
const elCtrlGlobalThresh = document.getElementById('ctrl-global-thresh');

const elShowAllTokens = document.getElementById('show-all-tokens');
const elBtnLayerDefault = document.getElementById('btn-layer-default');
const elBtnLayerAll = document.getElementById('btn-layer-all');

// Input Attribution Elements
const elAttrBpMode = document.getElementById('attr-bp-mode');
const elAttrBpStrategy = document.getElementById('attr-bp-strategy');
const elAttrBpRefId = document.getElementById('attr-bp-ref-id');
const elAttrOptRef = document.getElementById('attr-opt-ref');
const elAttrHideBos = document.getElementById('attr-hide-bos');
const elBtnComputeAttr = document.getElementById('btn-compute-attr');
const elBtnSaveGraph = document.getElementById('btn-save-graph'); // Save Button
const elInputAttributionDisplay = document.getElementById('input-attribution-display');
const elAttrDiffStrategyOpts = document.getElementById('attr-diff-strategy-opts'); // Container for strategy ops

const elTokenCountDisplay = document.getElementById('token-count-display');
const elVisLayerSpacing = document.getElementById('vis-layer-spacing');
const elValVisLayerSpacing = document.getElementById('val-vis-layer-spacing');

const elCanvas = document.getElementById('circuit-canvas');
const elTooltip = document.getElementById('tooltip');

// Event Listeners
console.log("Attaching event listeners...");
try {
    if(elBtnLoad) elBtnLoad.addEventListener('click', loadModel);
    else console.error("elBtnLoad not found");
    
    if(elBtnLogits) elBtnLogits.addEventListener('click', () => computeLogits(false));

    // Add Token Feature
    const elBtnAddToken = document.getElementById('btn-add-token');
    const elAddTokenInput = document.getElementById('add-token-input');
    
    if(elBtnAddToken && elAddTokenInput) {
        elBtnAddToken.addEventListener('click', () => {
            const val = elAddTokenInput.value.trim();
            if(!val) return;
            
            // Parse: is it ID or String?
            // If numeric, treat as ID.
            if (/^\d+$/.test(val)) {
                const id = parseInt(val);
                appState.extraTokens.push({ type: 'id', value: id });
            } else {
                appState.extraTokens.push({ type: 'str', value: val });
            }
            
            elAddTokenInput.value = ""; // Clear input
            computeLogits(true); // Keep extras!
        });
        
        // Allow Enter key
        elAddTokenInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') elBtnAddToken.click();
        });
    }
    if(elBpMode) elBpMode.addEventListener('change', updateBpOptions);
    if(elBpStrategy) elBpStrategy.addEventListener('change', updateBpOptions);
    if(elBtnCircuit) elBtnCircuit.addEventListener('click', computeCircuit);
    if(elShowAllTokens) elShowAllTokens.addEventListener('change', drawCircuit);
    
    if(elBtnLayerDefault) elBtnLayerDefault.addEventListener('click', () => setLayerPreset('default'));
    if(elBtnLayerAll) elBtnLayerAll.addEventListener('click', () => setLayerPreset('all'));

    // Input Attribution Listeners
    if(elAttrBpMode) elAttrBpMode.addEventListener('change', updateAttrBpOptions);
    if(elAttrBpStrategy) elAttrBpStrategy.addEventListener('change', updateAttrBpOptions);
    if(elBtnComputeAttr) elBtnComputeAttr.addEventListener('click', computeInputAttribution);
    if(elBtnSaveGraph) elBtnSaveGraph.addEventListener('click', saveCircuitGraph);
    const elBtnSavePdf = document.getElementById('btn-save-pdf');
    if(elBtnSavePdf) elBtnSavePdf.addEventListener('click', saveCircuitGraphPDF);
    
    // Attrib Save
    const elBtnSaveAttrPng = document.getElementById('btn-save-attr-png');
    const elBtnSaveAttrPdf = document.getElementById('btn-save-attr-pdf');
    if(elBtnSaveAttrPng) elBtnSaveAttrPng.addEventListener('click', saveAttributionMapPNG);
    if(elBtnSaveAttrPdf) elBtnSaveAttrPdf.addEventListener('click', saveAttributionMapPDF);
    
    
    if(elPruningMode) elPruningMode.addEventListener('change', updatePruningControls);

    console.log("Event listeners attached.");
} catch(e) {
    console.error("Error attaching listeners:", e);
}

// Visualization Controls (Slider <-> Input Sync)
function bindControl(slider, input, callback, debounceMs = 0) {
    if (!slider || !input) {
        // console.error("bindControl missing elements", slider, input);
        return;
    }
    
    let debounceTimer = null;
    const trigger = () => {
        if (debounceMs > 0) {
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                if(callback) callback();
            }, debounceMs);
        } else {
            if(callback) callback();
        }
    };

    // Slider updates Input
    slider.addEventListener('input', () => {
        input.value = slider.value;
        trigger();
    });
    // Input updates Slider
    input.addEventListener('input', () => {
        let val = parseFloat(input.value);
        if (!isNaN(val)) {
            // Clamp to slider limits
            val = Math.max(parseFloat(slider.min), Math.min(parseFloat(slider.max), val));
            slider.value = val;
            trigger();
        }
    });
}
console.log("Binding controls...");
// Bind with specific update functions for efficiency
// Layout updates (heavy): Edge/Node Thresholds, Spacing - Debounced
// bindControl(elVisThreshold, elValVisThreshold, updateCircuitLayout, 300); 
// bindControl(elVisNodeThreshold, elValVisNodeThreshold, updateCircuitLayout, 300);

// NEW Bindings for TopP (Callback: computeCircuit, Debounced)
bindControl(elVisTopP, elValVisTopP, computeCircuit, 600);

// Layout Geometry updates (moved to light): Spacing
bindControl(elVisLayerSpacing, elValVisLayerSpacing, drawCircuit, 0);
// Visual updates (light): Strength
bindControl(elVisStrength, elValVisStrength, drawCircuit, 0);

function updatePruningControls() {
    const mode = elPruningMode.value;
    if (mode === 'by_per_layer_cum_mass_percentile') {
        elCtrlTopP.classList.remove('hidden');
        elCtrlGlobalThresh.classList.add('hidden');
    } else {
        elCtrlTopP.classList.add('hidden');
        elCtrlGlobalThresh.classList.remove('hidden');
    }
}
// Init State
updatePruningControls();



// Functions

async function loadModel() {
    console.log("loadModel called");
    updateStatus('Loading...', 'normal');
    showLoading("Loading Model...");
    elBtnLoad.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/load_model`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                model_path: elModelPath.value,
                quantization_4bit: elQuant.checked,
                dtype: elModelDtype.value,
                revision: elModelRevision ? elModelRevision.value : null,
                lrp_rule: document.getElementById('model-lrp-rule').value
            })
        });
        
        const data = await response.json();
        if (response.ok) {
            updateStatus('Loaded', 'success');
            elBtnLogits.disabled = false;
            if(data.n_layers) {
                appState.n_layers = data.n_layers;
                // Update default presets
                generateLayerPresets();
            }
        } else {
            throw new Error(data.detail || 'Failed to load model');
        }
    } catch (e) {
        updateStatus(`Error: ${e.message}`, 'error');
        alert(e.message);
    } finally {
        elBtnLoad.disabled = false;
        hideLoading();
    }
}

async function computeLogits(keepExtras = false) {
    elBtnLogits.disabled = true;
    showLoading("Computing Logits...");
    
    // If this is a fresh run (not adding a token), clear extras
    if (!keepExtras) {
        appState.extraTokens = [];
    }

    elSectionLogits.classList.add('hidden');
    if(elSectionInputAttr) elSectionInputAttr.classList.add('hidden');
    elSectionCircuit.classList.add('hidden');
    
    // Prepare extras
    const extraIds = [];
    const extraStrs = [];
    
    appState.extraTokens.forEach(item => {
        if(item.type === 'id') extraIds.push(item.value);
        else extraStrs.push(item.value);
    });
    
    try {
        const response = await fetch(`${API_BASE}/compute_logits`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                prompt: elPrompt.value,
                is_append_bos: elAppendBos.checked,
                topk: 50,
                extra_token_ids: extraIds,
                extra_token_strs: extraStrs,
                capture_mid: document.getElementById('capture-mid') ? document.getElementById('capture-mid').checked : false
            })
        });
        
        const res = await response.json();
        if (response.ok) {
            renderLogitsTable(res.data);
            appState.tokens = res.tokens;
            
            // Display token count
            if (elTokenCountDisplay) {
                elTokenCountDisplay.textContent = `Total Input Tokens: ${res.tokens.length}`;
            }
            
            elSectionLogits.classList.remove('hidden');
            if(elSectionInputAttr) elSectionInputAttr.classList.remove('hidden');
            elSectionCircuit.classList.remove('hidden');
            
            // Set default selections
            if (res.data.length > 0) {
                appState.selectedTargetToken = res.data[0];
                elTopTokenDisplay.textContent = `${appState.selectedTargetToken.token_str} (ID: ${appState.selectedTargetToken.token_id})`;
            }
        } else {
            alert(`Error: ${res.detail}`);
        }
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        elBtnLogits.disabled = false;
        hideLoading();
    }
}

function renderLogitsTable(data) {
    elLogitsTableBody.innerHTML = '';
    data.forEach(item => {
        // Robust escaping for the onclick handler string
        const escapedTokenStr = formatTokenForDisplay(item.token_str, 'data');

        // VISUAL ESCAPE for the table cell
        let visualTokenStr = formatTokenForDisplay(item.token_str, 'visual');

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${item.rank}</td>
            <td>${item.token_id}</td>
            <td>${visualTokenStr}</td>
            <td>${item.logit.toFixed(4)}</td>
            <td><button id="btn-select-${item.token_id}" class="btn-select-ref" onclick="selectContrast(${item.token_id}, '${escapedTokenStr}', ${item.logit})">Select for Contrast</button></td>
            <td>
                <button class="btn-generate-cont" onclick="generateContinuation(${item.token_id}, this)">Generate</button>
                <div class="gen-result" style="font-size: 0.85em; color: #555; margin-top: 4px; max-width: 300px; white-space: pre-wrap;"></div>
            </td>
        `;
        elLogitsTableBody.appendChild(row);
    });
}

window.generateContinuation = async function(tokenId, btn) {
    const originalText = btn.textContent;
    btn.disabled = true;
    const elGenMax = document.getElementById('gen-max-tokens');
    const maxNewTokens = elGenMax ? parseInt(elGenMax.value) : 30;
    
    btn.textContent = `Generating (${maxNewTokens})...`;
    const resultDiv = btn.nextElementSibling;
    resultDiv.textContent = "";

    try {
        const payload = {
            prompt: elPrompt.value,
            max_new_tokens: maxNewTokens,
            append_token_id: tokenId
        };

        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Generation failed");
        }

        const data = await response.json();
        resultDiv.textContent = data.generated_text;

    } catch(e) {
        resultDiv.textContent = "Error: " + e.message;
        resultDiv.style.color = "red";
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
};

window.selectContrast = function(id, name, logit) {
    appState.selectedContrastToken = { id, name, logit };
    
    // Calculate Diff
    let diffText = "";
    if (appState.selectedTargetToken) {
        // Find top logit (usually the first one in the list if sorted, or stored in appState)
        // appState.selectedTargetToken comes from computeLogits result[0]
        const topLogit = appState.selectedTargetToken.logit;
        const diff = topLogit - logit;
        diffText = ` | Logit Diff: ${diff.toFixed(4)}`;
    }

    elContrastTokenDisplay.textContent = `${name} (ID: ${id})${diffText}`;
    
    // Visual Updates
    document.querySelectorAll('.btn-select-ref').forEach(btn => {
        btn.textContent = "Select for Contrast";
        btn.classList.remove('selected');
        btn.style.backgroundColor = ""; // Reset inline style if any (or rely on class)
    });
    
    const activeBtn = document.getElementById(`btn-select-${id}`);
    if (activeBtn) {
        activeBtn.textContent = "Selected for Contrast";
        activeBtn.classList.add('selected');
    }
    
    // Auto Update UI settings
    elBpMode.value = "logit_diff";
    updateBpOptions();
    elBpStrategy.value = "by_ref_token";
    updateBpOptions();
    elBpRefId.value = id;

    // Auto Update Input Attribution settings
    if (elAttrBpMode) {
        elAttrBpMode.value = "logit_diff";
        updateAttrBpOptions();
    }
    if (elAttrBpStrategy) {
        elAttrBpStrategy.value = "by_ref_token";
        updateAttrBpOptions();
    }
    if (elAttrBpRefId) {
        elAttrBpRefId.value = id;
    }
};

function updateBpOptions() {
    const isDiff = elBpMode.value === "logit_diff";
    document.getElementById('diff-strategy-opts').style.display = isDiff ? 'block' : 'none';
    
    if (isDiff) {
        const strategy = elBpStrategy.value;
        elOptTopk.classList.toggle('hidden', strategy !== 'by_topk_avg');
        elOptRef.classList.toggle('hidden', strategy !== 'by_ref_token');
    }
}

function updateAttrBpOptions() {
    const isDiff = elAttrBpMode.value === "logit_diff";
    if (elAttrDiffStrategyOpts) {
        elAttrDiffStrategyOpts.style.display = isDiff ? 'block' : 'none';
    }
    
    if (isDiff) {
        const strategy = elAttrBpStrategy.value;
        if (elAttrOptRef) {
             elAttrOptRef.classList.toggle('hidden', strategy !== 'by_ref_token');
        }
    }
}

async function computeInputAttribution() {
    if (!appState.selectedTargetToken) {
        alert("Please calculate logits and select a target token first.");
        return;
    }

    // Default values if inputs are missing (e.g. reused from circuit config or hardcoded)
    const bpK = 10; 

    const payload = {
        target_token_id: appState.selectedTargetToken.token_id,
        contrast_token_id: appState.selectedContrastToken ? appState.selectedContrastToken.id : null,
        backprop_config: {
            mode: elAttrBpMode.value,
            strategy: elAttrBpStrategy.value,
            k: bpK,
            ref_token_id: elAttrBpRefId && elAttrBpRefId.value ? parseInt(elAttrBpRefId.value) : null
        }
    };

    elInputAttributionDisplay.innerHTML = "Computing...";
    elBtnComputeAttr.disabled = true;

    try {
        const resp = await fetch(`${API_BASE}/compute_input_attribution`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            const err = await resp.json();
            const errMsg = (typeof err.detail === 'object') ? JSON.stringify(err.detail) : (err.detail || "Request failed");
            throw new Error(errMsg);
        }

        const data = await resp.json();
        // data.relevance is list of floats
        appState.lastRelevance = data.relevance;
        renderAttributionMap(data.relevance, appState.tokens);

    } catch (e) {
        console.error(e);
        elInputAttributionDisplay.textContent = "Error: " + e.message;
    } finally {
        elBtnComputeAttr.disabled = false;
    }
}
if (elAttrHideBos) {
    elAttrHideBos.addEventListener('change', () => {
        if (appState.lastRelevance && appState.tokens) {
            renderAttributionMap(appState.lastRelevance, appState.tokens);
        }
    });
}


function renderAttributionMap(relevance, tokens) {
    elInputAttributionDisplay.innerHTML = "";
    
    if (!relevance || relevance.length === 0) {
        elInputAttributionDisplay.textContent = "No data returned.";
        return;
    }

    const hideBos = elAttrHideBos ? elAttrHideBos.checked : false;
    
    // Determine range for normalization (exclude BOS if hideBos is true)
    // But we render everything from index 0 now, just styling BOS differently.
    
    let maxAbs = 0;
    // Calculation Loop
    for (let i = 0; i < relevance.length; i++) {
        if (hideBos && i < 1) continue; // Skip BOS for dynamic range calculation
        const r = relevance[i];
        if (Math.abs(r) > maxAbs) maxAbs = Math.abs(r);
    }
    if (maxAbs === 0) maxAbs = 1;

    // Helper for color
    function getColor(val) {
        if (Math.abs(val) === 0) return 'transparent'; // optimization
        
        // Red for positive, Blue for negative
        const norm = val / maxAbs;
        
        // Use HSL for better control? Or RGBA.
        // We want a clear white text if background is dark? 
        // Or keep background light.
        // Let's use slight alpha backgrounds.
        
        const alpha = Math.abs(norm); 
        // Cap alpha to avoid being too dark/unreadable if we don't change text color
        // But we want it visible.
        const cappedAlpha = Math.min(alpha, 0.6); 

        if (norm > 0) {
            return `rgba(255, 0, 0, ${cappedAlpha})`; 
        } else {
            return `rgba(0, 0, 255, ${cappedAlpha})`;
        }
    }

    tokens.forEach((tok, idx) => {
        // Handle mismatch length if any (shouldn't happen)
        if (idx >= relevance.length) return;

        const val = relevance[idx];
        const span = document.createElement("span");
        span.className = "token-span";
        
        // Check if this is a "Hidden" BOS token
        const isHiddenBos = hideBos && idx === 0;
        
        if (isHiddenBos) {
            // Apply special style
            span.style.color = "#aaa";
            span.style.backgroundColor = "#f0f0f0";
            // span.style.textDecoration = "line-through"; // Optional? 
        }

        // tok is a string from backend
        let tokenText = (typeof tok === 'string') ? tok : (tok.token_str || tok.text);
        
        // Use Central Formatter
        const visualText = formatTokenForDisplay(tokenText, 'visual');
        
        // Special structural handling for newlines
        const isNewline = (tokenText === '\n' || tokenText === '\r\n');
        const isDoubleNewline = (tokenText === '\n\n' || tokenText === '\r\n\r\n');

        if (isNewline || isDoubleNewline) {
             span.textContent = visualText;
             // Ensure it has shape/padding
             span.style.display = "inline-block";
             span.style.minWidth = isDoubleNewline ? "24px" : "12px";
             span.style.textAlign = "center";
        } else {
             span.textContent = tokenText; // Keep original for normal text? OR use visual? 
             // Usually normal text is fine, but if it has internal newlines, visualText handles it.
             // Let's use visualText if it differs significantly? No, visualText escapes \n.
             // For attribution map, we want visible \n, but regular text should probably wrap?
             // Actually, for wrapped text block, literal \n is confusing.
             // But valid tokens usually don't have internal newlines except specifically newline tokens.
             // Let's stick to original tokenText for normal tokens, just in case.
             // But if visualText detected a change (like mixed \n), we might want to show it.
             if (visualText !== tokenText) span.textContent = visualText;
             else span.textContent = tokenText;
        }
        
        const normVal = val / maxAbs;
        if (!isHiddenBos) {
            span.style.backgroundColor = getColor(val);
        }
        span.title = `Token: "${visualText}"\nPos: ${idx}\nRaw Rel: ${val.toFixed(5)}\nNorm Rel: ${normVal.toFixed(3)}`;
        
        elInputAttributionDisplay.appendChild(span);
        
        // Preserve line break behavior for newlines
        if (isNewline) {
             elInputAttributionDisplay.appendChild(document.createElement("br"));
        } else if (isDoubleNewline) {
             elInputAttributionDisplay.appendChild(document.createElement("br"));
             elInputAttributionDisplay.appendChild(document.createElement("br"));
        }
    });
}

async function computeCircuit() {
    // Parse Layers List
    // Expect: "0, 5, 20" -> [0, 5, 20]
    const rawVal = elLayersList.value;
    const layers = rawVal.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    
    // Sort and Validate
    layers.sort((a,b) => a - b);
    
    if (layers.length < 2) {
        alert("Please provide at least 2 layers (e.g., '0, 5, 20')");
        return;
    }
    
    elLayersList.value = layers.join(', ');
    
    elBtnCircuit.disabled = true;
    elBtnCircuit.textContent = "Computing...";
    
    // Progress UI
    const elProgressContainer = document.getElementById("compute-progress-container");
    const elProgressBar = document.getElementById("compute-progress-bar");
    const elProgressStatus = document.getElementById("progress-status-text");
    const elProgressPercent = document.getElementById("progress-percent-text");

    if(elProgressContainer) {
        elProgressContainer.style.display = "block";
        elProgressBar.value = 0;
        elProgressStatus.textContent = "Starting...";
        elProgressPercent.textContent = "0%";
    }
    
    // Build Backprop Config
    const bpConfig = {
        mode: elBpMode.value,
        strategy: elBpStrategy.value,
        k: parseInt(elBpK.value) || 10,
        ref_token_id: parseInt(elBpRefId.value) || 0,
        contrast_rank: 2 // simplified defaults
    };

    // Pruning Config
    const pruningMode = elPruningMode ? elPruningMode.value : "by_per_layer_cum_mass_percentile";
    const topP = elVisTopP ? parseFloat(elVisTopP.value) : 0.9;
    const edgeThresh = elValGlobalThresh ? parseFloat(elValGlobalThresh.value) : 0.01;
    
    try {
        const response = await fetch(`${API_BASE}/compute_circuit`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                backprop_config: bpConfig,
                layers: layers,
                pruning_mode: pruningMode,
                top_p: topP,
                edge_threshold: edgeThresh
            })
        });
        
        if (!response.ok) {
             // Try to parse error
             let errMsg = "Server Error";
             try {
                const err = await response.json();
                errMsg = err.detail || errMsg;
             } catch(e) {}
             throw new Error(errMsg);
        }
        
        // NDJSON Streaming Reader
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        // Clear previous data
        appState.circuitData = null;
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, {stream: true});
            buffer += chunk;
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep partial
            
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg.type === "progress") {
                        if(elProgressContainer) {
                            const pct = Math.min(100, Math.max(0, msg.percent || 0));
                            elProgressBar.value = pct;
                            elProgressStatus.textContent = msg.msg;
                            elProgressPercent.textContent = Math.round(pct) + "%";
                        }
                    } else if (msg.type === "graph_data") {
                        // Received FULL graph
                        appState.circuitData = {
                            graph: msg.graph,
                            pruning_details: msg.pruning_details,
                            layers: layers // Or extract from graph nodes
                        };
                        appState.layout = null; // Clear cached layout
                    } else if (msg.type === "complete") {
                         if (appState.circuitData) {
                            requestAnimationFrame(() => updateCircuitLayout()); // directly to layout
                        }
                    } else if (msg.type === "error") {
                        throw new Error(msg.msg);
                    }
                } catch (e) {
                     if (e instanceof SyntaxError) {
                        console.error("JSON Parse Error on line:", line, e);
                        continue;
                     }
                     throw e; 
                }
            }
        }
    } catch (e) {
        alert(e.message);
    } finally {
        elBtnCircuit.disabled = false;
        elBtnCircuit.textContent = "Visualize Connection";
        if(elProgressContainer) {
             setTimeout(() => {
                 elProgressContainer.style.display = "none";
             }, 2000);
        }
    }
}

function updateStatus(msg, type) {
    elStatus.textContent = msg;
    elStatus.className = `status ${type}`;
}

// --- Visualization Logic ---

// NEW: Separated heavy layout calculation from rendering
function updateCircuitLayout() {
    if (!appState.circuitData || !appState.circuitData.graph) return;
    showLoading("Updating Layout...");
    
    // Yield to browser to render spinner
    setTimeout(() => {
        try {
            _updateCircuitLayoutInternal();
        } catch(e) {
            console.error(e);
        } finally {
             hideLoading();
        }
    }, 50);
}

function _updateCircuitLayoutInternal() {
    if (!appState.circuitData || !appState.circuitData.graph) return;
    
    // Destructure graph early
    const { graph, layers: requestedLayers } = appState.circuitData;
    
    // Define robust accessors
    // NetworkX alignment: v2 uses 'links', v3 uses 'edges'
    const nodes = graph.nodes || [];
    const edges = graph.edges || graph.links || [];
    
    // DEBUG: Log Graph Data Structure
    console.log("Graph Data Received:", graph);
    if(nodes.length > 0) {
        console.log("Sample Node:", nodes[0]);
    }
    if(edges.length > 0) {
        console.log("Sample Link/Edge:", edges[0]);
    } else {
        console.log("No links/edges found. (Graph might be disconnected or threshold too high)");
    }
    
    // Check tokens
    if (!appState.tokens || appState.tokens.length === 0) {
        console.error("appState.tokens is missing! Cannot render graph. Please run Compute Logits first.");
        alert("Error: Tokens missing. Please run 'Compute Logits' first.");
        return;
    }

    // Check toggle listeners
    const toggle = document.getElementById('hide-bos-node');
    if (toggle && !toggle.hasAttribute('data-listening')) {
        toggle.addEventListener('change', updateCircuitLayout); // Need layout update to filter BOs
        toggle.setAttribute('data-listening', 'true');
    }
    const showAllToggle = document.getElementById('show-all-tokens');
    if (showAllToggle && !showAllToggle.hasAttribute('data-listening')) {
        showAllToggle.addEventListener('change', updateCircuitLayout); 
        showAllToggle.setAttribute('data-listening', 'true');
    }
    
    const showNodeValuesToggle = document.getElementById('show-node-values');
    if (showNodeValuesToggle && !showNodeValuesToggle.hasAttribute('data-listening')) {
        showNodeValuesToggle.addEventListener('change', drawCircuit); // Visual only
        showNodeValuesToggle.setAttribute('data-listening', 'true');
    }

    // const { graph, layers: requestedLayers } = appState.circuitData; // Moved up
    const tokens = appState.tokens;
    const layerHeight = parseFloat(elVisLayerSpacing ? elVisLayerSpacing.value : 300);
    
    const fullSeqLen = tokens.length;
    const hideFirstToken = document.getElementById('hide-bos-node') && document.getElementById('hide-bos-node').checked;
    const showAllTokens = elShowAllTokens ? elShowAllTokens.checked : true;
    
    // Data Slicing (if hiding BOS)
    let displaySeqLen = fullSeqLen;
    let displayTokens = tokens;
    let startTokenIdx = 0;
    
    if (hideFirstToken && fullSeqLen > 1) {
        displayTokens = tokens.slice(1);
        displaySeqLen = fullSeqLen - 1;
        startTokenIdx = 1;
    }

    // Process Graph Data
    // Nodes are { id: [layer, token], layer: L, token: T, relevance: R }
    // const nodes = graph.nodes; // Moved up
    // const edges = graph.links; // Moved up
    
    // Identify Layers involved in graph
    // Use requestedLayers if available, or infer from graph
    let layersList = requestedLayers || [];
    if (layersList.length === 0) {
        const layersSet = new Set(nodes.map(n => n.layer));
        layersList = Array.from(layersSet).sort((a,b)=>a-b);
    }
    
    // Determine active nodes (those in graph)
    const activeNodeMap = new Map(); // key: "layer,token" -> nodeObj
    let globalMaxValRel = 0;
    let globalMaxValEdge = 0;

    nodes.forEach(n => {
        // Filter BOS if needed
        if (hideFirstToken && n.token === 0) return; // Assume BOS is 0? Use n.token index check
        if (hideFirstToken && n.token < startTokenIdx) return;
        
        const key = `${n.layer},${n.token}`;
        activeNodeMap.set(key, n);
        globalMaxValRel = Math.max(globalMaxValRel, Math.abs(n.relevance || 0));
    });
    
    edges.forEach(e => {
        // e.source and e.target are arrays [L, T]
        const srcL = e.source[0];
        const srcT = e.source[1];
        const tgtL = e.target[0];
        const tgtT = e.target[1];
        
        if (hideFirstToken && (srcT < startTokenIdx || tgtT < startTokenIdx)) return;
        
        globalMaxValEdge = Math.max(globalMaxValEdge, Math.abs(e.weight || 0));
    });
    
    const maxNorm = Math.max(globalMaxValRel, globalMaxValEdge, 1e-9);

    // Layout Geometry Setup
    const nHops = layersList.length > 1 ? layersList.length - 1 : 1; 
    const totalHeight = Math.max(500, nHops * layerHeight + 150);
    const margin = { top: 50, left: 120, right: 50, bottom: 50 };
    
    
    // Generate Layout Nodes
    const nodesByLayer = {}; 
    const allNodesFlat = [];
    const layerTotals = {};
    
    // Helper to get active node data
    const getActiveData = (l, t) => activeNodeMap.get(`${l},${t}`);

    // DYNAMIC LAYOUT LOGIC
    // 1. Identify "Active Columns": Any token index that participates in an EDGE.
    // This filters out columns that might have nodes but no connections (pruned).
    let activeTokenIndices = new Set();
    
    // Use edges to find connected tokens
    edges.forEach(e => {
        // e.source/target are [layer, token]
        activeTokenIndices.add(e.source[1]); 
        activeTokenIndices.add(e.target[1]);
    });
    
    // Also include target node tokens? 
    // Usually target is connected, but what if we have a top-p that prunes EVERYTHING?
    // We should probably show the target node column regardless, so the user sees where they started.
    // If backend prunes everything, we might have edges=[], nodes=[target].
    // Let's iterate nodes too, but check a flag or just rely on edges.
    // User Request: "hide the token positions whose nodes in all layers do not have any connections".
    // This implies strictly edges.
    // BUT: If the target node has no incoming edges (e.g. threshold too high), it is isolated.
    // Should we hide the target node? That seems confusing.
    // Let's prioritize edges, but maybe keep target? 
    // For now, strict edge adherence based on user prompt.
    // If empty set (graph empty), we might show nothing or just full seq?
    if (activeTokenIndices.size === 0 && !showAllTokens && nodes.length > 0) {
        // Fallback: Show at least the nodes that exist (like target)
        nodes.forEach(n => activeTokenIndices.add(n.token));
    }

    // 2. Define "Render Columns"
    let renderColumns = [];
    if (showAllTokens) {
        // All tokens in display range
        for(let i=0; i<displaySeqLen; i++) renderColumns.push(startTokenIdx + i);
    } else {
        // Only active tokens
        renderColumns = Array.from(activeTokenIndices).sort((a,b) => a - b);
        // Filter by hideFirstToken if needed (activeNodeMap already filtered? No, activeNodeMap built from nodes list)
        // Ensure strictly respecting hideFirstToken setting
        if (hideFirstToken) {
            renderColumns = renderColumns.filter(t => t >= startTokenIdx);
        }
    }
    
    // 3. Spacing based on Render Columns count
    const nodesInRowForSpacing = renderColumns.length; 
    const minW = nodesInRowForSpacing * (showAllTokens ? 15 : 30) + 100;
    const width = Math.max(800, minW);
    const drawingW = width - margin.left - margin.right;
    const nodeSpacing = drawingW / Math.max(1, nodesInRowForSpacing);

    const yPositions = {};
    layersList.forEach((lIdx, i) => {
        yPositions[lIdx] = (totalHeight - margin.bottom) - (i * layerHeight);
    });

    layersList.forEach((lIdx) => {
        const y = yPositions[lIdx];
        const rowNodeMap = {}; 
        let runningTotal = 0;
        
        // Iterate Render Columns
        renderColumns.forEach((tokenIdx, visualColIdx) => {
            const activeData = getActiveData(lIdx, tokenIdx);
            const rel = activeData ? (activeData.relevance || 0) : 0;
            const normRel = Math.abs(rel) / maxNorm;
            
            // accumulate total only if we want layer sum to reflect EVERYTHING or just visible?
            // Usually layer sum is total relevance. If we hide tokens, do we hide their relevance from sum?
            // Let's sum only active data found in graph.
            if (activeData) runningTotal += rel;
            
            // X alignment: Based on visual column index
            const x = margin.left + visualColIdx * nodeSpacing + (nodeSpacing/2);
            
            // Token String
            const rawToken = (tokenIdx < tokens.length) ? tokens[tokenIdx] : "?";
            const tokenStr = (typeof rawToken === 'string') ? rawToken : (rawToken.token_str || "?");

            const node = {
                layer: lIdx,
                index: tokenIdx, // True token index
                visualIndex: visualColIdx, // For gap logic
                token: { token_str: tokenStr }, 
                x: x,
                y: y,
                rel: rel,
                normRel: normRel,
                isActive: !!activeData
            };
            allNodesFlat.push(node);
            rowNodeMap[tokenIdx] = node;
        });
        
        nodesByLayer[lIdx] = rowNodeMap;
        layerTotals[lIdx] = runningTotal;
    });

    // Generate Edges
    const visibleEdges = [];
    
    edges.forEach(e => {
        const srcL = e.source[0]; 
        const srcT = e.source[1]; 
        const tgtL = e.target[0]; 
        const tgtT = e.target[1];
        
        // Lookup in our generated layout nodes
        // nodesByLayer[layer][tokenIndex]
        if (!nodesByLayer[srcL] || !nodesByLayer[tgtL]) return;
        
        const sNode = nodesByLayer[srcL][srcT];
        const tNode = nodesByLayer[tgtL][tgtT];
        
        if (sNode && tNode) {
            const val = e.weight;
            const normVal = Math.abs(val) / maxNorm;
            
            visibleEdges.push({
                source: sNode,
                target: tNode,
                val: val,
                normVal: normVal
            });
        }
    });

    // STORE LAYOUT
    appState.layout = {
        width,
        totalHeight,
        margin,
        nodeSpacing,
        maxNorm,
        layersList,
        nodesByLayer,
        allNodesFlat,
        visibleEdges,
        activeIndicesList: [], // Not used in this sparse logic
        layerTotals,
        displaySeqLen,
        displayTokens, // Add this back for gap drawing
        displayConnections: [] // Legacy compat
    };
    
    drawCircuit();
}

function drawCircuit() {
    if (!appState.layout) {
        // Init layout if data exists
        if(appState.circuitData) updateCircuitLayout();
        return;
    }
    
    const L = appState.layout;
    
    // Check Spacing Updates first (Geometry Recalc)
    const currentLayerHeight = parseFloat(elVisLayerSpacing ? elVisLayerSpacing.value : 300);
    // Use stored layers count for recalculating total height
    // We infer layersList from layout
    const nHops = L.displayConnections ? L.displayConnections.length : 1; 
    let newTotalHeight = Math.max(500, nHops * currentLayerHeight + 150);
    
    if (L.layersList && L.layersList.length > 0) {
        newTotalHeight = Math.max(500, (L.layersList.length - 1) * currentLayerHeight + 200);

        // Pre-calculate Y for each layer (Optimization: O(Layers) instead of O(Nodes*Layers))
        const layerYMap = {};
        L.layersList.forEach((lid, idx) => {
            layerYMap[lid] = (newTotalHeight - L.margin.bottom) - (idx * currentLayerHeight);
        });

        // Update Node Y Positions in place
        L.allNodesFlat.forEach(node => {
             if (layerYMap.hasOwnProperty(node.layer)) {
                 node.y = layerYMap[node.layer];
             }
        });
        
        // Edge coordinates update automatically since they reference node objects
    }
    L.totalHeight = newTotalHeight;

    const ctx = elCanvas.getContext('2d');
    
    // Check canvas dims
    if (elCanvas.width !== L.width || elCanvas.height !== L.totalHeight) {
        elCanvas.width = L.width;
        elCanvas.height = L.totalHeight;
    }
    
    // Vis Settings
    const strengthScale = parseFloat(elVisStrength.value);
    const showNodeValues = document.getElementById('show-node-values') ? document.getElementById('show-node-values').checked : true;
    const showAllTokens = elShowAllTokens ? elShowAllTokens.checked : true;
    const hideFirstToken = document.getElementById('hide-bos-node') && document.getElementById('hide-bos-node').checked;
    
    // Add startTokenIdx logic here
    let startTokenIdx = 0;
    if (hideFirstToken && appState.tokens && appState.tokens.length > 1) {
        startTokenIdx = 1;
    }

    ctx.clearRect(0, 0, L.width, L.totalHeight);
    
    // --- Interaction ---
    const mouseX = appState.mouseX || -1;
    const mouseY = appState.mouseY || -1;
    function checkHit(x, y) { return Math.sqrt((x-mouseX)**2 + (y-mouseY)**2) < 8; }
    
    let hoveredNode = null;
    for (const n of L.allNodesFlat) {
        if (checkHit(n.x, n.y)) {
            hoveredNode = n;
            break;
        }
    }
    const activeNode = hoveredNode || selectedNode;
    const isHighlightActive = !!activeNode;
    
    // --- Highlight Computation (One-Hop) ---
    const highlightedNodes = new Set();
    const highlightedEdges = new Set(); // store edges to paint bold
    
    if (activeNode) {
        highlightedNodes.add(activeNode);
        
        // Use visibleEdges directly
        L.visibleEdges.forEach(e => {
             // 1. Outgoing
             if (e.source === activeNode) {
                 highlightedNodes.add(e.target);
                 highlightedEdges.add(e);
             }
             // 2. Incoming
             if (e.target === activeNode) {
                 highlightedNodes.add(e.source);
                 highlightedEdges.add(e);
             }
        });
    }

    // --- Draw Edges ---
    // Sort edges: Passive first, Active on top
    // Actually just draw active afterwards.
    
    L.visibleEdges.forEach(e => {
        const isActive = isHighlightActive && highlightedEdges.has(e);
        const isDim = isHighlightActive && !isActive;
        
        // If dim, maybe skip drawing very thin lines to save perf?
        // Or alpha.
        
        let width = e.normVal * strengthScale;
        if (width < 0.5 && isDim) return; // Culling
        
        width = Math.max(width, 0.5);   // Minimal width for visibility

        let alpha = isDim ? 0.1 : 1.0;
        if (isActive) {
            alpha = 1.0;
            width = Math.max(width, 1.0); // Boost active
        }
        
        const color = e.val >= 0 ? `rgba(211, 47, 47, ${alpha})` : `rgba(25, 118, 210, ${alpha})`;
        
        ctx.beginPath();
        ctx.moveTo(e.source.x, e.source.y);
        ctx.lineTo(e.target.x, e.target.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        
        // draw active later? No, simple batch is fine mostly.
        ctx.stroke();

        if (isActive) { // Use isActive logic for edge label drawing
            // Draw Edge Label
            const midX = (e.source.x + e.target.x)/2;
            const midY = (e.source.y + e.target.y)/2;
            ctx.save();
            const dx = e.target.x - e.source.x;
            const dy = e.target.y - e.source.y;
            let angle = Math.atan2(dy, dx);
            if (Math.abs(angle) > Math.PI / 2) angle += Math.PI; // Correct text orientation
            
            ctx.translate(midX, midY);
            ctx.rotate(angle);
            
            ctx.font = 'bold 10px Arial';
            const text = e.val.toFixed(2);
            const metrics = ctx.measureText(text);
            const p = 2; // padding
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.fillRect(-metrics.width/2 - p, -6, metrics.width + 2*p, 12);
            
            ctx.fillStyle = e.val >= 0 ? '#b71c1c' : '#0d47a1';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, 0, 0);
            ctx.restore();
        }
    });
    
    // --- Draw Nodes ---
    // Re-use Gap logic variables
    let prevIdx = -1;
    let listIdx = -1;
    const displayedGaps = [];
    const displayTokens = L.displayTokens;
    
    L.allNodesFlat.forEach((node) => {
        // Logic for tracking gap indices per layer
        if (node.layer === L.layersList[0]) {
            listIdx++;
        } else {
            // Reset for other layers? Actually gap logic only applied to bottom layer in original code
        }

        const isSelfActive = activeNode === node;
        const isConnected = highlightedNodes.has(node) && !isSelfActive;
        const isInteractive = isHighlightActive;
        const isDim = isInteractive && !isSelfActive && !isConnected;

        let radius = (node.normRel * strengthScale);
        radius = Math.max(radius, 2);
        if (isConnected) radius = Math.max(radius, strengthScale * 0.75); 
        if (isSelfActive) radius = Math.max(radius, strengthScale * 1.25);
        
        let fillStyle = '#e0e0e0';
        if (node.rel > 0.001) fillStyle = '#ef9a9a'; 
        else if (node.rel < -0.001) fillStyle = '#90caf9'; 
        else fillStyle = '#cfd8dc'; 

        if (isHighlightActive && !isDim) {
             if (node.rel > 0.001) fillStyle = '#d32f2f'; 
             else if (node.rel < -0.001) fillStyle = '#1976d2'; 
             else fillStyle = '#455a64'; 
        }

        if (isDim) {
            fillStyle = '#f5f5f5';
            radius = 3; 
        }
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, 2*Math.PI);
        ctx.fillStyle = fillStyle;
        ctx.fill();
        
        if (!isDim) {
             ctx.strokeStyle = (isSelfActive || isConnected) ? '#333' : '#bbb';
             ctx.lineWidth = 1;
             ctx.stroke();
        }
        
        // Labels (Layer)
        // Check if node is the first visible node in its layer OR if it's the very first node of the sequence
        // We use activeIndicesList logic if available, otherwise just use "first in row" logic
        const isFirstInLayer = (node.index === 0) || (Object.values(L.nodesByLayer[node.layer])[0] === node);

        if (isFirstInLayer) { 
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 12px Arial';
            ctx.textAlign = 'right';
            
            let label = "";
            if (Array.isArray(node.layer)) {
                 label = `L${node.layer[0]} (${node.layer[1]})`;
            } else {
                 label = node.layer === -1 ? "Embedding" : `Layer ${node.layer}`;
            }
            
            ctx.fillText(label, L.margin.left - 20, node.y + 4);            
            const total = L.layerTotals[node.layer] || 0;
            ctx.font = '11px Arial';
            ctx.fillText(`Sum: ${total.toFixed(2)}`, L.margin.left - 20, node.y + 18);
        }
        
        // Labels (Tokens)
        const isBottomLayer = (node.layer === L.layersList[0]);
        let showToken = isBottomLayer; 
        if (isHighlightActive && (isSelfActive || isConnected)) showToken = true;
        
        if (showToken) {
             ctx.fillStyle = '#000';
             ctx.font = isSelfActive ? 'bold 11px Arial' : '10px Arial';
             ctx.textAlign = 'center';
             ctx.save();
             
             if (isBottomLayer && node.layer !== L.layersList[L.layersList.length-1]) {
                 ctx.translate(node.x, node.y + 15);
                 ctx.rotate(Math.PI/4);
             } else {
                 ctx.translate(node.x, node.y - 12);
                 ctx.rotate(-Math.PI/4);
             }
             
             let txt = formatTokenForDisplay(node.token.token_str, 'visual');
             ctx.fillText(txt, 0, 0);
             ctx.restore();
             
             // Gap Marker Logic (Bottom Layer Only)
             if (isBottomLayer && !showAllTokens) {
                  // Preceding Gap (Before first visible token)
                  // If this is the first visible token (listIdx or visualIndex 0), check if there are hidden tokens before it
                  const trueStartIdx = hideFirstToken ? startTokenIdx : 0;
                  
                  if (node.visualIndex === 0 && node.index > trueStartIdx) {
                      const gx = node.x - L.nodeSpacing/2;
                      const gy = node.y + 45;
                      const hidden = [];
                      for(let k=trueStartIdx; k<node.index; k++) {
                          if(displayTokens && k < displayTokens.length) {
                              const tObj = displayTokens[k];
                              const tStr = (typeof tObj === 'string') ? tObj : tObj.token_str;
                              hidden.push(formatTokenForDisplay(tStr, 'visual'));
                          }
                      }
                      if (hidden.length > 0) displayedGaps.push({x: gx, y: gy, hidden: hidden});
                  }
                  
                  // Inter-token Gap
                  // Previous Visible Node Index vs Current Node Index
                  if (prevIdx !== -1 && (node.index - prevIdx) > 1) {
                      // We can use visualIndex to backtrack to previous node X
                      // node.visualIndex is current i. Previous was i-1.
                      // xPrev = x of column (node.visualIndex - 1)
                      const xPrev = node.x - L.nodeSpacing;
                      
                      const gx = (xPrev + node.x)/2;
                      const gy = node.y + 45;
                      const hidden = [];
                      for(let k=prevIdx+1; k<node.index; k++) {
                            if(displayTokens && k < displayTokens.length) {
                                const tObj = displayTokens[k];
                                const tStr = (typeof tObj === 'string') ? tObj : tObj.token_str;
                                hidden.push(formatTokenForDisplay(tStr, 'visual'));
                            }
                      }
                      if (hidden.length > 0) displayedGaps.push({x: gx, y: gy, hidden: hidden});
                  }
                  prevIdx = node.index;
             }
        }

        if (showNodeValues) {
             const valText = node.rel.toFixed(2);
             ctx.font = '9px Arial';
             ctx.fillStyle = node.rel >= 0 ? '#b71c1c' : '#0d47a1';
             ctx.textAlign = 'center';
             let textY = node.y - 14; 
             if (!isBottomLayer) textY = node.y - 16; 
             if (isSelfActive) ctx.font = 'bold 10px Arial';
             ctx.fillText(valText, node.x, textY);
        }
    });
    // --- Draw Gaps Loop ---
    let hoveredGap = null;
    // const { displayTokens } = L; // Already defined above

    displayedGaps.forEach(gap => {
        const mx = appState.mouseX;
        const my = appState.mouseY;
        const dist = Math.sqrt(Math.pow(gap.x - mx, 2) + Math.pow(gap.y - my, 2));
        
        if (dist < 15) hoveredGap = gap;
        
        ctx.save();
        ctx.translate(gap.x, gap.y);
        ctx.fillStyle = (dist < 15) ? '#e0e0e0' : '#f5f5f5';
        ctx.strokeStyle = '#bdbdbd';
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        if(ctx.roundRect) ctx.roundRect(-12, -8, 24, 16, 4);
        else ctx.rect(-12, -8, 24, 16); 
        ctx.fill();
        ctx.stroke();
        
        ctx.fillStyle = '#616161';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText("...", 0, -2);
        ctx.restore();
    });
    
    // --- Tooltip Updating ---
    const elTooltip = document.getElementById('tooltip');
    if (hoveredGap && elTooltip) {
        elTooltip.style.display = 'block';
        elTooltip.style.opacity = '1';
        
        // Update content FIRST to measure size
        const fullText = hoveredGap.hidden ? hoveredGap.hidden.join(" ") : "";
        elTooltip.textContent = fullText || "[Empty]";
        elTooltip.style.maxWidth = "300px";

        const rect = elCanvas.getBoundingClientRect();
        const gapX = rect.left + hoveredGap.x;
        const gapY = rect.top + hoveredGap.y;
        
        // Default: Bottom-Right
        let finalLeft = gapX + 10;
        let finalTop = gapY + 10;
        
        // Check Bounds
        const tooltipRect = elTooltip.getBoundingClientRect();
        
        // Vertical overflow (flip up)
        if (finalTop + tooltipRect.height > window.innerHeight) {
            finalTop = gapY - tooltipRect.height - 10;
        }
        
        // Horizontal overflow (flip left)
        if (finalLeft + tooltipRect.width > window.innerWidth) {
            finalLeft = gapX - tooltipRect.width - 10;
        }
        
        elTooltip.style.left = finalLeft + 'px'; 
        elTooltip.style.top = finalTop + 'px';
        
    } else if (elTooltip) {
        elTooltip.style.display = 'none';
        elTooltip.style.opacity = '0';
    }
    
    // Mouse Interaction Tracking
    // Note: Re-binding these every draw call is inefficient but matches original logic structure.
    // Ideally move these out of drawCircuit.
    elCanvas.onmousemove = function(e) {
         const rect = elCanvas.getBoundingClientRect();
         appState.mouseX = e.clientX - rect.left;
         appState.mouseY = e.clientY - rect.top;
         requestAnimationFrame(drawCircuit);
    };
    
    elCanvas.onmouseleave = function() {
        appState.mouseX = -1;
        appState.mouseY = -1;
        requestAnimationFrame(drawCircuit);
    };

    elCanvas.onclick = function(e) {
        const rect = elCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        let hit = null;
        if(L && L.allNodesFlat) {
            for (const n of L.allNodesFlat) {
                 if (Math.sqrt(Math.pow(x - n.x, 2) + Math.pow(y - n.y, 2)) < 10) {
                     hit = n;
                     break;
                 }
            }
        }
        
        if (hit) {
            if (activeNode && activeNode.layer === hit.layer && activeNode.index === hit.index && !hoveredNode) {
                selectedNode = null; 
            } else {
                selectedNode = hit;
            }
        } else {
            selectedNode = null;
        }
        requestAnimationFrame(drawCircuit);
    };
}

// Initialization and Cleanup
window.addEventListener('DOMContentLoaded', async () => {
    // Call cleanup endpoint to release GPU memory on page refresh
    try {
        await fetch(`${API_BASE}/cleanup`, { method: 'POST' });
        console.log("Backend memory cleaned up.");
    } catch (e) {
        console.warn("Failed to cleanup backend memory:", e);
    }
    
    // Init Datasets
    await fetchDatasets();
});

// --- Trace Loading Logic ---

async function fetchDatasets() {
    try {
        const resp = await fetch(`${API_BASE}/datasets`);
        const data = await resp.json();
        
        elTraceDataset.innerHTML = '<option value="">-- Select --</option>';
        data.datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = ds;
            opt.textContent = ds;
            elTraceDataset.appendChild(opt);
        });
    } catch (e) {
        console.error("Failed to fetch datasets:", e);
    }
}

async function fetchTraces(dataset) {
    elTraceFile.disabled = true;
    elTraceFile.innerHTML = '<option value="">Loading...</option>';
    showLoading("Fetching Traces...");
    
    try {
        const resp = await fetch(`${API_BASE}/traces/${dataset}`);
        if (!resp.ok) throw new Error("Failed");
        
        const data = await resp.json();
        
        elTraceFile.innerHTML = '<option value="">-- Select Trace --</option>';
        data.traces.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = t;
            elTraceFile.appendChild(opt);
        });
        elTraceFile.disabled = false;
    } catch (e) {
        console.error("Failed to fetch traces:", e);
        elTraceFile.innerHTML = '<option value="">Error</option>';
    } finally {
        hideLoading();
    }
}

async function loadTraceDetails(dataset, traceId) {
    showLoading("Loading Trace Config...");
    try {
        const resp = await fetch(`${API_BASE}/trace_details/${dataset}/${traceId}`);
        if (!resp.ok) throw new Error("Failed");
        
        const data = await resp.json();
        
        // Update UI
        if(data.model_path) elModelPath.value = data.model_path;
        if(data.prompt) elPrompt.value = data.prompt;

        // Populate "Prompt + Original Completion"
        if(elPromptOrig) {
            // Prefer raw_prompt if available (added to backend), else prompt (which might be full concat)
            const basePrompt = data.raw_prompt !== undefined ? data.raw_prompt : (data.prompt || "");
            elPromptOrig.value = basePrompt + (data.completion || "");
        }
        
        // Set Config Defaults as requested
        elQuant.checked = data.quantization; // Default false
        elModelDtype.value = data.dtype || "bfloat16"; 
        
        // Determine "Other" candidates data
        let otherData = null;
        let otherLabel = "Other Model Candidates";

        if (data.other_candidates && Object.keys(data.other_candidates).length > 0) {
            // Pick first likely candidate key
            const keys = Object.keys(data.other_candidates);
            const key = keys[0];
            otherData = data.other_candidates[key];
            
            // Heuristic for label
            if (key === '4b') otherLabel = "Qwen3-4B Candidates";
            else if (key === '1.7b' || key === '1_7b') otherLabel = "Qwen3-1.7B Candidates";
            else if (key.toLowerCase().includes('qwen')) otherLabel = key;
            else otherLabel = key + " Candidates";
            
        } else if (data.topk_token_explore_4b && data.topk_token_explore_4b.length) {
            // Legacy Fallback
            otherData = data.topk_token_explore_4b;
            otherLabel = "Qwen3-4B Candidates";
        }

        // Render Exploration Data if available
        if (elSectionTraceExplore) {
            let hasContent = false;
            
            if (data.topk_token_explore && data.topk_token_explore.length) {
                renderExploreTable(elContainerExploreOriginal, data.topk_token_explore);
                hasContent = true;
            } else {
                elContainerExploreOriginal.innerHTML = "<div style='padding:10px;'>No data</div>";
            }
            
            if (otherData && otherData.length) {
                renderExploreTable(elContainerExplore4b, otherData);
                if (elHeaderExploreOther) elHeaderExploreOther.textContent = otherLabel;
                hasContent = true;
            } else {
                elContainerExplore4b.innerHTML = "<div style='padding:10px; color:#666;'>No exploration data available.</div>";
                if (elHeaderExploreOther) elHeaderExploreOther.textContent = otherLabel;
            }

            if (hasContent) {
                 elSectionTraceExplore.classList.remove('hidden');
            } else {
                 elSectionTraceExplore.classList.add('hidden');
            }
        }

        // Flash success?
        console.log("Trace loaded:", data);
    } catch (e) {
        alert("Failed to load trace details: " + e.message);
    } finally {
        hideLoading();
    }
}

// Init Trace Listeners
if (elTraceDataset) {
    elTraceDataset.addEventListener('change', (e) => {
        const ds = e.target.value;
        if (ds) {
            fetchTraces(ds);
        } else {
            elTraceFile.innerHTML = '<option value="">-- Select Dataset First --</option>';
            elTraceFile.disabled = true;
        }
    });
}

if (elTraceFile) {
    elTraceFile.addEventListener('change', (e) => {
        const t = e.target.value;
        const ds = elTraceDataset.value;
        if (t && ds) {
            loadTraceDetails(ds, t);
        }
    });
}
// ---------------------------

function generateLayerPresets() {
   // Helper update
}

function setLayerPreset(mode) {
    const N = appState.n_layers || 28;
    let layers = [];
    if (mode === 'all') {
        layers.push(-1);
        for(let i=0; i<N; i++) layers.push(i);
    } else {
        // Default: 5 parts
        // 0, N/4, 2N/4, 3N/4, N-1
        const steps = 5;
        const stepSize = (N - 1) / steps;
        const set = new Set();
        set.add(-1); // Always include Embedding as requested
        for(let i=0; i<=steps; i++) {
             set.add(Math.round(i * stepSize));
        }
        layers = Array.from(set).sort((a,b)=>a-b);
    }
    elLayersList.value = layers.join(', ');
}

// FORMAT HELPER
function formatTokenForDisplay(tokenStr, escapeMode='visual') {
    if (!tokenStr) return tokenStr;
    
    // 1. Escaping for Data Attributes or Generic Log usage
    if (escapeMode === 'data') {
         return tokenStr.replace(/\\/g, '\\\\')
            .replace(/'/g, "\\'")
            .replace(/\n/g, '\\n')
            .replace(/\r/g, '\\r')
            .replace(/"/g, '&quot;');
    }
    
    // 2. Visual Display (Visible \n)
    if (escapeMode === 'visual') {
        if (tokenStr === '\n') return '\\n';
        if (tokenStr === '\n\n') return '\\n\\n';
        if (tokenStr === '\r\n') return '\\r\\n';
        
        // Check for mixed content
        if (tokenStr.trim() === '' && tokenStr.length > 0) {
            // It is whitespace
            if (tokenStr === ' ') return '␣'; // Optional
        }
        
        // If containing newlines mixed with text, escape the newlines
        if (tokenStr.includes('\n')) return tokenStr.replace(/\n/g, '\\n');
        
        return tokenStr;
    }
    
    return tokenStr;
}

// Initialize Preset on Load Model success (or when appState.n_layers is known)
document.addEventListener('DOMContentLoaded', () => {
    setLayerPreset('default');
});


function renderExploreTable(container, data) {
    if (!container) return;
    
    if (!data || data.length === 0) {
        container.innerHTML = '<div style="padding:10px; color: #888;">No exploration data available.</div>';
        return;
    }
    
    let html = `
    <table style="width: 100%; border-collapse: collapse; font-size: 0.85em;">
        <thead>
            <tr style="background: #f8f8f8; text-align: left; position: sticky; top: 0; z-index: 10;">
                <th style="padding: 6px; border-bottom: 2px solid #ddd;">Rank</th>
                <th style="padding: 6px; border-bottom: 2px solid #ddd;">Token</th>
                <th style="padding: 6px; border-bottom: 2px solid #ddd;">Logit</th>
                <th style="padding: 6px; border-bottom: 2px solid #ddd;">Res</th>
                <th style="padding: 6px; border-bottom: 2px solid #ddd;">Completion Start</th>
            </tr>
        </thead>
        <tbody>
    `;
    
    data.forEach(item => {
        const isCorrect = item.eval_result === true;
        const correctColor = isCorrect ? '#2e7d32' : '#c62828';
        const correctBg = isCorrect ? '#e8f5e9' : '#ffebee';
        const correctIcon = isCorrect ? 'OK' : 'Fail';
        
        let tokenStr = item.token_str || "";
        // Use the existing formatTokenForDisplay helper if available, or simple replacement
        let displayToken = tokenStr;
        if (typeof formatTokenForDisplay === 'function') {
            displayToken = formatTokenForDisplay(tokenStr);
        } else {
             displayToken = tokenStr.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
        }

        // Truncate completion
        let rawCompletion = item.completion || "";
        let displayCompletion = rawCompletion;
        
        const maxLen = 60;
        if (displayCompletion.length > maxLen) {
            displayCompletion = displayCompletion.substring(0, maxLen) + "...";
        }
        
        // Escape HTML
        const escapeHtml = (text) => {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        };

        const fullCompletionEscaped = escapeHtml(rawCompletion);
        const displayCompletionEscaped = escapeHtml(displayCompletion);

        html += `
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 4px 6px;">${item.rank}</td>
            <td style="padding: 4px 6px; font-family: monospace; background: #fafafa;">${displayToken}</td>
            <td style="padding: 4px 6px;">${(item.logits || 0).toFixed(2)}</td>
            <td style="padding: 4px 6px;">
                <span style="font-size: 0.8em; padding: 2px 4px; border-radius: 4px; background: ${correctBg}; color: ${correctColor}; font-weight: bold;">
                    ${correctIcon}
                </span>
            </td>
            <td style="padding: 4px 6px; color: #555; font-size: 0.9em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;" title="${fullCompletionEscaped}">${displayCompletionEscaped}</td>
        </tr>
        `;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

/* MODEL SELECTION LOGIC */
async function initModelSelectors() {
    if(!elModelSeries || !elModelSelect) return;

    // Series change handler
    elModelSeries.addEventListener('change', async () => {
        const series = elModelSeries.value;
        elModelSelect.innerHTML = "<option>Loading...</option>";
        elModelSelect.disabled = true;
        
        try {
            const res = await fetch(`${API_BASE}/list_hf_models?series=${series}`);
            const data = await res.json();
            
            elModelSelect.innerHTML = "";
            if(data.models && data.models.length > 0) {
                 // Sort models alphabetically
                 data.models.sort();
                 data.models.forEach(modelName => {
                    const opt = document.createElement("option");
                    opt.value = modelName;
                    opt.textContent = modelName;
                    elModelSelect.appendChild(opt);
                });
                // Select first
                elModelSelect.disabled = false;
                elModelSelect.dispatchEvent(new Event('change'));
            } else {
                 elModelSelect.innerHTML = "<option>No models found</option>";
            }
        } catch(e) {
            console.error("Failed to list models", e);
            elModelSelect.innerHTML = "<option>Error loading list</option>";
        }
    });

    // Model change handler
    elModelSelect.addEventListener('change', async () => {
        const modelName = elModelSelect.value;
        if(!modelName || modelName.includes("Loading")) return;
        
        // Update Path Input
        elModelPath.value = modelName;

        // Update Revisions
        elModelRevision.innerHTML = '<option>Loading...</option>';
        elModelRevision.disabled = true;
        
        try {
            const res = await fetch(`${API_BASE}/list_model_revisions?model_id=${encodeURIComponent(modelName)}`);
            const data = await res.json();
            
            elModelRevision.innerHTML = '<option value="">Latest (Default)</option>';
            
            const allRevs = [...(data.branches || []), ...(data.tags || [])];
            
            if(allRevs.length > 0) {
                // Sort revisions alphabetically
                allRevs.sort();
                allRevs.forEach(rev => {
                    if (rev === 'main') return; // Skip main as it is usually default
                    const opt = document.createElement("option");
                    opt.value = rev;
                    opt.textContent = rev;
                    elModelRevision.appendChild(opt);
                });
            }
            elModelRevision.disabled = false;
        } catch (e) {
            console.error("Failed to list revisions", e);
            elModelRevision.innerHTML = '<option value="">Latest (Default)</option>';
            elModelRevision.disabled = false;
        }
    });

    // Trigger initial population
    elModelSeries.dispatchEvent(new Event('change'));
}
console.log("Initializing model selectors...");

function saveCircuitGraph() {
    if (!elCanvas) return;
    
    // Create a temporary link
    const link = document.createElement('a');
    
    // Generate filename with timestamp
    const date = new Date();
    const timestamp = date.toISOString().replace(/[:.]/g, '-');
    link.download = `circuit_graph_${timestamp}.png`;
    
    // Convert canvas to blob/dataURL
    // High quality PNG
    const dataUrl = elCanvas.toDataURL('image/png', 1.0);
    
    link.href = dataUrl;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function saveCircuitGraphPDF() {
    if (!elCanvas) return;
    
    try {
        const { jsPDF } = window.jspdf;
        
        // Create temp canvas to flatten background
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = elCanvas.width;
        tempCanvas.height = elCanvas.height;
        const tCtx = tempCanvas.getContext('2d');
        
        // Fill white
        tCtx.fillStyle = '#ffffff';
        tCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        // Draw original
        tCtx.drawImage(elCanvas, 0, 0);
        
        // Calculate PDF size
        // Orientation based on aspect ratio
        const orientation = (elCanvas.width > elCanvas.height) ? 'l' : 'p';
        
        const doc = new jsPDF({
            orientation: orientation,
            unit: 'px',
            format: [elCanvas.width, elCanvas.height] // Custom size matching canvas
        });
        
        const imgData = tempCanvas.toDataURL('image/jpeg', 1.0);
        
        doc.addImage(imgData, 'JPEG', 0, 0, elCanvas.width, elCanvas.height);
        
        const date = new Date();
        const timestamp = date.toISOString().replace(/[:.]/g, '-');
        doc.save(`circuit_graph_${timestamp}.pdf`);
    } catch (e) {
        console.error("PDF generation failed:", e);
        alert("PDF generation failed. Ensure jsPDF is loaded.");
    }
}

async function saveAttributionMapPNG() {
    if (!elInputAttributionDisplay) return;
    try {
        const canvas = await html2canvas(elInputAttributionDisplay, {
            backgroundColor: '#ffffff'
        });
        
        const link = document.createElement('a');
        const date = new Date();
        const timestamp = date.toISOString().replace(/[:.]/g, '-');
        link.download = `attribution_map_${timestamp}.png`;
        link.href = canvas.toDataURL('image/png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (e) {
        console.error("PNG save failed", e);
        alert("Failed to save PNG");
    }
}

async function saveAttributionMapPDF() {
    if (!elInputAttributionDisplay) return;
    try {
        const { jsPDF } = window.jspdf;
        const canvas = await html2canvas(elInputAttributionDisplay, {
             backgroundColor: '#ffffff',
             scale: 2 // Better quality
        });
        
        const imgData = canvas.toDataURL('image/jpeg', 1.0);
        
        // Calculate PDF dims to match canvas (scaled back to points if needed, or just px)
        // Unit 'px' in jsPDF usually corresponds to 1/96 inch, same as canvas pixel?
        // Let's match the canvas pixel dimensions exactly for custom format.
        
        const imgWidth = canvas.width; 
        const imgHeight = canvas.height;
        
        const orientation = (imgWidth > imgHeight) ? 'l' : 'p';
        
        const doc = new jsPDF({
            orientation: orientation,
            unit: 'px',
            format: [imgWidth, imgHeight]
        });
        
        doc.addImage(imgData, 'JPEG', 0, 0, imgWidth, imgHeight);
        
        const date = new Date();
        const timestamp = date.toISOString().replace(/[:.]/g, '-');
        doc.save(`attribution_map_${timestamp}.pdf`);
    } catch (e) {
        console.error("PDF save failed", e);
        alert("Failed to save PDF. Ensure html2canvas/jspdf loaded.");
    }
}


initModelSelectors();
