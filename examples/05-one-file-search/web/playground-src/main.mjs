// Pikelet playground: query a real .pikelet knowledge pack straight from a
// static host, entirely in the browser. Same reader, encoder, sketch scan,
// and calibration Node runs — no server-side search. Every request the
// page makes is visible in the live log; there is no /search, /embed,
// /query, or /api endpoint anywhere behind this URL.
//
// The default pack is served same-origin (relative path) rather than from
// a GitHub release asset: GitHub's release-asset host (and the Azure blob
// storage behind it) sends no Access-Control-Allow-Origin header, so a
// cross-origin browser fetch for it is blocked outright — a real CORS
// limitation of that host, not something this reader can route around.
// Range-reading an arbitrary .pikelet URL still works from here as long as
// that host sends CORS headers permitting it; type any such URL into the
// mount field to try one.

import { openPikeletFile } from '../../pikelet-file-reader.mjs';
import { httpRangeSource } from '../../sources.mjs';

const DEFAULT_PACK_URL =
  'https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev/wikipedia.pikelet';

const els = {
    packUrl: document.getElementById('pack-url'),
    mountBtn: document.getElementById('mount-btn'),
    status: document.getElementById('status'),
    queryForm: document.getElementById('query-form'),
    queryInput: document.getElementById('query-input'),
    verdict: document.getElementById('verdict'),
    results: document.getElementById('results'),
    log: document.getElementById('request-log'),
    reqCount: document.getElementById('req-count'),
    byteCount: document.getElementById('byte-count'),
    pctCount: document.getElementById('pct-count'),
    identityCard: document.getElementById('identity-card'),
    idFilename: document.getElementById('id-filename'),
    idSize: document.getElementById('id-size'),
    idRecords: document.getElementById('id-records'),
    idEncoder: document.getElementById('id-encoder'),
    idHash: document.getElementById('id-hash'),
    transferBanner: document.getElementById('transfer-banner'),
    transferHeadline: document.getElementById('transfer-headline'),
    suggestedQueries: document.getElementById('suggested-queries'),
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

let search = null;
let source = null;
let fileBytes = 0;
let logRows = [];
const MAX_LOG_ROWS = 60;

function renderLog() {
    els.log.innerHTML = logRows
        .slice(-MAX_LOG_ROWS)
        .map((row) => `<div class="log-row"><span class="log-verb">GET</span><span class="log-range">${row}</span></div>`)
        .join('');
    els.log.scrollTop = els.log.scrollHeight;
}

// Range-read listeners other than the search tab's own counters (currently
// just the ablation tab) register here — wireFetchLogging notifies all of
// them on every request, since fetch is patched exactly once, globally.
const rangeReadListeners = [];

function wireFetchLogging() {
    const origFetch = globalThis.fetch;
    globalThis.fetch = async (url, opts) => {
        const range = opts?.headers?.Range || opts?.headers?.range;
        const response = await origFetch(url, opts);
        if (range) {
            const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range) || [];
            if (a !== undefined) {
                const len = Number(b) - Number(a) + 1;
                stats.requests++;
                stats.bytes += len;
                logRows.push(`bytes=${a}-${b}  <span class="log-len">(${formatBytes(len)})</span>`);
                renderLog();
                updateCounters();
                for (const listener of rangeReadListeners) listener(url, a, b, len);
            }
        }
        return response;
    };
}

const stats = { requests: 0, bytes: 0 };

function updateCounters() {
    els.reqCount.textContent = stats.requests;
    els.byteCount.textContent = formatBytes(stats.bytes);
    els.pctCount.textContent = fileBytes
        ? `${((stats.bytes / fileBytes) * 100).toFixed(3)}%`
        : '—';
    if (fileBytes && stats.bytes > 0) {
        const pct = (stats.bytes / fileBytes) * 100;
        els.transferHeadline.innerHTML = `<span class="dim">${formatBytes(fileBytes)} artifact ·</span> `
            + `${formatBytes(stats.bytes)} fetched <span class="dim">·</span> ${pct < 0.1 ? pct.toFixed(3) : pct.toFixed(1)}% transferred`;
        els.transferBanner.hidden = false;
    } else {
        els.transferBanner.hidden = true;
    }
}

function setStatus(text, cls) {
    els.status.textContent = text;
    els.status.className = `status${cls ? ` ${cls}` : ''}`;
}

async function mount(url) {
    els.mountBtn.disabled = true;
    els.queryInput.disabled = true;
    setStatus('Resolving…', 'busy');
    logRows = [];
    stats.requests = 0;
    stats.bytes = 0;
    renderLog();
    updateCounters();

    try {
        source = httpRangeSource(url);
        search = await openPikeletFile(source, {
            sketchScanner: async (sketch) => {
                const { default: Pikelet } = await import('../../../../pikelet.web.mjs');
                return Pikelet.createSketchScanner(sketch, { maxRerank: 4096 });
            },
        });
        const info = search.info();
        fileBytes = info.fileBytes;
        updateCounters();
        setStatus(`Mounted — hash verified: ${info.residentVerified}`, 'ok');
        renderIdentity(url, info);
        els.queryInput.disabled = false;
        els.queryInput.focus();
    } catch (err) {
        setStatus(`Failed to mount: ${err?.message || err}`, 'error');
        els.identityCard.hidden = true;
    } finally {
        els.mountBtn.disabled = false;
    }
}

function renderIdentity(url, info) {
    let filename = url;
    try { filename = new URL(url).pathname.split('/').pop() || url; } catch { /* relative path — keep as-is */ }
    els.idFilename.textContent = filename;
    els.idSize.textContent = `${(info.fileBytes / 1024 ** 2).toFixed(1)} MB`;
    els.idRecords.textContent = info.records.toLocaleString();
    els.idEncoder.textContent = info.encoder?.kind
        ? `${info.encoder.kind}${info.encoderVerified === false ? ' (unverified)' : ''}`
        : 'embedded';
    els.idHash.textContent = info.identity;
    els.identityCard.hidden = false;
}

let running = false;
async function runQuery() {
    const text = els.queryInput.value.trim();
    if (!text || running || !search) return;
    running = true;
    els.verdict.textContent = 'Searching…';
    els.verdict.className = 'verdict';
    els.results.innerHTML = '';
    const before = { ...stats };
    const t0 = performance.now();
    try {
        const out = await search.query(text, { k: 5 });
        const ms = performance.now() - t0;
        const usedReq = stats.requests - before.requests;
        const usedBytes = stats.bytes - before.bytes;
        const badge = `<span class="badge ${out.matchQuality}">${out.matchQuality}</span>`;
        const confidence = out.confidence !== undefined ? ` · confidence ${out.confidence.toFixed(3)}` : '';
        els.verdict.innerHTML = `${badge}${confidence} · ${ms.toFixed(0)} ms · `
            + `${usedReq} range reads · ${formatBytes(usedBytes)} fetched`;
        els.results.innerHTML = out.results.length === 0
            ? '<p class="empty">No results — the file knows this corpus cannot answer that.</p>'
            : out.results.map((r) => `
                <article class="hit">
                  <h3>${escapeHtml(r.title)}</h3>
                  <p class="meta">${escapeHtml(r.sourcePath || r.url || '')}${r.anchor ? `#${escapeHtml(r.anchor)}` : ''} · distance ${r.distance?.toFixed(3)}</p>
                  <p class="excerpt">${escapeHtml((r.preview || r.text || '').slice(0, 240))}…</p>
                </article>`).join('');
    } catch (err) {
        els.verdict.textContent = String(err?.message || err);
        els.verdict.className = 'verdict error';
    } finally {
        running = false;
    }
}

wireFetchLogging();

els.mountBtn.addEventListener('click', () => mount(els.packUrl.value.trim() || DEFAULT_PACK_URL));
els.queryForm.addEventListener('submit', (event) => {
    event.preventDefault();
    runQuery();
});
els.suggestedQueries.addEventListener('click', (event) => {
    const chip = event.target.closest('.chip');
    if (!chip || els.queryInput.disabled) return;
    els.queryInput.value = chip.dataset.q;
    runQuery();
});

els.packUrl.value = DEFAULT_PACK_URL;
mount(DEFAULT_PACK_URL);

// --- Ablation demo tab: same question against a full pack, a pack missing
// one record, and (separately) a pack with one record edited to contradict
// the rest. Fictional Station Veyra research corpus — see
// examples/05-one-file-search/web/veyra-corpus/README.md for how it's
// generated and compiled. Loaded lazily (first tab open), not on page
// load, since these packs are ~25 MB each and most visitors to the search
// tab will never open this one.
const VEYRA_BASE = 'https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev';

// Which single fact gets removed is picked from this list, not fixed to
// one example — a single hardcoded before/after reads as cherry-picked.
// Each entry here was individually compiled (one fact removed from the
// 94-fact corpus, corpus-<fact>/) and verified end to end against a real
// build before being listed, against two bars: the FULL pack must answer
// with a strong verdict at k=1 (otherwise there's no contrast to show),
// and the ABLATED pack must then abstain (or, for the counting question,
// correctly decline to guess a wrong count). More candidate removals were
// tried than are listed here — some failed the first bar (the full pack
// already hedges on a multi-hop question before anything is removed) and
// some failed the second (the retriever finds a plausible-looking but
// wrong passage after the fact is gone — a different fact about the same
// project, or an unrelated passage that happens to share vocabulary with
// a broken reasoning chain). That's a real, known limitation of the
// retrieval/calibration signal, not something to paper over by only ever
// showing the cases that work without saying so.
// Each scenario also carries a "contradicted" variant for the third panel:
// a pack identical to the full one except that this same fact's record was
// edited to state a different, still-plausible value (another clearance
// name, vault letter, or supervisor already used elsewhere in the corpus —
// never a nonsense value). contradictQuestion is tuned per scenario where
// the ablation-panel question's own top-1 retrieval doesn't land on the
// edited record (a ranking quirk, not a contradiction-specific one — see
// the loc-tovash/clr-sarnix note below) so the panel reliably demonstrates
// the point instead of an unrelated near-miss.
const FACT_SCENARIOS = [
    {
        id: 'loc-tovash',
        label: 'Where is Tovash housed?',
        question: 'What chamber is the Tovash project housed in?',
        removedText: '"The Tovash project is housed in Chamber 17."',
        // This one needs all 6 of the corpus's curated removals gone at
        // once (veyra-ablated.pikelet) — removing loc-tovash alone left
        // this corpus too small to pass the calibration quality gate, so
        // no other single-fact pack for it exists.
        ablatedUrl: `${VEYRA_BASE}/veyra-ablated.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-chamber43.pikelet`,
        contradictQuestion: 'What chamber is the Tovash project housed in?',
        contradictNote: 'Chamber 17 → Chamber 43',
    },
    {
        id: 'clr-sarnix',
        label: 'What clearance does Sarnix use?',
        // Not the more obvious phrasing ("What clearance does the Sarnix
        // project use?") — that one ties for top-1 at k=1 against an
        // unrelated "Sarnix Location" record on this pack. This phrasing,
        // which echoes the fact's own second sentence, reliably retrieves
        // the actual clearance record on every pack this scenario uses.
        question: 'What clearance badge do Sarnix project members carry?',
        removedText: '"The Sarnix project uses cobalt clearance."',
        ablatedUrl: `${VEYRA_BASE}/veyra-remove-clr-sarnix.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-contradict-clr-sarnix.pikelet`,
        contradictQuestion: 'What clearance badge do Sarnix project members carry?',
        contradictNote: 'cobalt clearance → ochre clearance',
    },
    {
        id: 'vault-crimson',
        label: 'Which vault does crimson clearance open?',
        question: 'Which vault does crimson clearance permit access to?',
        removedText: '"Crimson clearance permits access to Vault A."',
        ablatedUrl: `${VEYRA_BASE}/veyra-remove-vault-crimson.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-contradict-vault-crimson.pikelet`,
        contradictQuestion: 'Which vault does crimson clearance permit access to?',
        contradictNote: 'Vault A → Vault B',
    },
    {
        id: 'sup-betzel',
        label: 'How many projects does Irena Sol supervise?',
        question: 'How many projects does Irena Sol supervise?',
        removedText: '"Irena Sol supervises the Betzel project." — the count should drop from 3, but the file correctly declines to guess a new number rather than state a wrong one.',
        ablatedUrl: `${VEYRA_BASE}/veyra-remove-sup-betzel.pikelet`,
        contradictUrl: `${VEYRA_BASE}/veyra-contradict-sup-betzel.pikelet`,
        // A direct question, not the counting one above: counting questions
        // don't map onto "confidently states the wrong value" the way a
        // direct fact does, so the contradiction panel asks the direct
        // version of the same underlying record instead.
        contradictQuestion: 'Who supervises the Betzel project?',
        contradictNote: 'Irena Sol → Tomas Vale',
    },
];

const ablationEls = {
    tabBtnSearch: document.getElementById('tab-btn-search'),
    tabBtnAblation: document.getElementById('tab-btn-ablation'),
    tabSearch: document.getElementById('tab-search'),
    tabAblation: document.getElementById('tab-ablation'),
    factPicker: document.getElementById('fact-picker'),
    question: document.getElementById('ablation-question'),
    removed: document.getElementById('ablation-removed'),
    runBtn: document.getElementById('ablation-run-btn'),
    fullBadge: document.getElementById('abl-full-badge'),
    fullAnswer: document.getElementById('abl-full-answer'),
    fullRecord: document.getElementById('abl-full-record'),
    ablatedBadge: document.getElementById('abl-ablated-badge'),
    ablatedAnswer: document.getElementById('abl-ablated-answer'),
    ablatedRecord: document.getElementById('abl-ablated-record'),
    c43Question: document.getElementById('ablation-question-c43'),
    c43Note: document.getElementById('ablation-c43-note'),
    c43RunBtn: document.getElementById('ablation-c43-run-btn'),
    c43Badge: document.getElementById('abl-c43-badge'),
    c43Answer: document.getElementById('abl-c43-answer'),
    c43Record: document.getElementById('abl-c43-record'),
    reqCount: document.getElementById('abl-req-count'),
    byteCount: document.getElementById('abl-byte-count'),
    log: document.getElementById('abl-request-log'),
};

// Own network-activity monitor for the ablation tab: fetch is patched once,
// globally (wireFetchLogging), so this just registers as another listener
// on every range read rather than re-patching fetch itself. Kept separate
// from the search tab's stats/log since the two tabs mount different files
// with different sizes — a shared "% of the file" figure wouldn't mean
// anything once more than one file is in play.
const ablationStats = { requests: 0, bytes: 0 };
let ablationLogRows = [];
const ABL_MAX_LOG_ROWS = 40;
rangeReadListeners.push((url, a, b, len) => {
    ablationStats.requests++;
    ablationStats.bytes += len;
    ablationEls.reqCount.textContent = ablationStats.requests;
    ablationEls.byteCount.textContent = formatBytes(ablationStats.bytes);
    let filename = url;
    try { filename = new URL(url).pathname.split('/').pop() || url; } catch { /* relative path */ }
    ablationLogRows.push(`<span class="log-file">${escapeHtml(filename)}</span> bytes=${a}-${b}  <span class="log-len">(${formatBytes(len)})</span>`);
    ablationEls.log.innerHTML = ablationLogRows
        .slice(-ABL_MAX_LOG_ROWS)
        .map((row) => `<div class="log-row"><span class="log-verb">GET</span><span class="log-range">${row}</span></div>`)
        .join('');
    ablationEls.log.scrollTop = ablationEls.log.scrollHeight;
});

function switchTab(name) {
    const toAblation = name === 'ablation';
    ablationEls.tabBtnSearch.classList.toggle('active', !toAblation);
    ablationEls.tabBtnAblation.classList.toggle('active', toAblation);
    ablationEls.tabBtnSearch.setAttribute('aria-selected', String(!toAblation));
    ablationEls.tabBtnAblation.setAttribute('aria-selected', String(toAblation));
    ablationEls.tabSearch.hidden = toAblation;
    ablationEls.tabAblation.hidden = !toAblation;
    if (toAblation) initAblationDemo();
}

async function runAblationPanel(url, question, badgeEl, answerEl, recordEl) {
    badgeEl.innerHTML = '';
    answerEl.textContent = 'loading…';
    recordEl.innerHTML = '';
    try {
        const source = httpRangeSource(url);
        const packSearch = await openPikeletFile(source);
        // No showAbstained: an abstained verdict here should visibly
        // withhold its results the way a real caller who wants that
        // behavior sees it, not surface the nearest-but-wrong passage
        // next to a "none" badge as if it were an answer.
        const out = await packSearch.query(question, { k: 1 });
        badgeEl.innerHTML = `<span class="badge ${out.matchQuality}">${out.matchQuality}</span> confidence ${out.confidence?.toFixed(3) ?? '—'}`;
        if (out.results.length === 0) {
            answerEl.textContent = 'No results — the file knows this corpus cannot answer that.';
            recordEl.innerHTML = '';
            return;
        }
        const top = out.results[0];
        answerEl.textContent = top.preview || top.text || '';
        recordEl.innerHTML = `record ${escapeHtml(String(top.id))} — ${escapeHtml(top.title || '')}`;
    } catch (err) {
        badgeEl.innerHTML = '<span class="badge none">error</span>';
        answerEl.textContent = String(err?.message || err);
    }
}

// The full pack's answer never changes across scenarios (same question set,
// same complete file) — cached per question text so re-running doesn't
// needlessly re-mount and re-query the 25 MB full pack every time.
const fullPackAnswerCache = new Map();
async function runFullPanel(question) {
    if (!fullPackAnswerCache.has(question)) {
        fullPackAnswerCache.set(question, (async () => {
            const source = httpRangeSource(`${VEYRA_BASE}/veyra.pikelet`);
            const packSearch = await openPikeletFile(source);
            return packSearch.query(question, { k: 1 });
        })());
    }
    ablationEls.fullBadge.innerHTML = '';
    ablationEls.fullAnswer.textContent = 'loading…';
    ablationEls.fullRecord.innerHTML = '';
    try {
        const out = await fullPackAnswerCache.get(question);
        ablationEls.fullBadge.innerHTML = `<span class="badge ${out.matchQuality}">${out.matchQuality}</span> confidence ${out.confidence?.toFixed(3) ?? '—'}`;
        const top = out.results[0];
        ablationEls.fullAnswer.textContent = top ? (top.preview || top.text || '') : 'No results.';
        ablationEls.fullRecord.innerHTML = top ? `record ${escapeHtml(String(top.id))} — ${escapeHtml(top.title || '')}` : '';
    } catch (err) {
        fullPackAnswerCache.delete(question);
        ablationEls.fullBadge.innerHTML = '<span class="badge none">error</span>';
        ablationEls.fullAnswer.textContent = String(err?.message || err);
    }
}

let currentScenario = null;

// Updates the labels for the selected scenario without firing any network
// request — the two "Run query" buttons are what actually trigger reads,
// so switching the picker never mounts a pack the visitor didn't ask for.
function selectFact(scenario) {
    if (currentScenario?.id === scenario.id) return;
    currentScenario = scenario;
    [...ablationEls.factPicker.children].forEach((el) => el.classList.toggle('selected', el.dataset.fact === scenario.id));
    ablationEls.question.textContent = `"${scenario.question}"`;
    ablationEls.removed.innerHTML = `removed record: <code>${escapeHtml(scenario.id)}</code> — ${scenario.removedText}`;
    ablationEls.c43Question.textContent = `"${scenario.contradictQuestion}"`;
    ablationEls.c43Note.innerHTML = `edited record: <code>${escapeHtml(scenario.id)}</code> — ${escapeHtml(scenario.contradictNote)}`;
    for (const [badge, answer, record] of [
        [ablationEls.fullBadge, ablationEls.fullAnswer, ablationEls.fullRecord],
        [ablationEls.ablatedBadge, ablationEls.ablatedAnswer, ablationEls.ablatedRecord],
        [ablationEls.c43Badge, ablationEls.c43Answer, ablationEls.c43Record],
    ]) {
        badge.innerHTML = '';
        record.innerHTML = '';
    }
    ablationEls.fullAnswer.textContent = 'Press "Run query on both files" above.';
    ablationEls.ablatedAnswer.textContent = 'Press "Run query on both files" above.';
    ablationEls.c43Answer.textContent = 'Press "Run query on the contradicted file" above.';
}

async function runRemovalPair() {
    if (!currentScenario) return;
    ablationEls.runBtn.disabled = true;
    try {
        // Sequential, not concurrent: independent fetches resolving in
        // network-dependent order filled panels out of reading order (right
        // side before left) — confusing for a demo whose whole point is a
        // careful side-by-side comparison. Not a hot path, so the ~2x wait
        // is an easy trade for filling in the order a reader expects.
        await runFullPanel(currentScenario.question);
        await runAblationPanel(currentScenario.ablatedUrl, currentScenario.question, ablationEls.ablatedBadge, ablationEls.ablatedAnswer, ablationEls.ablatedRecord);
    } finally {
        ablationEls.runBtn.disabled = false;
    }
}

async function runContradictionPanel() {
    if (!currentScenario) return;
    ablationEls.c43RunBtn.disabled = true;
    try {
        await runAblationPanel(currentScenario.contradictUrl, currentScenario.contradictQuestion, ablationEls.c43Badge, ablationEls.c43Answer, ablationEls.c43Record);
    } finally {
        ablationEls.c43RunBtn.disabled = false;
    }
}

let ablationStarted = false;
function initAblationDemo() {
    if (ablationStarted) return;
    ablationStarted = true;
    ablationEls.factPicker.innerHTML = FACT_SCENARIOS
        .map((s) => `<button type="button" class="chip" data-fact="${escapeHtml(s.id)}">${escapeHtml(s.label)}</button>`)
        .join('');
    ablationEls.factPicker.addEventListener('click', (event) => {
        const chip = event.target.closest('.chip');
        if (!chip) return;
        const scenario = FACT_SCENARIOS.find((s) => s.id === chip.dataset.fact);
        if (scenario) selectFact(scenario);
    });
    ablationEls.runBtn.addEventListener('click', runRemovalPair);
    ablationEls.c43RunBtn.addEventListener('click', runContradictionPanel);
    selectFact(FACT_SCENARIOS[0]);
}

ablationEls.tabBtnSearch.addEventListener('click', () => switchTab('search'));
ablationEls.tabBtnAblation.addEventListener('click', () => switchTab('ablation'));

document.getElementById('link-to-ablation-tab')?.addEventListener('click', (event) => {
    event.preventDefault();
    switchTab('ablation');
});
