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
        setStatus(
            `Mounted — ${(fileBytes / 1024 ** 2).toFixed(1)} MB, ${info.records.toLocaleString()} records, `
            + `identity ${info.identity.slice(0, 12)}…, hash verified: ${info.residentVerified}`,
            'ok',
        );
        els.queryInput.disabled = false;
        els.queryInput.focus();
    } catch (err) {
        setStatus(`Failed to mount: ${err?.message || err}`, 'error');
    } finally {
        els.mountBtn.disabled = false;
    }
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

els.packUrl.value = DEFAULT_PACK_URL;
mount(DEFAULT_PACK_URL);
