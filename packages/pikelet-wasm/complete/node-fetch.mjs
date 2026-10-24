// A fetch-shaped wrapper over node:http/node:https, for Node callers of the
// range source (and every other network read the pikelet CLI makes). The
// Node entry of 'pikelet-wasm/complete' makes it httpRangeSource's default.
//
// It exists to keep pack bytes away from the runtime's response decoder.
// Node's built-in fetch decompresses any body a server labels with a
// Content-Encoding; https.request does no automatic decompression, so the
// decoder never runs. On top of that it sends Accept-Encoding: identity and
// fails closed on any other Content-Encoding: a pack is content-addressed
// bytes, and a compressed range response cannot be sliced anyway. The one
// opt-in exception is options.inflateGzip, for HTML a crawler reads from a
// server that compresses whatever it was asked: gzip and deflate only,
// inflated by node:zlib, never Brotli or zstd.
//
// Only what the callers use is implemented: GET/HEAD without a request
// body, a headers object (or Headers), an AbortSignal, and the three
// redirect modes -- 'follow' for plain reads, 'manual' for the range
// source's HEAD walk (which reads Location itself), and 'error' for reads
// under a redirectGuard. The result is a real Response, so callers keep
// using status/ok/headers/body/arrayBuffer/text unchanged; an error on the
// response stream rejects those rather than hanging.

import http from 'node:http';
import https from 'node:https';
import zlib from 'node:zlib';
import { Readable } from 'node:stream';

const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const MAX_REDIRECTS = 20;
// Shared keep-alive agents bounded at the range source's default
// parallelism: connections are reused across a query's reads, and a burst
// opens at most 32 sockets per host instead of one per read (GitHub's
// release CDN answers unbounded bursts with 429s).
const MAX_SOCKETS = 32;
const agents = {
  'http:': new http.Agent({ keepAlive: true, maxSockets: MAX_SOCKETS }),
  'https:': new https.Agent({ keepAlive: true, maxSockets: MAX_SOCKETS }),
};

function requestOnce(url, { method, headers, signal }) {
  const transport = url.protocol === 'https:' ? https : url.protocol === 'http:' ? http : null;
  if (!transport) return Promise.reject(new TypeError(`unsupported protocol ${url.protocol} in ${url.href}`));
  return new Promise((resolve, reject) => {
    const req = transport.request(url, { method, headers, signal, agent: agents[url.protocol] }, resolve);
    req.on('error', reject);
    req.end();
  });
}

// Content-Encoding is a list, applied in order; identity entries are no-ops.
export function contentCodings(value) {
  return String(value || '').split(',').map((c) => c.trim().toLowerCase()).filter((c) => c && c !== 'identity');
}

function toResponse(res, method, body = res) {
  const headers = new Headers();
  for (const [name, value] of Object.entries(res.headers)) {
    if (name === 'set-cookie') continue;
    headers.set(name, Array.isArray(value) ? value.join(', ') : String(value));
  }
  if (body !== res) {
    // Inflated: the encoded length and coding no longer describe the body.
    headers.delete('content-encoding');
    headers.delete('content-length');
  }
  // Null-body statuses: new Response() throws if handed a stream for them.
  const bodyless = method === 'HEAD' || [204, 205, 304].includes(res.statusCode);
  if (bodyless) res.resume();
  return new Response(bodyless ? null : Readable.toWeb(body), {
    status: res.statusCode,
    statusText: res.statusMessage || '',
    headers,
  });
}

export async function nodeFetch(input, init = {}) {
  const method = String(init.method || 'GET').toUpperCase();
  if (method !== 'GET' && method !== 'HEAD') throw new TypeError(`nodeFetch supports GET and HEAD, not ${method}`);
  const redirect = init.redirect || 'follow';
  const headers = Object.fromEntries(new Headers(init.headers || {}));
  headers['accept-encoding'] = 'identity';
  headers['user-agent'] ??= 'pikelet';
  let url = new URL(String(input));
  for (let hop = 0; ; hop++) {
    const res = await requestOnce(url, { method, headers, signal: init.signal });
    const location = res.headers.location;
    if (REDIRECT_STATUSES.has(res.statusCode) && location) {
      res.resume();
      if (redirect === 'manual') return toResponse(res, 'HEAD');
      if (redirect === 'error') throw new TypeError(`redirect refused (redirect: 'error'): ${url.href} -> ${location}`);
      if (hop >= MAX_REDIRECTS) throw new TypeError(`too many redirects from ${String(input)}`);
      const next = new URL(location, url);
      if (url.protocol === 'https:' && next.protocol !== 'https:') {
        throw new TypeError(`refused redirect from https to ${next.protocol.slice(0, -1)}: ${url.href} -> ${next.href}`);
      }
      // Request headers (Range, If-Range) ride along unchanged: a signed-URL
      // hop serves the same range the front door was asked for.
      url = next;
      continue;
    }
    const codings = contentCodings(res.headers['content-encoding']);
    if (codings.length === 0) return toResponse(res, method);
    if (init.inflateGzip && codings.length === 1 && ['gzip', 'x-gzip', 'deflate'].includes(codings[0])) {
      if (method === 'HEAD') return toResponse(res, method);
      const inflate = codings[0] === 'deflate' ? zlib.createInflate() : zlib.createGunzip();
      res.on('error', (err) => inflate.destroy(err));
      return toResponse(res, method, res.pipe(inflate));
    }
    res.destroy();
    throw new Error(`${url.href} answered with Content-Encoding: ${res.headers['content-encoding']} after Accept-Encoding: identity; `
      + 'refusing encoded bytes (the server must serve this file unencoded)');
  }
}
