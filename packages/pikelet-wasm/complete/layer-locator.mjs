// Parent resolution — LAYERED_PROFILE.md 5.1.1.
//
// Identity verification (3.2) prevents substitution of bytes. It does NOT
// prevent a hostile manifest from directing a reader's fetches: a layer's
// `parent.locator` is publisher-supplied, and the publisher is the adversary
// for a distributed artifact. So a locator is confined before any fetch, and
// every redirect hop is confined again with the same rule, never a weaker one.
//
// The governing principle of 5.1.1, which shapes every rule below:
//
//   "Normalization defects live in the disagreements between a validator and
//    the server that finally interprets the path, so the reader refuses every
//    form on which the two could disagree rather than picking one reading."
//
// That is why this module rejects rather than normalizes: an encoded separator
// (%2f), an encoded or literal backslash, an encoded dot segment, or a doubled
// slash all fail outright even though a normalizer could pick a reading for
// each. Picking a reading is how these bugs happen.

/** Forms on which a validator and a server could disagree. Checked on the raw
 *  string, before any decoding, because decoding is itself a reading. */
const FORBIDDEN_RAW = [
    { re: /%2f/i, why: 'an encoded path separator (%2f)' },
    { re: /%5c/i, why: 'an encoded backslash (%5c)' },
    { re: /\\/, why: 'a literal backslash' },
    { re: /%2e/i, why: 'an encoded dot segment (%2e)' },
    { re: /\/\//, why: 'a doubled slash' },
    { re: /%00/i, why: 'an encoded NUL' },
    { re: /[\u0000-\u001f\u007f]/, why: 'a control character' },
];

/**
 * Validate a `layer.parent.locator` as a relative reference, textually, before
 * anything is fetched or resolved. 5.1.1: "A reader MUST reject a locator that
 * violates this, at the manifest, before any fetch."
 */
export function validateLocatorShape(locator) {
    if (typeof locator !== 'string' || locator.length === 0) {
        throw new Error('locator must be a non-empty string');
    }
    if (locator.length > 1024) {
        throw new Error('locator is implausibly long');
    }
    // Shape first, so the refusal names the actual violation: an absolute URL
    // also trips the doubled-slash rule, and "contains a doubled slash" would
    // be a confusing reason to reject "https://evil.example/x".
    // MUST be a relative reference: no scheme, no authority.
    if (/^[a-z][a-z0-9+.-]*:/i.test(locator)) {
        throw new Error('locator must be a relative reference with no scheme');
    }
    if (locator.startsWith('//')) {
        throw new Error('locator must have no authority');
    }
    for (const { re, why } of FORBIDDEN_RAW) {
        if (re.test(locator)) throw new Error(`locator contains ${why}; refused rather than normalized`);
    }
    // An absolute path escapes the child's directory by construction.
    if (locator.startsWith('/')) {
        throw new Error('locator must be relative, not root-absolute');
    }
    // Dot segments in any form: a locator names a sibling, never a traversal.
    const segments = locator.split('/');
    for (const seg of segments) {
        if (seg === '..' || seg === '.') {
            throw new Error(`locator contains the dot segment ${JSON.stringify(seg)}`);
        }
    }
    if (locator.endsWith('/')) {
        throw new Error('locator must name a file, not a directory');
    }
    if (locator.includes('?') || locator.includes('#')) {
        throw new Error('locator must carry no query or fragment');
    }
    return true;
}

/** The directory part of a URL path, with its trailing slash. */
function directoryOf(pathname) {
    const at = pathname.lastIndexOf('/');
    return at === -1 ? '/' : pathname.slice(0, at + 1);
}

/**
 * Resolve a locator against the child's own URL and confirm the result stays
 * within the child's origin and directory.
 *
 * @param {string} locator  a locator that already passed validateLocatorShape
 * @param {string} childUrl the URL the CHILD's bytes were fetched from
 * @returns {string} the absolute URL to fetch the parent from
 */
export function resolveLocatorUrl(locator, childUrl) {
    validateLocatorShape(locator);
    let child;
    try { child = new URL(childUrl); } catch (err) {
        throw new Error(`child location ${JSON.stringify(childUrl)} is not a URL`, { cause: err });
    }
    if (child.protocol !== 'http:' && child.protocol !== 'https:') {
        throw new Error(`parent resolution supports http and https only, not ${child.protocol}`);
    }
    const resolved = new URL(locator, child);
    // Canonical comparison: URL already lowercases scheme and host and removes
    // dot segments. The shape check above refused the forms where that
    // normalization could disagree with a server's reading.
    assertConfined(resolved, child, 'locator');
    return resolved.href;
}

/**
 * The confinement rule, applied to a locator's result and to EVERY redirect
 * hop. 5.1.1: "Redirects encountered while fetching a parent are subject to the
 * same confinement as the locator, not a weaker one ... and the check is
 * repeated at every hop before the redirected request is issued."
 *
 * A same-origin redirect to a path outside the directory fails exactly as a
 * cross-origin one does.
 */
export function assertConfined(target, child, what = 'redirect target') {
    const t = target instanceof URL ? target : new URL(target);
    const c = child instanceof URL ? child : new URL(child);
    if (t.protocol !== c.protocol) {
        throw new Error(`${what} changes scheme from ${c.protocol} to ${t.protocol}`);
    }
    if (t.host !== c.host) {
        throw new Error(`${what} leaves the child's origin: ${t.host} is not ${c.host}`);
    }
    if (t.username || t.password) {
        throw new Error(`${what} carries credentials`);
    }
    for (const { re, why } of FORBIDDEN_RAW) {
        // Re-checked on the resolved href: a redirect's Location header is a
        // fresh string from the network, not the validated locator.
        if (re.test(t.pathname)) throw new Error(`${what} contains ${why}`);
    }
    const dir = directoryOf(c.pathname);
    if (!t.pathname.startsWith(dir)) {
        throw new Error(`${what} path ${t.pathname} is not under the child's directory ${dir}`);
    }
    if (t.pathname.length === dir.length) {
        throw new Error(`${what} names the directory itself, not a file`);
    }
    return true;
}

/**
 * The file-source analogue. 5.1.1 requires the check to run on the REAL path,
 * "with every symbolic link followed", so a symlink cannot escape the
 * directory. The caller supplies a realpath function because this module stays
 * environment-neutral (no node:fs import in a file the browser bundle pulls in).
 *
 * @param {string} locator
 * @param {string} childPath  the child artifact's path
 * @param {{realpath: (p: string) => Promise<string>, dirname: (p: string) => string,
 *          join: (...p: string[]) => string, sep?: string}} fsops
 */
export async function resolveLocatorPath(locator, childPath, fsops) {
    validateLocatorShape(locator);
    if (!fsops || typeof fsops.realpath !== 'function') {
        throw new Error('resolveLocatorPath needs a realpath implementation');
    }
    const childDir = fsops.dirname(childPath);
    // The child's own directory, resolved through links first: comparing against
    // an unresolved directory would let a link in the PARENT position pass.
    const realChildDir = await fsops.realpath(childDir);
    const candidate = fsops.join(childDir, locator);
    let realCandidate;
    try {
        realCandidate = await fsops.realpath(candidate);
    } catch (err) {
        // A locator naming something that does not exist is an availability
        // failure (3.2), reported as such rather than as a security refusal.
        throw new Error(`parent named by locator ${JSON.stringify(locator)} cannot be located`, { cause: err });
    }
    const sep = fsops.sep || '/';
    const prefix = realChildDir.endsWith(sep) ? realChildDir : realChildDir + sep;
    if (!realCandidate.startsWith(prefix)) {
        throw new Error(`locator resolves to ${realCandidate}, outside the child's real directory ${realChildDir}`);
    }
    return realCandidate;
}

/**
 * Resolution order of 5.1.1: lineage listing, then locator, then host resolver.
 * Whatever located the bytes, the identity check of 3.2 is what admits them —
 * so this returns a *candidate location only*, never a verdict on the bytes.
 *
 * A lineage listing is operator-supplied (the operator chose the shelf) and MAY
 * name any origin the operator trusts; a manifest locator is publisher-supplied
 * and gets the confinement above. That asymmetry is deliberate.
 *
 * @param {{identity: string, locator: string|null}} want
 * @param {{lineage?: Map<string,string>|null, childLocation?: string|null,
 *          kind?: 'url'|'file', resolveParents?: boolean,
 *          hostResolver?: ((identity: string) => string|null)|null,
 *          fsops?: object|null}} opts
 */
export async function resolveParentLocation(want, opts = {}) {
    const {
        lineage = null, childLocation = null, kind = 'url',
        resolveParents = true, hostResolver = null, fsops = null,
    } = opts;
    if (!want || typeof want.identity !== 'string') throw new Error('resolveParentLocation needs the wanted identity');

    // 1. Lineage listing: operator-supplied, trusted for location.
    if (lineage && lineage.has(want.identity)) {
        return { location: lineage.get(want.identity), via: 'lineage' };
    }
    // 2. Manifest locator: publisher-supplied, confined.
    if (want.locator) {
        if (!resolveParents) {
            throw new Error('locator resolution is disabled (resolveParents: false) and no lineage listing names this parent');
        }
        if (!childLocation) {
            throw new Error('a locator needs the child\'s own location to resolve against');
        }
        const location = kind === 'file'
            ? await resolveLocatorPath(want.locator, childLocation, fsops)
            : resolveLocatorUrl(want.locator, childLocation);
        return { location, via: 'locator' };
    }
    // 3. Host resolver.
    if (typeof hostResolver === 'function') {
        const location = hostResolver(want.identity);
        if (location) return { location, via: 'host' };
    }
    // 3.2: a reader that cannot locate an ancestor MUST fail the mount
    // explicitly — never serve the layers it did reach as a partial corpus.
    throw new Error(`cannot locate chain member ${want.identity.slice(0, 12)}…: no lineage entry, no locator and no host resolver`);
}
