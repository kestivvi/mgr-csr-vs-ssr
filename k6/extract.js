// Pure asset-discovery used by k6 setup() and unit tests.
// Parser and fetcher are injected so the same code runs under:
//   - k6: parseHTML from 'k6/html', http.get sync fetcher
//   - Node tests: cheerio adapter + fixture-backed fetcher

const STATIC_FROM_RE = /\bimport\s*(?:[^;'"]*?)?from\s*["']([^"']+)["']/g;
const STATIC_BARE_RE = /\bimport\s*["']([^"']+)["']/g;
const DYNAMIC_IMPORT_RE = /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g;

export function parseEsmImports(text) {
  const out = [];
  // Some HTML adapters return a non-string handle for textContent (k6/html
  // returns a Selection). Coerce defensively so the regex can run.
  if (text != null && typeof text !== 'string') text = String(text);
  if (!text || text.indexOf('import') === -1) return out;
  let m;
  STATIC_FROM_RE.lastIndex = 0;
  while ((m = STATIC_FROM_RE.exec(text)) !== null) out.push(m[1]);
  STATIC_BARE_RE.lastIndex = 0;
  while ((m = STATIC_BARE_RE.exec(text)) !== null) out.push(m[1]);
  DYNAMIC_IMPORT_RE.lastIndex = 0;
  while ((m = DYNAMIC_IMPORT_RE.exec(text)) !== null) out.push(m[1]);
  return out;
}

// URL resolution implemented manually because Goja (the JS runtime k6
// uses) does not expose the WHATWG `URL` constructor. Mirrors the subset
// of behaviour needed: absolute, scheme-relative (//), root-relative (/),
// and dot-relative (./ ../). Query/hash are passed through unchanged.
export function splitUrl(url) {
  const m = /^(https?:)\/\/([^\/]+)(\/[^?#]*|)(\?[^#]*)?(#.*)?$/.exec(url);
  if (!m) return null;
  return { scheme: m[1], host: m[2], path: m[3] || '/', search: m[4] || '', hash: m[5] || '' };
}

export function normalizePath(path) {
  const isAbs = path.startsWith('/');
  const parts = path.split('/');
  const out = [];
  for (const p of parts) {
    if (p === '' || p === '.') continue;
    if (p === '..') { if (out.length) out.pop(); continue; }
    out.push(p);
  }
  let joined = out.join('/');
  if (isAbs) joined = '/' + joined;
  if (path.endsWith('/') && !joined.endsWith('/')) joined += '/';
  return joined || '/';
}

export function resolveUrl(spec, importerUrl) {
  if (!spec) return null;
  if (spec.startsWith('data:') || spec.startsWith('#')) return null;
  if (/^https?:\/\//.test(spec)) return spec;
  const base = splitUrl(importerUrl);
  if (!base) return null;
  if (spec.startsWith('//')) return base.scheme + spec;
  if (spec.startsWith('/')) return base.scheme + '//' + base.host + spec;
  // Relative: drop importer's file segment, join, normalize.
  const dir = base.path.replace(/[^/]*$/, '');
  return base.scheme + '//' + base.host + normalizePath(dir + spec);
}

function originOf(url) {
  const p = splitUrl(url);
  return p ? p.scheme + '//' + p.host : null;
}

export function sameOrigin(url, baseUrl) {
  const a = originOf(url);
  const b = originOf(baseUrl);
  return a !== null && a === b;
}

export function discoverAssets({ html, baseUrl, parserAdapter, fetcher, maxAssets = 500 }) {
  const assets = new Set();
  const doc = parserAdapter.parse(html);

  function add(spec, importerUrl) {
    const resolved = resolveUrl(spec, importerUrl);
    if (resolved) assets.add(resolved);
  }

  // 1) Standard tags
  parserAdapter.each(doc, 'link[href]', (el) => add(parserAdapter.attr(el, 'href'), baseUrl));
  parserAdapter.each(doc, 'script[src]', (el) => add(parserAdapter.attr(el, 'src'), baseUrl));
  parserAdapter.each(doc, 'img[src]', (el) => add(parserAdapter.attr(el, 'src'), baseUrl));

  // 2) Astro Islands
  parserAdapter.each(doc, 'astro-island', (el) => {
    add(parserAdapter.attr(el, 'component-url'), baseUrl);
    add(parserAdapter.attr(el, 'renderer-url'), baseUrl);
  });

  // 3) Inline <script> bodies: scan for ESM imports
  parserAdapter.each(doc, 'script', (el) => {
    if (parserAdapter.attr(el, 'src')) return;
    const body = parserAdapter.text(el);
    for (const spec of parseEsmImports(body)) add(spec, baseUrl);
  });

  // 4) Qwik bundle-graph.json: lazy chunks (q-*.js) appear only as string
  //    literals inside the manifest, never as <script src> or import().
  //    Detect any */bundle-graph*.json URL already in the set, fetch it,
  //    and add every `q-*.js` string under the q:base prefix declared on
  //    <html> (defaults to /build/).
  let qBase = parserAdapter.attrOf(doc, 'html', 'q:base');
  if (!qBase) qBase = '/build/';
  if (!qBase.endsWith('/')) qBase += '/';
  for (const url of [...assets]) {
    if (!/bundle-graph[^/]*\.json($|\?)/.test(url)) continue;
    if (!sameOrigin(url, baseUrl)) continue;
    const res = fetcher(url);
    if (!res || res.status !== 200) continue;
    let parsed;
    try {
      parsed = typeof res.body === 'string' ? JSON.parse(res.body) : res.body;
    } catch {
      continue;
    }
    const list = Array.isArray(parsed) ? parsed : Object.keys(parsed || {});
    for (const item of list) {
      if (typeof item !== 'string') continue;
      if (!/^q-[A-Za-z0-9_-]+\.m?js$/.test(item)) continue;
      add(qBase + item, baseUrl);
    }
  }

  // 5) Recursive same-origin .js follow (SvelteKit + chunked builds)
  const visited = new Set();
  const queue = [...assets].filter((u) => sameOrigin(u, baseUrl) && /\.(m?js)(\?|$)/.test(u));
  while (queue.length && visited.size < maxAssets) {
    const url = queue.shift();
    if (visited.has(url)) continue;
    visited.add(url);
    const res = fetcher(url);
    if (!res || res.status !== 200) continue;
    for (const spec of parseEsmImports(res.body)) {
      const resolved = resolveUrl(spec, url);
      if (!resolved) continue;
      if (!sameOrigin(resolved, baseUrl)) continue;
      if (!assets.has(resolved)) {
        assets.add(resolved);
        if (/\.(m?js)(\?|$)/.test(resolved)) queue.push(resolved);
      }
    }
  }

  return assets;
}

// Cheerio adapter for tests. Mirrors the shape of the k6/html adapter
// (defined in script.js) so discoverAssets works the same under both.
export function cheerioParserAdapter(cheerioMod) {
  return {
    parse: (html) => cheerioMod.load(html),
    each: ($, selector, fn) => {
      $(selector).each((_i, el) => fn(el));
    },
    attr: (el, name) => {
      const v = el.attribs ? el.attribs[name] : undefined;
      return v == null ? null : v;
    },
    attrOf: ($, selector, name) => {
      const el = $(selector).get(0);
      if (!el || !el.attribs) return null;
      const v = el.attribs[name];
      return v == null ? null : v;
    },
    text: (el) => {
      const childTexts = [];
      if (el.children) {
        for (const c of el.children) {
          if (c.type === 'text') childTexts.push(c.data);
        }
      }
      return childTexts.join('');
    },
  };
}
