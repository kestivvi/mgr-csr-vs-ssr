import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as cheerio from 'cheerio';

import {
  discoverAssets,
  cheerioParserAdapter,
  parseEsmImports,
  resolveUrl,
  normalizePath,
  splitUrl,
  sameOrigin,
} from './extract.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');

function loadFixture(appId, routeSlug) {
  const routeDir = join(FIXTURES, appId, routeSlug);
  const assetsDir = join(routeDir, 'assets');
  const html = readFileSync(join(routeDir, 'index.html'), 'utf-8');
  const origin = 'http://localhost:9999';
  // Match the URL a browser uses as the document base. Index has a
  // trailing slash; nested paths do not (mirrors what `/dynamic-app/:id`
  // looks like in the address bar).
  const baseUrl =
    routeSlug === 'index' ? origin + '/' : origin + '/' + routeSlug.replace(/__/g, '/');

  const fetcher = (url) => {
    const u = new URL(url);
    if (u.origin !== origin) return { status: 0, body: '' };
    const rel = u.pathname.replace(/^\//, '');
    const path = join(assetsDir, rel);
    if (!existsSync(path) || !statSync(path).isFile()) return { status: 404, body: '' };
    return { status: 200, body: readFileSync(path, 'utf-8'), bytes: statSync(path).size };
  };
  return { html, baseUrl, fetcher };
}

function totalBytes(assetSet, fetcher, htmlSize) {
  let total = htmlSize;
  for (const url of assetSet) {
    const r = fetcher(url);
    if (r.status === 200) total += (r.bytes ?? Buffer.byteLength(r.body));
  }
  return total;
}

// Generic regression net: each framework's /dynamic-app/:id payload must
// at minimum reach a known floor of bytes and include the chunks the
// framework cannot live without. Floors come from a one-time live capture;
// raise them only when an app legitimately grows.
const REGRESSION_CASES = [
  { app: 'ssr-nuxtjs-node',         route: 'dynamic-app__100123', minBytes:  38_000, mustInclude: [/_nuxt\/.+\.js$/] },
  { app: 'ssr-react-router-node',   route: 'dynamic-app__100123', minBytes: 320_000, mustInclude: [/entry\.client/, /jsx-runtime/] },
  { app: 'ssr-lit-node',            route: 'dynamic-app__100123', minBytes:  68_000, mustInclude: [/\/static\/client\.js$/] },
  { app: 'ssr-solid-start-node',    route: 'dynamic-app__100123', minBytes:  40_000, mustInclude: [/_build\/assets\/.+\.js$/] },
  { app: 'ssr-tanstack-start-react-node', route: 'dynamic-app__100123', minBytes: 200_000, mustInclude: [/assets\/.+\.js$/] },
  // Fresh and Astro have no client islands on /dynamic-app/:id (pure SSR
  // table); islands only on /, where the hydration bundle is shipped.
  { app: 'ssr-fresh-deno',          route: 'index',               minBytes:  25_000, mustInclude: [/client-entry-/] },
  { app: 'ssr-astro-react-node',    route: 'index',               minBytes: 190_000, mustInclude: [/_astro\/client\./, /_astro\/Counter\./] },
  // CSR (nginx try_files → index.html). The dynamic route serves the
  // same shell as /, so byte totals match. Tests that recursive ESM
  // follow ALSO works for CSR bundles, not just SSR.
  { app: 'csr-react-nginx',         route: 'dynamic-app__100123', minBytes: 280_000, mustInclude: [/assets\/index-.*\.js$/] },
  { app: 'csr-svelte-kit-nginx',    route: 'dynamic-app__100123', minBytes:  70_000, mustInclude: [/entry\/start\./, /chunks\//] },
  { app: 'csr-vanilla-nginx',       route: 'dynamic-app__100123', minBytes:   3_000, mustInclude: [/\/app\.js$/, /\/styles\.css$/] },
];

describe.each(REGRESSION_CASES)('discoverAssets: $app [$route]', ({ app, route, minBytes, mustInclude }) => {
  it('meets byte floor and includes critical chunks', () => {
    const { html, baseUrl, fetcher } = loadFixture(app, route);
    const assets = discoverAssets({
      html, baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio),
      fetcher,
    });
    const urls = [...assets];
    for (const re of mustInclude) {
      expect(urls.some((u) => re.test(u)), `missing pattern ${re}`).toBe(true);
    }
    expect(totalBytes(assets, fetcher, Buffer.byteLength(html))).toBeGreaterThanOrEqual(minBytes);
  });
});

describe('discoverAssets: Qwik /dynamic-app/:id', () => {
  it('parses bundle-graph.json and adds lazy q-*.js bundles', () => {
    const { html, baseUrl, fetcher } = loadFixture('ssr-qwik-city-node', 'dynamic-app__100123');
    const assets = discoverAssets({
      html, baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio),
      fetcher,
    });
    const urls = [...assets];
    // bundle-graph.json itself is discovered via <link rel="preload">.
    expect(urls.some((u) => /bundle-graph[^/]*\.json$/.test(u))).toBe(true);
    // These chunks only appear as string literals inside the JSON manifest;
    // the standard scan cannot reach them without parsing it.
    expect(urls.some((u) => /\/q-15auWeVR\.js$/.test(u))).toBe(true);
    expect(urls.some((u) => /\/q-CfRrFzvN\.js$/.test(u))).toBe(true);
    // Resolved under the q:base path declared on <html>.
    expect(urls.find((u) => /\/q-15auWeVR\.js$/.test(u))).toMatch(/\/build\/q-15auWeVR\.js$/);
  });
});

describe('splitUrl', () => {
  it('parses scheme/host/path/search/hash', () => {
    expect(splitUrl('http://example.com/a/b?x=1#h')).toEqual({
      scheme: 'http:', host: 'example.com', path: '/a/b', search: '?x=1', hash: '#h',
    });
  });
  it('returns sane defaults when path/search/hash absent', () => {
    expect(splitUrl('https://x.test')).toEqual({
      scheme: 'https:', host: 'x.test', path: '/', search: '', hash: '',
    });
  });
  it('returns null for non-http(s)', () => {
    expect(splitUrl('ftp://x/y')).toBeNull();
    expect(splitUrl('not-a-url')).toBeNull();
  });
});

describe('normalizePath', () => {
  it.each([
    ['/a/b/c',           '/a/b/c'],
    ['/a/./b',           '/a/b'],
    ['/a/b/../c',        '/a/c'],
    ['/a/b/../../c',     '/c'],
    ['/a/b/../../../c',  '/c'], // do not escape root
    ['a/b/./c',          'a/b/c'],
    ['/',                '/'],
    ['/a/b/',            '/a/b/'],
    ['',                 '/'],
  ])('normalises %s -> %s', (input, expected) => {
    expect(normalizePath(input)).toBe(expected);
  });
});

describe('resolveUrl', () => {
  const base = 'http://app.test/page/sub';
  it('passes data: and # through as ignored', () => {
    expect(resolveUrl('data:image/png;base64,AAA', base)).toBeNull();
    expect(resolveUrl('#anchor', base)).toBeNull();
  });
  it('returns absolute http(s) URLs unchanged', () => {
    expect(resolveUrl('https://cdn.x/y.js', base)).toBe('https://cdn.x/y.js');
  });
  it('keeps protocol of importer for scheme-relative //', () => {
    expect(resolveUrl('//cdn.x/y.js', 'http://app.test/')).toBe('http://cdn.x/y.js');
    expect(resolveUrl('//cdn.x/y.js', 'https://app.test/')).toBe('https://cdn.x/y.js');
  });
  it('joins root-relative /x against importer origin', () => {
    expect(resolveUrl('/style.css', base)).toBe('http://app.test/style.css');
  });
  it('drops the importer file segment for dot-relative refs', () => {
    expect(resolveUrl('./a.js', 'http://x.test/dir/file.html'))
      .toBe('http://x.test/dir/a.js');
    expect(resolveUrl('../a.js', 'http://x.test/dir/file.html'))
      .toBe('http://x.test/a.js');
  });
  it('does not climb above the origin root', () => {
    expect(resolveUrl('../../../a.js', 'http://x.test/dir/file.html'))
      .toBe('http://x.test/a.js');
  });
  it('handles bare relative (no leading dot)', () => {
    expect(resolveUrl('img/a.png', 'http://x.test/dir/'))
      .toBe('http://x.test/dir/img/a.png');
  });
  it('returns null when importer is unparseable', () => {
    expect(resolveUrl('a.js', 'not-a-url')).toBeNull();
  });
});

describe('sameOrigin', () => {
  it('matches by scheme + host', () => {
    expect(sameOrigin('http://x.test/a', 'http://x.test/b')).toBe(true);
    expect(sameOrigin('http://x.test/a', 'https://x.test/b')).toBe(false);
    expect(sameOrigin('http://x.test/a', 'http://y.test/b')).toBe(false);
  });
  it('false on garbage input', () => {
    expect(sameOrigin('whatever', 'http://x.test')).toBe(false);
  });
});

describe('parseEsmImports', () => {
  it('catches `import "x"` (bare static, no whitespace before quote)', () => {
    expect(parseEsmImports('import"a.js";import "b.js";')).toEqual(['a.js', 'b.js']);
  });
  it('catches minified `import{x}from"y"`', () => {
    expect(parseEsmImports('import{a,b as c}from"./chunk.js";')).toEqual(['./chunk.js']);
  });
  it('catches `import * as X from "y"`', () => {
    expect(parseEsmImports('import * as X from "./mod.js";')).toEqual(['./mod.js']);
  });
  it('catches dynamic `import("y")`', () => {
    expect(parseEsmImports('const m = import("./lazy.js"); import ( "./b.js" );'))
      .toEqual(expect.arrayContaining(['./lazy.js', './b.js']));
  });
  it('finds multiple imports in one source', () => {
    const src = 'import a from"./a.js";import"./b.js";import("./c.js");';
    expect(parseEsmImports(src).sort()).toEqual(['./a.js', './b.js', './c.js']);
  });
  it('returns [] for source without "import" token', () => {
    expect(parseEsmImports('const x = 1; export default x;')).toEqual([]);
  });
  it('coerces non-string defensively (no throw)', () => {
    expect(parseEsmImports(null)).toEqual([]);
    expect(parseEsmImports(undefined)).toEqual([]);
    expect(parseEsmImports({ toString: () => 'import "x.js"' })).toEqual(['x.js']);
  });
});

describe('discoverAssets: edge cases', () => {
  const baseUrl = 'http://app.test/';
  const noopFetcher = () => ({ status: 404, body: '' });

  it('returns empty set for empty body', () => {
    const assets = discoverAssets({
      html: '', baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio), fetcher: noopFetcher,
    });
    expect(assets.size).toBe(0);
  });

  it('ignores data: URLs and # anchors entirely', () => {
    const html = `<!doctype html><html><head>
      <link rel="icon" href="data:image/png;base64,AAA">
      <link rel="stylesheet" href="#fragment">
      <link rel="stylesheet" href="/real.css">
    </head></html>`;
    const assets = discoverAssets({
      html, baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio), fetcher: noopFetcher,
    });
    expect([...assets]).toEqual(['http://app.test/real.css']);
  });

  it('deduplicates assets referenced multiple times', () => {
    const html = `<!doctype html><html><head>
      <link rel="modulepreload" href="/a.js">
      <script src="/a.js"></script>
      <script type="module">import "/a.js";</script>
    </head></html>`;
    const assets = discoverAssets({
      html, baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio), fetcher: noopFetcher,
    });
    expect([...assets]).toEqual(['http://app.test/a.js']);
  });

  it('tolerates malformed HTML without throwing', () => {
    const html = '<html><head><link href=/a.css><script src="/b.js"';
    expect(() => discoverAssets({
      html, baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio), fetcher: noopFetcher,
    })).not.toThrow();
  });

  it('skips recursion for cross-origin .js (no fetcher leak)', () => {
    const html = `<!doctype html><html><head>
      <script src="https://cdn.other/x.js"></script>
    </head></html>`;
    let fetched = 0;
    const fetcher = (url) => { fetched++; return { status: 200, body: 'import "./y.js";' }; };
    const assets = discoverAssets({
      html, baseUrl, parserAdapter: cheerioParserAdapter(cheerio), fetcher,
    });
    expect(fetched).toBe(0); // cross-origin .js never traversed
    expect([...assets]).toEqual(['https://cdn.other/x.js']);
  });

  it('honours maxAssets to bound runaway graphs', () => {
    const html = `<script>import "/c0.js";</script>`;
    // Each chunk imports the next: c0 -> c1 -> c2 -> ...
    const fetcher = (url) => {
      const m = /\/c(\d+)\.js$/.exec(url);
      if (!m) return { status: 404, body: '' };
      const next = parseInt(m[1], 10) + 1;
      return { status: 200, body: `import "/c${next}.js";` };
    };
    const assets = discoverAssets({
      html, baseUrl, parserAdapter: cheerioParserAdapter(cheerio), fetcher, maxAssets: 5,
    });
    // Discovery stops once the visit cap is reached.
    expect(assets.size).toBeLessThanOrEqual(6); // initial + up to 5 visited
  });
});

describe('discoverAssets: cycle / loop safety', () => {
  it('terminates when JS chunks import each other circularly', () => {
    const html = `<!doctype html><html><body>
      <script type="module">import "/app/a.js";</script>
    </body></html>`;
    const origin = 'http://example.com';
    const baseUrl = origin + '/';
    // a.js -> b.js -> a.js (cycle)
    const sources = new Map([
      [origin + '/app/a.js', 'import "./b.js";'],
      [origin + '/app/b.js', 'import "./a.js";'],
    ]);
    const fetcher = (url) =>
      sources.has(url) ? { status: 200, body: sources.get(url) } : { status: 404, body: '' };

    const assets = discoverAssets({
      html, baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio),
      fetcher,
    });
    expect(assets.size).toBe(2);
    expect([...assets]).toEqual(expect.arrayContaining([
      origin + '/app/a.js',
      origin + '/app/b.js',
    ]));
  });
});

describe('discoverAssets: SvelteKit /dynamic-app/:id', () => {
  it('follows recursive ESM imports to reach chunk files', () => {
    const { html, baseUrl, fetcher } = loadFixture('ssr-svelte-kit-node', 'dynamic-app__100123');
    const assets = discoverAssets({
      html, baseUrl,
      parserAdapter: cheerioParserAdapter(cheerio),
      fetcher,
    });
    const urls = [...assets];
    // Must include both entry stubs AND at least one chunk reached recursively.
    expect(urls.some((u) => /\/entry\/start\.[^/]+\.js$/.test(u))).toBe(true);
    expect(urls.some((u) => /\/entry\/app\.[^/]+\.js$/.test(u))).toBe(true);
    expect(urls.some((u) => /\/chunks\/[^/]+\.js$/.test(u))).toBe(true);
    // End-to-end byte count should reflect deep traversal, not just stubs.
    const total = totalBytes(assets, fetcher, Buffer.byteLength(html));
    expect(total).toBeGreaterThan(60_000);
  });
});
