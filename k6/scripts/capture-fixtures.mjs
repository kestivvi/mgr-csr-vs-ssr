#!/usr/bin/env node
// Capture HTML + recursive same-origin assets from a running app
// (started separately via `mgr preview`). Saves payloads under
// fixtures/<app-id>/<route-slug>/{index.html, assets/<path>}.
//
// Usage:
//   node scripts/capture-fixtures.mjs <app-id> <port> <route1> [route2 ...]
// Example:
//   mgr preview ssr-svelte-kit-node -p 3013 &
//   node scripts/capture-fixtures.mjs ssr-svelte-kit-node 3013 / /dynamic-app/100123
//
// Recursion: follows imports in .js/.mjs files (static + dynamic) and
// asset refs in HTML, bounded to same-origin URLs. Stops on cycles.

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES_ROOT = join(HERE, '..', 'fixtures');

// Match minified forms too: `import{a,b}from"x"`, `import*as X from"x"`,
// `import "x"`, and dynamic `import("x")`.
const STATIC_FROM_RE = /\bimport\s*(?:[^;'"]*?)?from\s*["']([^"']+)["']/g;
const STATIC_BARE_RE = /\bimport\s*["']([^"']+)["']/g;
const DYNAMIC_IMPORT_RE = /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g;

function resolveAgainst(importerUrl, spec) {
  if (spec.startsWith('http://') || spec.startsWith('https://')) return spec;
  return new URL(spec, importerUrl).href;
}

function safeSlugRoute(route) {
  if (route === '/') return 'index';
  return route.replace(/^\//, '').replace(/\//g, '__');
}

function safePathFromUrl(url, origin) {
  const u = new URL(url);
  if (u.origin !== origin) throw new Error(`cross-origin asset: ${url}`);
  // strip leading /
  let p = u.pathname.replace(/^\//, '');
  if (p.endsWith('/') || p === '') p += 'index';
  if (u.search) p += '__' + u.search.replace(/[^\w]/g, '_');
  return p;
}

async function fetchText(url) {
  const res = await fetch(url);
  const buf = Buffer.from(await res.arrayBuffer());
  return { status: res.status, body: buf };
}

function extractFromHtml(html) {
  const out = new Set();
  // very permissive; capture script doesn't need correctness, just
  // breadth — extract.js (the unit under test) is what we'll validate.
  const tagRe = /<(?:link|script|img|astro-island)\b[^>]*>/gi;
  const attrRe = /(?:href|src|component-url|renderer-url)\s*=\s*["']([^"']+)["']/gi;
  for (const tag of html.match(tagRe) || []) {
    let m;
    attrRe.lastIndex = 0;
    while ((m = attrRe.exec(tag)) !== null) out.add(m[1]);
  }
  // inline scripts: static + dynamic imports
  const scriptBlocks = html.match(/<script\b[^>]*>([\s\S]*?)<\/script>/gi) || [];
  for (const block of scriptBlocks) {
    const body = block.replace(/^<script\b[^>]*>/i, '').replace(/<\/script>$/i, '');
    if (!body.includes('import')) continue;
    let m;
    while ((m = STATIC_FROM_RE.exec(body)) !== null) out.add(m[1]);
    while ((m = STATIC_BARE_RE.exec(body)) !== null) out.add(m[1]);
    while ((m = DYNAMIC_IMPORT_RE.exec(body)) !== null) out.add(m[1]);
  }
  return out;
}

function extractFromJs(jsText) {
  const out = new Set();
  let m;
  STATIC_FROM_RE.lastIndex = 0;
  STATIC_BARE_RE.lastIndex = 0;
  DYNAMIC_IMPORT_RE.lastIndex = 0;
  while ((m = STATIC_FROM_RE.exec(jsText)) !== null) out.add(m[1]);
  while ((m = STATIC_BARE_RE.exec(jsText)) !== null) out.add(m[1]);
  while ((m = DYNAMIC_IMPORT_RE.exec(jsText)) !== null) out.add(m[1]);
  return out;
}

async function main() {
  const [appId, portStr, ...routes] = process.argv.slice(2);
  if (!appId || !portStr || routes.length === 0) {
    console.error('Usage: capture-fixtures.mjs <app-id> <port> <route1> [route2 ...]');
    process.exit(2);
  }
  const origin = `http://localhost:${portStr}`;
  const appDir = join(FIXTURES_ROOT, appId);
  mkdirSync(appDir, { recursive: true });

  for (const route of routes) {
    const routeSlug = safeSlugRoute(route);
    const routeDir = join(appDir, routeSlug);
    const assetsDir = join(routeDir, 'assets');
    mkdirSync(assetsDir, { recursive: true });

    const htmlUrl = origin + route;
    const htmlRes = await fetchText(htmlUrl);
    if (htmlRes.status !== 200) {
      console.error(`! ${route}: status ${htmlRes.status}`);
      continue;
    }
    writeFileSync(join(routeDir, 'index.html'), htmlRes.body);

    const visited = new Set();
    const queue = [];
    for (const ref of extractFromHtml(htmlRes.body.toString('utf-8'))) {
      try { queue.push(resolveAgainst(htmlUrl, ref)); } catch {}
    }

    while (queue.length) {
      const url = queue.shift();
      if (visited.has(url)) continue;
      visited.add(url);
      let u;
      try { u = new URL(url); } catch { continue; }
      if (u.origin !== origin) continue;
      if (visited.size > 500) { console.warn('! cap 500'); break; }

      let assetRes;
      try { assetRes = await fetchText(url); } catch (e) {
        console.warn(`  skip ${url}: ${e.message}`);
        continue;
      }
      if (assetRes.status !== 200) {
        console.warn(`  ${url} -> ${assetRes.status}`);
        continue;
      }
      const relPath = safePathFromUrl(url, origin);
      const outPath = join(assetsDir, relPath);
      mkdirSync(dirname(outPath), { recursive: true });
      writeFileSync(outPath, assetRes.body);

      // recurse into .js / .mjs
      if (/\.(m?js)(?:\?|$)/.test(u.pathname)) {
        const text = assetRes.body.toString('utf-8');
        for (const imp of extractFromJs(text)) {
          try { queue.push(resolveAgainst(url, imp)); } catch {}
        }
      }
    }
    console.log(`  ${routeSlug}: html ${htmlRes.body.length} B, assets ${visited.size}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
