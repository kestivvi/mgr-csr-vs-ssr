# `/dynamic-app/{id}` — Canonical Page Specification

**Status:** Source of truth for all 32 ports.
**Reference implementation:** [`csr-react-nginx/src/pages/DynamicApp.tsx`](../../csr-react-nginx/src/pages/DynamicApp.tsx)
**Reference screenshot:** [`reference.png`](./reference.png) (captured at 1280×800 from `/dynamic-app/42`)

This document is the only reference a port author needs. If you find yourself reading the PRD or any conversation context to port the page, that is a SPEC bug — file an issue.

---

## 1. Route

- **Path:** `/dynamic-app/:id`
- **Param parsing:** `parseInt(id, 10) || 1`. Treat `id` as opaque. Never return 400.
- **Coexists with** the existing `/dynamic/:name` route. Both are load-bearing.

## 2. Row schema

All fields are primitives. No nesting. No dates. No arrays inside rows.

```ts
type Row = {
  id: number;        // 1..100, just the index
  name: string;      // `${noun} ${suffix}`, e.g. "Lampa Mk1"
  category: string;  // one of 5 fixed categories
  price: number;     // 0.00..999.99, two decimals
  inStock: boolean;  // ~65% true
};
```

## 3. Constants — copy verbatim

```js
const NOUNS = [
  'Lampa', 'Głośnik', 'Klawiatura', 'Mysz', 'Monitor',
  'Słuchawki', 'Kabel', 'Ładowarka', 'Adapter', 'Router',
  'Dysk', 'Kamera',
];

const SUFFIXES = ['Mk1', 'Mk2', 'Pro', 'Lite', 'Plus', 'Mini', 'Max', 'X'];

const CATEGORIES = ['Elektronika', 'Akcesoria', 'Audio', 'Biuro', 'Komputery'];

const ROW_COUNT = 100;
```

Names are `${noun} ${suffix}` (e.g. `Lampa Mk1`, `Klawiatura Pro`). No adjectives, no Polish gender agreement — keep the generator trivial.

## 4. Procedural generator — copy verbatim

```js
function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function generate(id, n) {
  const seed = parseInt(id, 10) || 1;
  const rand = mulberry32(seed);
  const rows = [];
  for (let i = 1; i <= n; i++) {
    const noun = NOUNS[Math.floor(rand() * NOUNS.length)];
    const suffix = SUFFIXES[Math.floor(rand() * SUFFIXES.length)];
    const category = CATEGORIES[Math.floor(rand() * CATEGORIES.length)];
    const price = Math.floor(rand() * 100000) / 100;
    const inStock = rand() < 0.65;
    rows.push({ id: i, name: `${noun} ${suffix}`, category, price, inStock });
  }
  return rows;
}
```

**Rules:**

- Each port copy-pastes this generator into its page file (or a co-located helper if idiomatic). No shared package.
- Translate to TypeScript types / language idioms as needed — but the *numeric output* per `id` must match the JavaScript reference byte-for-byte. Cross-port output equality is not asserted, but divergence from the JS reference is a port bug.
- Call exactly: `const rows = generate(id, ROW_COUNT)` where `id` is the URL param (string).

## 5. Price formatting — hand-rolled, no library

```js
const formatPrice = (n) => n.toFixed(2) + ' PLN';
```

**Banned:**

- `Intl.NumberFormat`
- Angular `CurrencyPipe`
- Vue filters / Svelte stores wrapping a formatter
- Any locale-aware library

Reason: locale-formatter CPU costs vary wildly across runtimes and would confound the rendering benchmark.

## 6. DOM structure

```
<div class="dynamic-app">
  <h1>Items for #{id}</h1>
  <p class="summary">{count} items · {inStock} in stock · total {formatPrice(sum)}</p>

  <div class="row">                                  ← repeats exactly 100 times
    <span class="cell cell-id">#{row.id}</span>
    <span class="cell cell-name">{row.name}</span>
    <span class="cell cell-category">{row.category}</span>
    <span class="cell cell-price">{formatPrice(row.price)}</span>
    <StockBadge inStock={row.inStock} />             ← nested sub-component
  </div>
  ...
</div>
```

**`<StockBadge>` contract:**

- Receives exactly one prop: `inStock: boolean`.
- Renders: `<span class="badge badge-in">IN</span>` if `true`, `<span class="badge badge-out">OUT</span>` if `false`.
- Must be a distinct component (not an inline ternary in the parent). This is what gives the page its "component composition" claim.

**Summary line computation:** a single `reduce` over rows producing `{ count, inStock, sum }`. Do not split into three passes.

**Markers preserved across all 32 ports (verifier contract — see issue 003):**

- The literal substring `Items for #{id}` must appear exactly once.
- The substring `class="row"` (or framework-equivalent serialization for SSR) must appear exactly 100 times in SSR responses.
- For CSR ports the server delivers only a shell; markers materialize in the browser.

## 7. CSS visual contract

Idiomatic per framework (S2: `<style scoped>`, CSS modules, `ViewEncapsulation`, plain `<link>` — author's call). The *values* below are the contract; the *mechanism* is per-framework.

```css
.dynamic-app                { width: min(1000px, calc(100vw - 2rem)); margin: 0 auto;
                              padding: 1rem; font-family: system-ui, sans-serif;
                              text-align: left; }
.dynamic-app h1             { font-size: 1.5rem; margin: 0 0 0.5rem 0; color: #222; }
.dynamic-app .summary       { color: #555; margin: 0 0 1rem 0; font-size: 0.95rem; }
.dynamic-app .row           { display: flex; gap: 1rem; padding: 0.35rem 0;
                              border-bottom: 1px solid #eee; align-items: center;
                              font-size: 0.9rem; }
.dynamic-app .cell          { flex: 1; }
.dynamic-app .cell-id       { flex: 0 0 3rem;  color: #888; }
.dynamic-app .cell-name     { flex: 2; }
.dynamic-app .cell-category { flex: 1; color: #666; }
.dynamic-app .cell-price    { flex: 0 0 6rem; text-align: right; }
.dynamic-app .badge         { display: inline-block; padding: 0.1rem 0.45rem;
                              border-radius: 3px; font-size: 0.7rem; font-weight: 700;
                              min-width: 2rem; text-align: center; }
.dynamic-app .badge-in      { background: #d4edda; color: #155724; }
.dynamic-app .badge-out     { background: #f8d7da; color: #721c24; }
```

No CSS framework, no reset, no media queries, no animations. The reference screenshot is the visual ground truth — match it visually, not byte-for-byte in markup.

If a port's global stylesheet centres or flex-aligns `body`/`#root`, scope your new styles tightly under `.dynamic-app` so existing pages stay untouched.

## 8. CSR vs SSR contract

- **SSR ports:** server runs `generate(id, 100)` and serialises all 100 rows into the HTML response on every request. Payload ≈ 10 KB.
- **CSR ports:** server delivers the static shell (≈ 1–2 KB). The generator runs in the browser on mount. k6 does not execute JS — it measures only shell delivery.

Both produce the **same visible result in a real browser**. The asymmetry is the experiment, not a bug.

## 9. Verifier expectations (issue 003 — preview)

After `docker compose up`, `mgr verify` will issue `GET /dynamic-app/42` and:

- All ports → assert HTTP 200.
- SSR ports only → additionally assert body contains `Items for #42` and exactly 100 occurrences of `class="row"` (or the framework's serialized equivalent).
- CSR ports → 200 only. No body checks.

A port that fails to register the route, throws in the generator, or produces the wrong row count will fail verify locally — before any AWS spend.

## 10. Porting checklist

Per Application:

- [ ] Add page file `DynamicApp.{tsx,vue,svelte,ts,html,…}` with generator copy-pasted in (or a co-located helper).
- [ ] Register `/dynamic-app/:id` in the framework's router config — one line.
- [ ] Apply idiomatic styling matching the CSS visual contract in §7.
- [ ] No Dockerfile changes. No build-config changes. No new dependencies.
- [ ] Local `docker compose up` → `curl http://localhost:<port>/dynamic-app/42` → 200.
- [ ] Open in browser → page matches [`reference.png`](./reference.png).
