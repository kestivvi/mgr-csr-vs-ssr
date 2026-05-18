import express from 'express';
import { render } from '@lit-labs/ssr';
import { html } from 'lit';
import { RenderResultReadable } from '@lit-labs/ssr/lib/render-result-readable.js';
import path from 'path';
import { fileURLToPath } from 'url';

// Import components to register them
import './components/counter-element.js';
import './components/stock-badge.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Serve static assets (bundled client)
app.use('/static', express.static(path.join(__dirname, 'static')));

const NOUNS = [
  'Lampa', 'Głośnik', 'Klawiatura', 'Mysz', 'Monitor',
  'Słuchawki', 'Kabel', 'Ładowarka', 'Adapter', 'Router',
  'Dysk', 'Kamera',
];
const SUFFIXES = ['Mk1', 'Mk2', 'Pro', 'Lite', 'Plus', 'Mini', 'Max', 'X'];
const CATEGORIES = ['Elektronika', 'Akcesoria', 'Audio', 'Biuro', 'Komputery'];
const ROW_COUNT = 100;

type Row = {
  id: number;
  name: string;
  category: string;
  price: number;
  inStock: boolean;
};

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function generate(id: string | undefined, n: number): Row[] {
  const seed = parseInt(id ?? '', 10) || 1;
  const rand = mulberry32(seed);
  const rows: Row[] = [];
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

const formatPrice = (n: number): string => n.toFixed(2) + ' PLN';

// Root route
app.get('/', (req, res) => {
  const content = html`
    <main id="root">
      <h1>Hello World</h1>
      <counter-element></counter-element>
    </main>
  `;
  renderPage(res, content);
});

// Dynamic route
app.get('/dynamic/:id', (req, res) => {
  const id = req.params.id;
  const content = html`
    <main id="root">
      <h1>Hello World</h1>
      <p>Dynamic ID: ${id}</p>
    </main>
  `;
  renderPage(res, content);
});

// Dynamic App route — /dynamic-app/:id (SSR with 100 rows)
app.get('/dynamic-app/:id', (req, res) => {
  const id = req.params.id;
  const displayId = parseInt(id, 10) || 1;
  const rows = generate(id, ROW_COUNT);
  const summary = rows.reduce(
    (acc, r) => ({
      count: acc.count + 1,
      inStock: acc.inStock + (r.inStock ? 1 : 0),
      sum: acc.sum + r.price,
    }),
    { count: 0, inStock: 0, sum: 0 },
  );
  const content = html`
    <div class="dynamic-app">
      <h1>${`Items for #${displayId}`}</h1>
      <p class="summary">${`${summary.count} items · ${summary.inStock} in stock · total ${formatPrice(summary.sum)}`}</p>
      ${rows.map(
        (row) => html`
          <div class="row">
            <span class="cell cell-id">#${row.id}</span>
            <span class="cell cell-name">${row.name}</span>
            <span class="cell cell-category">${row.category}</span>
            <span class="cell cell-price">${formatPrice(row.price)}</span>
            <stock-badge ?inStock=${row.inStock}></stock-badge>
          </div>
        `,
      )}
    </div>
  `;
  renderPage(res, content);
});

function renderPage(res: express.Response, content: unknown) {
  const page = html`
    <!DOCTYPE html>
    <html lang="pl">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>MGR - Lit SSR</title>
        <link rel="stylesheet" href="/static/global.css" />
        <style>
          /* Inline critical styles to match the body styling of global.css */
          body {
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 0;
            background: #f4f4f9;
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            text-align: center;
          }
          #root {
            background: white;
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          }
        </style>
      </head>
      <body>
        ${content}
        <script type="module" src="/static/client.js"></script>
      </body>
    </html>
  `;

  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  const ssrResult = render(page);
  new RenderResultReadable(ssrResult).pipe(res);
}

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
