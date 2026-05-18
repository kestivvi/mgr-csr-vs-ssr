import { LitElement, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import type { RouterLocation } from '@vaadin/router';

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

@customElement('stock-badge')
export class StockBadge extends LitElement {
  @property({ type: Boolean }) inStock = false;

  protected override createRenderRoot() {
    return this;
  }

  render() {
    return this.inStock
      ? html`<span class="badge badge-in">IN</span>`
      : html`<span class="badge badge-out">OUT</span>`;
  }
}

@customElement('dynamic-app-page')
export class DynamicAppPage extends LitElement {
  @property({ type: Object })
  location?: RouterLocation;

  protected override createRenderRoot() {
    return this;
  }

  render() {
    const rawId = (this.location?.params?.id ?? '') as string;
    const displayId = parseInt(rawId, 10) || 1;
    const rows = generate(rawId, ROW_COUNT);
    const summary = rows.reduce(
      (acc, r) => ({
        count: acc.count + 1,
        inStock: acc.inStock + (r.inStock ? 1 : 0),
        sum: acc.sum + r.price,
      }),
      { count: 0, inStock: 0, sum: 0 },
    );

    return html`
      <div class="dynamic-app">
        <h1>Items for #${displayId}</h1>
        <p class="summary">
          ${summary.count} items · ${summary.inStock} in stock · total ${formatPrice(summary.sum)}
        </p>
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
  }
}
