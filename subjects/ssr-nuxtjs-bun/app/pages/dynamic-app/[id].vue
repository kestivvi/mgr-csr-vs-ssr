<script setup lang="ts">
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

const route = useRoute();
const rawId = route.params.id;
const id = Array.isArray(rawId) ? rawId[0] : rawId;
const displayId = parseInt(id ?? '', 10) || 1;
const rows = generate(id, ROW_COUNT);
const summary = rows.reduce(
  (acc, r) => ({
    count: acc.count + 1,
    inStock: acc.inStock + (r.inStock ? 1 : 0),
    sum: acc.sum + r.price,
  }),
  { count: 0, inStock: 0, sum: 0 },
);
</script>

<template>
  <div class="dynamic-app">
    <h1>{{ `Items for #${displayId}` }}</h1>
    <p class="summary">{{ `${summary.count} items · ${summary.inStock} in stock · total ${formatPrice(summary.sum)}` }}</p>
    <div v-for="row in rows" :key="row.id" class="row">
      <span class="cell cell-id">{{ `#${row.id}` }}</span>
      <span class="cell cell-name">{{ row.name }}</span>
      <span class="cell cell-category">{{ row.category }}</span>
      <span class="cell cell-price">{{ formatPrice(row.price) }}</span>
      <StockBadge :in-stock="row.inStock" />
    </div>
  </div>
</template>

<style scoped>
.dynamic-app { width: min(1000px, calc(100vw - 2rem)); margin: 0 auto; padding: 1rem; font-family: system-ui, sans-serif; text-align: left; }
.dynamic-app h1 { font-size: 1.5rem; margin: 0 0 0.5rem 0; color: #222; }
.dynamic-app .summary { color: #555; margin: 0 0 1rem 0; font-size: 0.95rem; }
.dynamic-app .row { display: flex; gap: 1rem; padding: 0.35rem 0; border-bottom: 1px solid #eee; align-items: center; font-size: 0.9rem; }
.dynamic-app .cell { flex: 1; }
.dynamic-app .cell-id { flex: 0 0 3rem; color: #888; }
.dynamic-app .cell-name { flex: 2; }
.dynamic-app .cell-category { flex: 1; color: #666; }
.dynamic-app .cell-price { flex: 0 0 6rem; text-align: right; }
.dynamic-app .badge { display: inline-block; padding: 0.1rem 0.45rem; border-radius: 3px; font-size: 0.7rem; font-weight: 700; min-width: 2rem; text-align: center; }
.dynamic-app .badge-in { background: #d4edda; color: #155724; }
.dynamic-app .badge-out { background: #f8d7da; color: #721c24; }
</style>
