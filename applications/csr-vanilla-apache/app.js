// Simple SPA Router - following KISS principle

const NOUNS = [
    'Lampa', 'Głośnik', 'Klawiatura', 'Mysz', 'Monitor',
    'Słuchawki', 'Kabel', 'Ładowarka', 'Adapter', 'Router',
    'Dysk', 'Kamera',
];
const SUFFIXES = ['Mk1', 'Mk2', 'Pro', 'Lite', 'Plus', 'Mini', 'Max', 'X'];
const CATEGORIES = ['Elektronika', 'Akcesoria', 'Audio', 'Biuro', 'Komputery'];
const ROW_COUNT = 100;

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

const formatPrice = (n) => n.toFixed(2) + ' PLN';

const stockBadge = (inStock) => inStock
    ? `<span class="badge badge-in">IN</span>`
    : `<span class="badge badge-out">OUT</span>`;

class SimpleRouter {
    constructor() {
        this.count = 0;
        this.init();
    }

    init() {
        this.handleRoute();
        window.addEventListener('popstate', () => this.handleRoute());
    }

    handleRoute() {
        const path = window.location.pathname;
        const app = document.getElementById('app');

        const dynamicAppMatch = path.match(/^\/dynamic-app\/(.+)$/);
        const dynamicMatch = path.match(/^\/dynamic\/(.+)$/);
        if (dynamicAppMatch) {
            app.innerHTML = this.renderDynamicApp(dynamicAppMatch[1]);
        } else if (dynamicMatch) {
            app.innerHTML = this.renderDynamic(dynamicMatch[1]);
        } else {
            app.innerHTML = this.renderHome();
        }

        this.bindEvents();
    }

    renderHome() {
        return `
            <div>
                <h1>Hello World</h1>
                <div>
                    <p>Count: <span id="counter-value">${this.count}</span></p>
                    <button id="counter-button">Increment</button>
                </div>
            </div>
        `;
    }

    renderDynamic(name) {
        return `
            <div>
                <h1>Hello World</h1>
                <p>Dynamic ID: ${name}</p>
                <div>
                    <p>Count: <span id="counter-value">${this.count}</span></p>
                    <button id="counter-button">Increment</button>
                </div>
            </div>
        `;
    }

    renderDynamicApp(id) {
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
        const rowsHtml = rows.map(row => `
            <div class="row">
                <span class="cell cell-id">#${row.id}</span>
                <span class="cell cell-name">${row.name}</span>
                <span class="cell cell-category">${row.category}</span>
                <span class="cell cell-price">${formatPrice(row.price)}</span>
                ${stockBadge(row.inStock)}
            </div>
        `).join('');
        return `
            <div class="dynamic-app">
                <h1>Items for #${displayId}</h1>
                <p class="summary">${summary.count} items · ${summary.inStock} in stock · total ${formatPrice(summary.sum)}</p>
                ${rowsHtml}
            </div>
        `;
    }

    bindEvents() {
        const button = document.getElementById('counter-button');
        if (button) {
            button.onclick = () => {
                this.count++;
                const valueDisplay = document.getElementById('counter-value');
                if (valueDisplay) {
                    valueDisplay.innerText = this.count;
                }
            };
        }
    }

}

// Initialize the router when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SimpleRouter();
});
