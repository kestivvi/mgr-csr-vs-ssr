// Simple SPA Router - following KISS principle
class SimpleRouter {
    constructor() {
        this.count = 0;
        this.init();
    }

    init() {
        // Handle initial load
        this.handleRoute();
        
        // Handle browser back/forward
        window.addEventListener('popstate', () => this.handleRoute());
    }

    handleRoute() {
        const path = window.location.pathname;
        const app = document.getElementById('app');
        
        // Check for dynamic routes first
        const dynamicMatch = path.match(/^\/dynamic\/(.+)$/);
        if (dynamicMatch) {
            const name = dynamicMatch[1];
            app.innerHTML = this.renderDynamic(name);
        } else {
            // Default to home page
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
