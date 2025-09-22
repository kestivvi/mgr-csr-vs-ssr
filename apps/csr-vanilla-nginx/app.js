// Simple SPA Router - following KISS principle
class SimpleRouter {
    constructor() {
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
            return;
        }
        
        // Default to home page
        app.innerHTML = this.renderHome();
    }

    renderHome() {
        return `
            <div>
                <h1>Hello World</h1>
            </div>
        `;
    }

    renderDynamic(name) {
        return `
            <div>
                <h1>Hello, ${name}</h1>
            </div>
        `;
    }

}

// Initialize the router when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SimpleRouter();
});
