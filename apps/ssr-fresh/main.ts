import { App, staticFiles } from "fresh";

export const app = new App({
  trustProxy: true,
});

// Serve static files from the static/ directory
app.use(staticFiles());

// Enable file-system based routing
app.fsRoutes();

export default app;
