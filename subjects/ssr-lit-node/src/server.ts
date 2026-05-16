import express from 'express';
import { render } from '@lit-labs/ssr';
import { html } from 'lit';
import { RenderResultReadable } from '@lit-labs/ssr/lib/render-result-readable.js';
import path from 'path';
import { fileURLToPath } from 'url';

// Import components to register them
import './components/counter-element.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;

// Serve static assets (bundled client)
app.use('/static', express.static(path.join(__dirname, 'static')));

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
