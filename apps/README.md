# Test Applications Specification (MGR Project)

This directory serves as the laboratory for the Master's Thesis research: *"Comparative Analysis of Server Performance for Web Applications in Client-Side Rendering (CSR) and Server-Side Rendering (SSR) Architectures"*.

Each directory herein is a **standardized scientific instrument** designed to measure the resource cost of specific rendering strategies under identical conditions.

## 🛠 Mandatory Application Requirements

To maintain research integrity and "apples-to-apples" comparability, every application MUST strictly adhere to the following specifications:

### 1. Functional Identity
All applications must be functionally indistinguishable to the load generator:
- **Root Path (`/`)**: Must render a top-level `<h1>Hello World</h1>` and the interactive counter component.
- **Dynamic Path (`/dynamic/:id`)**: Must render `<h1>Hello World</h1>` and a text element containing `Dynamic ID: [id]`.
- **Interactive Counter**: A client-side JavaScript component with an increment button. In SSR architectures, this component MUST trigger a hydration process.
- **Language**: All HTML documents must set `<html lang="pl">`.

### 2. Infrastructure Standards (Static Offloading)
Following the research methodology, all applications must implement the **Static Offloading Principle**:
- **Nginx Reverse Proxy**: Every app container must include Nginx acting as a reverse proxy.
- **Asset Delegation**: All static assets (CSS, JS, images) MUST be served directly by Nginx. The application process (Node.js, Bun, Deno) must be responsible ONLY for dynamic HTML generation.
- **Pathing**:
    - User assets must be stored in `/static/` and referenced via the `/static/` prefix.
    - Framework-specific assets (e.g., `/_next/static/`, `/_fresh/client/`) must be explicitly offloaded in the Nginx configuration.
- **Compression**: Gzip compression MUST be enabled at the Nginx level.

### 3. Visual Consistency
- **Global Styles**: All apps must utilize the identical global CSS file provided in the repository.
- **Standardized Titles**: The `<title>` tag must follow the pattern: `MGR - [Technology Name]`.

### 4. Technical Configuration
- **Internal Port**: The application server must listen on port `3000`.
- **External Ports**: Nginx must expose ports `80` (HTTP), `443` (HTTPS), and `8060` (Status).
- **Proxy Headers**: `trustProxy` (or equivalent) must be enabled to correctly handle `X-Forwarded-*` headers.

## 🧪 Runtime Environment (Standard 2026)

All applications use verified Alpine-based images to ensure minimum overhead and maximum reproducibility:

| Runtime / Component | Version / Image Tag |
| :--- | :--- |
| **Node.js** | `25.9.0-alpine3.23` |
| **Deno** | `alpine-2.7.13` |
| **Bun** | `1.3.13-alpine` |
| **NGINX** | `1.30.0-alpine3.23` |
| **Apache (httpd)** | `2.4.66-alpine3.23` |

## 📁 Application Registry

The project currently encompasses **30 implementations** (9 CSR/SSG, 21 SSR):

### SSR (Server-Side Rendering)
| App Name | Framework | Runtime |
| :--- | :--- | :--- |
| `ssr-analogjs` | AnalogJS / Angular | Node.js |
| `ssr-angular` | Angular SSR | Node.js |
| `ssr-astro` | Astro | Node.js |
| `ssr-astro-bun` | Astro | Bun |
| `ssr-fresh` | Fresh 2.0 / Preact | Deno |
| `ssr-lit` | Lit | Node.js |
| `ssr-nextjs` | Next.js / React | Node.js |
| `ssr-nextjs-bun` | Next.js / React | Bun |
| `ssr-nuxtjs` | Nuxt / Vue | Node.js |
| `ssr-nuxtjs-bun` | Nuxt / Vue | Bun |
| `ssr-qwik-city` | Qwik City | Node.js |
| `ssr-qwik-city-bun` | Qwik City | Bun |
| `ssr-react-router` | React Router | Node.js |
| `ssr-solid-start` | SolidStart | Node.js |
| `ssr-solid-start-bun` | SolidStart | Bun |
| `ssr-svelte-kit` | SvelteKit | Node.js |
| `ssr-svelte-kit-bun` | SvelteKit | Bun |
| `ssr-tanstack-start-react` | TanStack Start / React | Node.js |
| `ssr-tanstack-start-react-bun` | TanStack Start / React | Bun |
| `ssr-tanstack-start-solid` | TanStack Start / Solid | Node.js |
| `ssr-tanstack-start-solid-bun` | TanStack Start / Solid | Bun |

### CSR / SSG (Static Serving)
| App Name | Framework / Type | Server |
| :--- | :--- | :--- |
| `csr-angular` | Angular | Nginx |
| `csr-lit` | Lit | Nginx |
| `csr-react` | React | Nginx |
| `csr-solidjs` | SolidJS | Nginx |
| `csr-solidjs-apache` | SolidJS | Apache |
| `csr-svelte-kit-static` | SvelteKit (SSG) | Nginx |
| `csr-vanilla-nginx` | Vanilla HTML | Nginx |
| `csr-vanilla-apache` | Vanilla HTML | Apache |
| `csr-vue` | Vue | Nginx |

## ✅ Verification
Before inclusion in benchmark runs, every application MUST pass the automated verification suite:
```bash
# From the orchestrator directory
./venv/bin/mgr verify --apps [app-name]
```
The suite validates HTTP/HTTPS connectivity, Gzip headers, and content parity.
