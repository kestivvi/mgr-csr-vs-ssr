# Research Subjects Specification (MGR Project)

This directory serves as the laboratory for the Master's Thesis research: *"Comparative Analysis of Server Performance for Web Applications in Client-Side Rendering (CSR) and Server-Side Rendering (SSR) Architectures"*.

Each directory herein is a **standardized scientific instrument** (a **Subject**) designed to measure the resource cost of specific rendering strategies under identical conditions.

## 🛠 Mandatory Subject Requirements

To maintain research integrity and "apples-to-apples" comparability, every **Subject** MUST strictly adhere to the following specifications:

### 1. Functional Identity
All subjects must be functionally indistinguishable to the load generator:
- **Root Path (`/`)**: Must render a top-level `<h1>Hello World</h1>` and the interactive counter component.
- **Dynamic Path (`/dynamic/:id`)**: Must render `<h1>Hello World</h1>` and a text element containing `Dynamic ID: [id]`.
- **Interactive Counter**: A client-side JavaScript component with an increment button. In SSR architectures, this component MUST trigger a hydration process.
- **Language**: All HTML documents must set `<html lang="pl">`.

### 2. Infrastructure Standards (Static Offloading)
Following the research methodology, all subjects must implement the **Static Offloading Principle**:
- **Web Server / Proxy**: Every subject container must include a front-facing web server (Nginx/Apache) acting as a reverse proxy.
- **Asset Delegation**: All static assets (CSS, JS, images) MUST be served directly by the web server. The application runtime (Node.js, Bun, Deno) must be responsible ONLY for dynamic HTML generation.
- **Pathing**:
    - User assets must be stored in `/static/` and referenced via the `/static/` prefix.
    - Framework-specific assets (e.g., `/_next/static/`, `/_fresh/client/`) must be explicitly offloaded in the web server configuration.
- **Compression**: Gzip compression MUST be enabled at the web server level.

### 3. Visual Consistency
- **Global Styles**: All subjects must utilize the identical global CSS file provided in the repository.
- **Standardized Titles**: The `<title>` tag must follow the pattern: `MGR - [Subject ID]`.

### 4. Technical Configuration
- **Internal Port**: The application runtime must listen on port `3000`.
- **External Ports**: The web server must expose ports `80` (HTTP), `443` (HTTPS), and `8060` (Status).
- **Proxy Headers**: `trustProxy` (or equivalent) must be enabled to correctly handle `X-Forwarded-*` headers.

## 🧪 Runtime Environment (Standard 2026)

All subjects use verified Alpine-based images to ensure minimum overhead and maximum reproducibility:

| Runtime / Component | Version / Image Tag |
| :--- | :--- |
| **Node.js** | `25.9.0-alpine3.23` |
| **Deno** | `alpine-2.7.13` |
| **Bun** | `1.3.13-alpine` |
| **NGINX** | `1.30.0-alpine3.23` |
| **Apache (httpd)** | `2.4.66-alpine3.23` |

## 📁 Subject Registry

The project currently encompasses **31 Research Subjects** (10 CSR/SSG, 21 SSR). The registry is maintained automatically via **Subject Autodiscovery**.

### SSR (Server-Side Rendering)
| Subject ID | Framework | Runtime |
| :--- | :--- | :--- |
| `ssr-analogjs-node` | AnalogJS / Angular | Node.js |
| `ssr-angular-node` | Angular SSR | Node.js |
| `ssr-astro-react-node` | Astro (React) | Node.js |
| `ssr-astro-react-bun` | Astro (React) | Bun |
| `ssr-fresh-deno` | Fresh 2.0 / Preact | Deno |
| `ssr-lit-node` | Lit | Node.js |
| `ssr-nextjs-node` | Next.js / React | Node.js |
| `ssr-nextjs-bun` | Next.js / React | Bun |
| `ssr-nuxtjs-node` | Nuxt / Vue | Node.js |
| `ssr-nuxtjs-bun` | Nuxt / Vue | Bun |
| `ssr-qwik-city-node` | Qwik City | Node.js |
| `ssr-qwik-city-bun` | Qwik City | Bun |
| `ssr-react-router-node` | React Router | Node.js |
| `ssr-solid-start-node` | SolidStart | Node.js |
| `ssr-solid-start-bun` | SolidStart | Bun |
| `ssr-svelte-kit-node` | SvelteKit | Node.js |
| `ssr-svelte-kit-bun` | SvelteKit | Bun |
| `ssr-tanstack-start-react-node` | TanStack Start / React | Node.js |
| `ssr-tanstack-start-react-bun` | TanStack Start / React | Bun |
| `ssr-tanstack-start-solid-node` | TanStack Start / Solid | Node.js |
| `ssr-tanstack-start-solid-bun` | TanStack Start / Solid | Bun |

### CSR / SSG (Static Serving)
| Subject ID | Framework / Type | Web Server |
| :--- | :--- | :--- |
| `csr-angular-nginx` | Angular | Nginx |
| `csr-lit-nginx` | Lit | Nginx |
| `csr-react-nginx` | React | Nginx |
| `csr-nextjs-nginx` | Next.js (Export) | Nginx |
| `csr-solid-nginx` | SolidJS | Nginx |
| `csr-solid-apache` | SolidJS | Apache |
| `csr-svelte-kit-nginx` | SvelteKit (Static) | Nginx |
| `csr-vanilla-nginx` | Vanilla HTML | Nginx |
| `csr-vanilla-apache` | Vanilla HTML | Apache |
| `csr-vue-nginx` | Vue | Nginx |

## ✅ Verification
Before inclusion in benchmark runs, every subject MUST pass the automated verification suite:
```bash
# From the orchestrator directory
./venv/bin/mgr verify --apps [subject-id]
```
The suite validates HTTP/HTTPS connectivity, Gzip headers, and content parity.
