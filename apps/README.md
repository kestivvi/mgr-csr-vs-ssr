# Applications Overview (MGR Project)

This directory contains the various framework implementations for the performance benchmarking project. All applications have been updated to the latest available versions as of **April 2026**.

## 🛠 Standardized Environment (2026)

To ensure consistency and reproducibility across all benchmarks, the following standards are enforced:

### 1. Package Management
- **Unified Manager**: All Node-based applications use **npm**.
- **Strict Versioning**: All `package.json` files use **exact versions** (no `^` or `~` prefixes).
- **Lockfiles**: Every app contains a `package-lock.json` generated from a clean install.

### 2. Docker Base Images
All applications use strict, verified Alpine-based tags for 2026:

| Component | Image Tag |
| :--- | :--- |
| **Node.js** | `25.9.0-alpine3.23` |
| **NGINX** | `1.30.0-alpine3.23` |
| **Apache (httpd)** | `2.4.66-alpine3.23` |
| **Bun** | `1.3.13-alpine` |
| **Deno** | `alpine-2.7.13` |

## 📁 Application List

The project includes **20 implementations** categorized by rendering strategy:

### SSR (Server-Side Rendering)
- `ssr-tanstack-start-react`
- `ssr-tanstack-start-solid`
- `ssr-nextjs`
- `ssr-nextjs-bun`
- `ssr-nuxtjs`
- `ssr-qwik-city`
- `ssr-react-router`
- `ssr-solid-start`
- `ssr-svelte-kit`
- `ssr-svelte-kit-bun`
- `ssr-angular`
- `ssr-analogjs`

### CSR (Client-Side Rendering)
- `csr-react`
- `csr-vue`
- `csr-solidjs`
- `csr-angular`
- `csr-svelte-kit-static`
- `csr-vanilla-nginx`
- `csr-vanilla-apache`
- `csr-solidjs-apache`

## 🚀 Build and Run

To rebuild and start any application:

```bash
cd apps/[app-name]
docker-compose build --no-cache && docker-compose up --force-recreate
```

## 📝 Maintenance Note
When adding new apps or updating existing ones, ensure that `package.json` versions are stripped of range prefixes and Dockerfiles are updated to match the versions listed above.
