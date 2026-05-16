# Nuxt (Bun) - SSR Performance Test App

This is a specialized variant of the Nuxt SSR application optimized for the **Bun** runtime, part of the Master's Thesis research portfolio.

## 🚀 Bun Optimization

- **Runtime**: Bun 1.3.13
- **Nitro Preset**: `bun`
- **Container**: `oven/bun:1.3.13-slim`

## 🛠 Commands

```bash
# Install dependencies
bun install

# Development
bun run dev

# Build for production
bun run build

# Start production server
bun run --bun server/index.mjs
```

## 🏗 Infrastructure

This application follows the **Static Offloading Principle**:
- **Bun**: Handles dynamic HTML generation.
- **Nginx**: Acts as a reverse proxy and serves all static assets from `/.output/public/`.
