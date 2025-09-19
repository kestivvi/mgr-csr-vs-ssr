# Stage 1: Install dependencies (UNCHANGED)
FROM node:22-alpine3.22 AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install

# Stage 2: Build the application (UNCHANGED)
FROM node:22-alpine3.22 AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Stage 3: Production image (REFACTORED FOR DENO)
FROM denoland/deno:alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT 3000

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

# Start the server using Deno with the CJS detection flag
CMD ["deno", "run", "--allow-net", "--allow-read", "--allow-env", "--allow-sys", "--node-modules-dir", "--unstable-detect-cjs", "server.js"]