# Stage 1: Install dependencies (UNCHANGED)
FROM node:25.9.0-alpine3.23 AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# Stage 2: Build the application (UNCHANGED)
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Stage 3: Production image (REFACTORED FOR DENO)
FROM denoland/deno:alpine-2.7.13 AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT 3000

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/package.json ./

# Pre-cache node dependencies so they are not downloaded at runtime
RUN deno install

EXPOSE 3000

# Start the server using Deno with all permissions, auto node_modules, and CJS detection
CMD ["deno", "run", "-A", "--node-modules-dir=auto", "--unstable-detect-cjs", "server.js"]