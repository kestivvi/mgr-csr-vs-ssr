# Stage 1: Install dependencies
FROM node:22-alpine3.22 AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

# Stage 2: Build the application
FROM node:22-alpine3.22 AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 3: Production image
FROM node:22-alpine3.22 AS runner
WORKDIR /app

ENV NODE_ENV=production

# COPY --from=builder /app/public ./public
# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

# server.js is created by the build script when output is standalone
CMD ["node", "server.js"] 