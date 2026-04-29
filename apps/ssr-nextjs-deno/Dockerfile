# Stage 1: Build the application using Node.js
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

# Copy source code and build
COPY . .
RUN npm run build

# Stage 2: Production image using Deno
FROM denoland/deno:alpine-2.7.13 AS runner
WORKDIR /app

# Set production environment and disable telemetry
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Ensure the working directory is owned by the deno user
RUN chown deno:deno /app

# Copy standalone build and assets with correct ownership
# The standalone folder contains everything needed to run the server, including node_modules
COPY --from=builder --chown=deno:deno /app/.next/standalone ./
COPY --from=builder --chown=deno:deno /app/.next/static ./.next/static
COPY --from=builder --chown=deno:deno /app/public ./public

# Switch to non-root user
USER deno

EXPOSE 3000

# Start the server using Deno's Node.js compatibility layer
# We use --node-modules-dir=auto to use the existing node_modules from the standalone build
CMD ["deno", "run", "-A", "--node-modules-dir=auto", "--unstable-detect-cjs", "server.js"]