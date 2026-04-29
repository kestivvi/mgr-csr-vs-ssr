# Stage 1: Build the SvelteKit application with Bun
FROM oven/bun:1.3.13-alpine AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

# Copy source code and build
COPY . .
RUN bun run --bun build

# Stage 2: Run the application
FROM oven/bun:1.3.13-alpine AS runner
WORKDIR /app

# Set production environment
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# The official Bun images come with a non-root 'bun' user.
USER bun

# Copy built application and production dependencies
COPY --from=builder /app/package.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/build ./build

EXPOSE 3000

# Start the SvelteKit server using Bun runtime
CMD ["bun", "--bun", "build/index.js"]