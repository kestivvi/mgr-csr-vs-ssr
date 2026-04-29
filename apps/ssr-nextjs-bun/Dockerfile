# Stage 1: Build the application with Bun
FROM oven/bun:1.3.13-alpine AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Build the Next.js application
RUN bun --bun run build

# Stage 2: Production image
FROM oven/bun:1.3.13-alpine AS runner
WORKDIR /app

# Set production environment and disable telemetry
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Copy standalone build and assets
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# The official Bun images come with a non-root 'bun' user.
USER bun

EXPOSE 3000

# Start the server using the Bun runtime
CMD ["bun", "--bun", "server.js"]
