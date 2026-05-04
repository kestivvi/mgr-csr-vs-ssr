# Stage 1: Build the application with Bun
FROM oven/bun:1.3.13-slim AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Build the Next.js application
RUN bun --bun run build

# Stage 2: Production image (Runtime)
FROM oven/bun:1.3.13-slim AS runner
WORKDIR /app

# Set production environment and disable telemetry
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    BUN_JSC_forceRAMSize=1610612736 \
    BUN_GARBAGE_COLLECTOR_LEVEL=1 \
    BUN_METRICS_SINK=0

# Copy standalone build and assets
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# The official Bun images come with a non-root 'bun' user.
USER bun

EXPOSE 3000

# Start the server using the Bun runtime
CMD ["bun", "--bun", "server.js"]

# Stage 3: Serve the application with Nginx (Static Offloading)
FROM nginx:1.30.0-alpine3.23 AS nginx

# Install openssl, generate certificates, and remove the tool to reduce image size
RUN apk add --no-cache openssl \
    && mkdir -p /etc/nginx/ssl \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/key.pem \
        -out /etc/nginx/ssl/cert.pem \
        -subj "/CN=localhost" \
    && apk del openssl

# Copy configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Tuning main Nginx configuration for high throughput
RUN sed -i 's/worker_processes  1;/worker_processes  auto;/' /etc/nginx/nginx.conf \
    && sed -i 's/worker_connections  1024;/worker_connections  65535;/' /etc/nginx/nginx.conf \
    && sed -i '/events {/a \    multi_accept on;' /etc/nginx/nginx.conf \
    && sed -i '/worker_processes  auto;/a worker_rlimit_nofile 100000;' /etc/nginx/nginx.conf

# Redirect logs to stdout/stderr
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80 443 8060

# Statyki z etapu builder
COPY --from=builder /app/.next/static /usr/share/nginx/html/_next/static
COPY --from=builder /app/public/      /usr/share/nginx/html/

CMD ["nginx", "-g", "daemon off;"]
