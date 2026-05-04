# Stage 1: Build the SvelteKit application with Bun
FROM oven/bun:1.3.13-slim AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

# Copy source code and build
COPY . .
RUN bun run --bun build

# Stage 2: Run the application (Production Runtime)
FROM oven/bun:1.3.13-slim AS runner
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

# Statyki SvelteKit z etapu builder
COPY --from=builder /app/build/client/ /usr/share/nginx/html/

CMD ["nginx", "-g", "daemon off;"]