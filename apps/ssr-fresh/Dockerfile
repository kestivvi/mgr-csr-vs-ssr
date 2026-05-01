# Stage 1: Build the Fresh application
FROM denoland/deno:alpine-2.7.13 AS builder
WORKDIR /app

# Copy project files
COPY . .

# Build the Fresh app
RUN deno task build

# Stage 2: Run the application (Production Runtime)
FROM denoland/deno:alpine-2.7.13 AS runner
WORKDIR /app

# Set up non-root user and directories
RUN chown deno:deno /app

# Switch to non-root user
USER deno

# Copy built application from builder
COPY --from=builder --chown=deno:deno /app ./

# Expose Fresh default port
EXPOSE 8000

# Set environment variables
ENV PORT=8000 \
    DENO_DEPLOYMENT_ID=production

# Run the built server
CMD ["deno", "serve", "-A", "_fresh/server.js"]

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

# Statyki Fresh z etapu builder
COPY --from=builder /app/_fresh/static/ /usr/share/nginx/html/_fresh/static/

CMD ["nginx", "-g", "daemon off;"]
