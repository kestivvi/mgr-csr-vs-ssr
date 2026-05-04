# Stage 1: Build the application
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

# Copy source code and build
COPY . .
RUN npm run build

# Copy global.css to static directory for Nginx to serve
RUN mkdir -p dist/static && cp global.css dist/static/global.css

# Stage 2: Run the application (Production Runtime)
FROM node:25.9.0-alpine3.23 AS runner
WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000

# Create a non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S litssr -u 1001
USER litssr

# Copy built application and dependencies
COPY --from=builder --chown=litssr:nodejs /app/package.json /app/package-lock.json ./
COPY --from=builder --chown=litssr:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=litssr:nodejs /app/dist ./dist

EXPOSE 3000
CMD ["node", "dist/server.js"]

# Stage 3: Serve static assets with Nginx
FROM nginx:1.30.0-alpine3.23 AS nginx

# Install openssl, generate certificates
RUN apk add --no-cache openssl \
    && mkdir -p /etc/nginx/ssl \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/key.pem \
        -out /etc/nginx/ssl/cert.pem \
        -subj "/CN=localhost" \
    && apk del openssl

# Copy configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Tuning Nginx
RUN sed -i 's/worker_processes  1;/worker_processes  auto;/' /etc/nginx/nginx.conf \
    && sed -i 's/worker_connections  1024;/worker_connections  65535;/' /etc/nginx/nginx.conf \
    && sed -i '/events {/a \    multi_accept on;' /etc/nginx/nginx.conf \
    && sed -i '/worker_processes  auto;/a worker_rlimit_nofile 100000;' /etc/nginx/nginx.conf

# Redirect logs
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80 443 8060

# Copy static assets and public files from builder
COPY --from=builder /app/dist/static /usr/share/nginx/html/static
COPY --from=builder /app/public /usr/share/nginx/html

CMD ["nginx", "-g", "daemon off;"]
