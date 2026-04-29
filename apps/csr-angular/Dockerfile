# Stage 1: Build the Angular application
FROM node:25.9.0-alpine3.23 AS builder

WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

# Copy the rest of the code and build the application
COPY . .
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:1.30.0-alpine3.23

# Install openssl, generate certificates, and remove the tool to reduce image size
RUN apk add --no-cache openssl \
    && mkdir -p /etc/nginx/ssl \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/key.pem \
        -out /etc/nginx/ssl/cert.pem \
        -subj "/CN=localhost" \
    && apk del openssl

# Copy built files and configuration
COPY --from=builder /app/dist/csr-angular/browser /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Tuning main Nginx configuration for 30k+ RPS
RUN sed -i 's/worker_processes  1;/worker_processes  auto;/' /etc/nginx/nginx.conf \
    && sed -i 's/worker_connections  1024;/worker_connections  65535;/' /etc/nginx/nginx.conf \
    && sed -i '/events {/a \    multi_accept on;' /etc/nginx/nginx.conf \
    && sed -i '/worker_processes  auto;/a worker_rlimit_nofile 100000;' /etc/nginx/nginx.conf

# Redirect logs to stdout/stderr (for better visibility in docker logs)
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# Healthcheck - checks if Nginx responds on port 80
HEALTHCHECK --interval=30s --timeout=3s \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

EXPOSE 80 443 8060

CMD ["nginx", "-g", "daemon off;"]
