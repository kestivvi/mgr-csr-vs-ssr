# Build stage: Minify assets
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app
COPY . .

# Install minification tools
RUN npm install -g terser clean-css-cli html-minifier-terser

# Minify JavaScript
RUN terser app.js -o app.min.js -c -m

# Minify CSS
RUN cleancss -o styles.min.css styles.css

# Minify HTML
RUN html-minifier-terser -o index.min.html index.html --remove-comments --collapse-whitespace --remove-redundant-attributes --remove-script-type-attributes --remove-style-link-type-attributes --use-short-doctype --minify-css --minify-js

# Final stage: Serve with Nginx
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
COPY --from=builder /app/index.min.html /usr/share/nginx/html/index.html
COPY --from=builder /app/app.min.js /usr/share/nginx/html/app.js
COPY --from=builder /app/styles.min.css /usr/share/nginx/html/styles.css
COPY --from=builder /app/favicon.ico /usr/share/nginx/html/
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
