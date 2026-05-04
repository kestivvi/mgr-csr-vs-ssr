# Stage 1: Build
FROM denoland/deno:alpine-2.7.13 AS builder
WORKDIR /app
COPY deno.json ./
RUN deno install
COPY . .
# Fresh 2.0 build uses Vite to generate _fresh directory
RUN deno task build

# Stage 2: Runner
FROM denoland/deno:alpine-2.7.13 AS runner
WORKDIR /app
# Pass build arg for deployment ID to ensure unique asset paths
ARG GIT_REVISION=1
ENV DENO_DEPLOYMENT_ID=${GIT_REVISION}
COPY --from=builder /app ./
USER deno
EXPOSE 3000
# Run the built server using Deno's native serve
CMD ["deno", "task", "start"]

# Stage 3: Nginx
FROM nginx:1.30.0-alpine3.23 AS nginx
RUN apk add --no-cache openssl \
    && mkdir -p /etc/nginx/ssl \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/key.pem \
        -out /etc/nginx/ssl/cert.pem \
        -subj "/CN=localhost" \
    && apk del openssl
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN sed -i 's/worker_processes  1;/worker_processes  auto;/' /etc/nginx/nginx.conf \
    && sed -i 's/worker_connections  1024;/worker_connections  65535;/' /etc/nginx/nginx.conf \
    && sed -i '/events {/a \    multi_accept on;' /etc/nginx/nginx.conf \
    && sed -i '/worker_processes  auto;/a worker_rlimit_nofile 100000;' /etc/nginx/nginx.conf
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log
HEALTHCHECK --interval=30s --timeout=3s \
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1
EXPOSE 80 443 8060
# Copy static files to nginx for offloading
COPY --from=builder /app/static/ /usr/share/nginx/html/static/
COPY --from=builder /app/_fresh/client/ /usr/share/nginx/html/_fresh/client/
CMD ["nginx", "-g", "daemon off;"]
