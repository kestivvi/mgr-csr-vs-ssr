FROM mgr-base-node-builder AS builder
WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npm run build

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/out /usr/share/nginx/html
