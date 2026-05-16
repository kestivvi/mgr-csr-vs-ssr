FROM mgr-base-node-builder AS builder
WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npm run build

FROM mgr-base-node-runner AS runner

COPY --from=builder --chown=tanstackuser:nodejs /app/.output ./

CMD ["node", "server/index.mjs"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/.output/public/ /usr/share/nginx/html/
