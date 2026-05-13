FROM mgr-base-node-builder AS builder
WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npx astro telemetry disable
RUN npm run build

FROM mgr-base-node-runner AS runner

COPY --from=builder --chown=mgruser /app/dist ./dist
COPY --from=builder --chown=mgruser /app/node_modules ./node_modules
COPY --from=builder --chown=mgruser /app/package.json ./package.json

CMD ["node", "./dist/server/entry.mjs"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/dist/client/ /usr/share/nginx/html/
