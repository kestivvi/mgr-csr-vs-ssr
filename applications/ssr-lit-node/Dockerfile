FROM mgr-base-node-builder AS builder
WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npm run build

RUN mkdir -p dist/static && cp global.css dist/static/global.css

FROM mgr-base-node-runner AS runner

COPY --from=builder --chown=mgruser /app/package.json /app/package-lock.json ./
COPY --from=builder --chown=mgruser /app/node_modules ./node_modules
COPY --from=builder --chown=mgruser /app/dist ./dist

CMD ["node", "dist/server.js"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/dist/static /usr/share/nginx/html/static
COPY --from=builder /app/public /usr/share/nginx/html
