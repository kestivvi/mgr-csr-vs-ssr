FROM mgr-base-node-builder AS builder
WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npm run build

FROM mgr-base-node-runner AS runner

COPY --from=builder --chown=mgruser /app/.next/standalone ./
COPY --from=builder --chown=mgruser /app/.next/static ./.next/static
COPY --from=builder --chown=mgruser /app/public ./public

CMD ["node", "server.js"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/.next/static /usr/share/nginx/html/_next/static
COPY --from=builder /app/public/      /usr/share/nginx/html/
