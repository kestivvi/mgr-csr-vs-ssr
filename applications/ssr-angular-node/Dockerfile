FROM mgr-base-node-builder AS builder
WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npm run build

FROM mgr-base-node-runner AS runner

COPY --from=builder --chown=angularjs:nodejs /app/dist/ssr-angular ./

CMD ["node", "server/server.mjs"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/dist/ssr-angular/browser/ /usr/share/nginx/html/
