FROM mgr-base-node-builder AS builder
WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npm run build
RUN npm run build.preview

FROM mgr-base-node-runner AS runner

COPY --from=builder --chown=qwikuser:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=qwikuser:nodejs /app/dist ./dist
COPY --from=builder --chown=qwikuser:nodejs /app/server ./server
COPY --from=builder --chown=qwikuser:nodejs /app/src ./src
COPY --from=builder --chown=qwikuser:nodejs /app/package.json ./
COPY --from=builder --chown=qwikuser:nodejs /app/vite.config.ts ./
COPY --from=builder --chown=qwikuser:nodejs /app/tsconfig.json ./

CMD ["node", "./node_modules/vite/bin/vite.js", "preview", "--host", "0.0.0.0", "--port", "3000", "--strictPort"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/dist/ /usr/share/nginx/html/
