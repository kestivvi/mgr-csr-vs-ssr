FROM mgr-base-node-builder AS builder
WORKDIR /app

COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

COPY . .
RUN npm run build

FROM mgr-base-node-runner AS runner

COPY --from=builder --chown=svelteuser:nodejs /app/package.json /app/package-lock.json ./
COPY --from=builder --chown=svelteuser:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=svelteuser:nodejs /app/build ./build

CMD ["node", "build"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/build/client/ /usr/share/nginx/html/
