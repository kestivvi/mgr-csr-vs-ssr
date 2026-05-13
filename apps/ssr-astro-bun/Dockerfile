FROM mgr-base-bun-builder AS builder
WORKDIR /app

COPY package.json bun.lock* package-lock.json* ./
RUN bun install

COPY . .
RUN bun run build

FROM mgr-base-bun-runner AS runner

COPY --from=builder --chown=bun:bun /app/dist ./dist
COPY --from=builder --chown=bun:bun /app/node_modules ./node_modules
COPY --from=builder --chown=bun:bun /app/package.json ./

CMD ["bun", "--bun", "./dist/server/entry.mjs"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/dist/client/ /usr/share/nginx/html/
