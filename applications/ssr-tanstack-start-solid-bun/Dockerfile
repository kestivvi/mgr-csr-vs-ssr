FROM mgr-base-bun-builder AS builder
WORKDIR /app

COPY package.json bun.lock* package-lock.json* ./
RUN bun install

COPY . .
RUN bun run build

FROM mgr-base-bun-runner AS runner

COPY --from=builder --chown=bun:bun /app/.output ./

CMD ["bun", "--bun", "server/index.mjs"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/.output/public/ /usr/share/nginx/html/
