FROM mgr-base-bun-builder AS builder
WORKDIR /app

COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

COPY . .
RUN bun run --bun build

FROM mgr-base-bun-runner AS runner

COPY --from=builder --chown=bun:bun /app/package.json ./
COPY --from=builder --chown=bun:bun /app/node_modules ./node_modules
COPY --from=builder --chown=bun:bun /app/build ./build

CMD ["bun", "--bun", "build/index.js"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/build/client/ /usr/share/nginx/html/
