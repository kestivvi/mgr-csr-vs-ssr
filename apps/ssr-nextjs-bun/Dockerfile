FROM mgr-base-bun-builder AS builder
WORKDIR /app

COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

COPY . .

RUN bun --bun run build

FROM mgr-base-bun-runner AS runner

COPY --from=builder --chown=bun:bun /app/.next/standalone ./
COPY --from=builder --chown=bun:bun /app/.next/static ./.next/static
COPY --from=builder --chown=bun:bun /app/public ./public

CMD ["bun", "--bun", "server.js"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/.next/static /usr/share/nginx/html/_next/static
COPY --from=builder /app/public/      /usr/share/nginx/html/
