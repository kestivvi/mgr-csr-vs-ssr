FROM mgr-base-bun-builder AS builder
WORKDIR /app

COPY package.json bun.lock* package-lock.json* ./
RUN bun install

COPY . .
RUN bun run build
RUN bun run build.preview

FROM mgr-base-bun-runner AS runner

COPY --from=builder --chown=bun:bun /app/node_modules ./node_modules
COPY --from=builder --chown=bun:bun /app/dist ./dist
COPY --from=builder --chown=bun:bun /app/server ./server
COPY --from=builder --chown=bun:bun /app/src ./src
COPY --from=builder --chown=bun:bun /app/package.json ./
COPY --from=builder --chown=bun:bun /app/vite.config.ts ./
COPY --from=builder --chown=bun:bun /app/tsconfig.json ./

CMD ["bun", "--bun", "./node_modules/vite/bin/vite.js", "preview", "--host", "0.0.0.0", "--port", "3000", "--strictPort"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/dist/ /usr/share/nginx/html/
