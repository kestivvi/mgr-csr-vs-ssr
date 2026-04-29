# Stage 1: Build the Qwik City application
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

# Copy source code and build
COPY . .
RUN npm run build
RUN npm run build.preview

# Stage 2: Run the application
FROM node:25.9.0-alpine3.23 AS runner
WORKDIR /app

# Set production environment
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Create a non-root user for security
RUN addgroup -g 1001 -S nodejs && adduser -S qwikuser -u 1001
USER qwikuser

# Copy what's needed to run preview server
COPY --from=builder --chown=qwikuser:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=qwikuser:nodejs /app/dist ./dist
COPY --from=builder --chown=qwikuser:nodejs /app/server ./server
COPY --from=builder --chown=qwikuser:nodejs /app/src ./src
COPY --from=builder --chown=qwikuser:nodejs /app/package.json ./
COPY --from=builder --chown=qwikuser:nodejs /app/vite.config.ts ./
COPY --from=builder --chown=qwikuser:nodejs /app/tsconfig.json ./

EXPOSE 3000

# Start the Qwik City server using vite preview
CMD ["node", "./node_modules/vite/bin/vite.js", "preview", "--host", "0.0.0.0", "--port", "3000", "--strictPort"]
