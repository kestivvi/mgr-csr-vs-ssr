# Stage 1: Build the Nuxt application
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app

# Copy lock files to leverage layer caching
COPY package.json package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

# Copy source code and build
COPY . .
RUN npm run build

# Stage 2: Run the application
FROM node:25.9.0-alpine3.23 AS runner
WORKDIR /app

# Set production environment
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Create a non-root user for security
RUN addgroup -g 1001 -S nodejs && adduser -S nuxtjs -u 1001
USER nuxtjs

# Copy built application (.output directory)
COPY --from=builder --chown=nuxtjs:nodejs /app/.output ./

EXPOSE 3000

# Start the Nuxt Nitro server
CMD ["node", "server/index.mjs"]
