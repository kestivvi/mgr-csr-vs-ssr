# Stage 1: Build the React Router application
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
RUN addgroup -g 1001 -S nodejs && adduser -S reactrouter -u 1001
USER reactrouter

# Copy built application and dependencies
COPY --from=builder --chown=reactrouter:nodejs /app/package.json /app/package-lock.json ./
COPY --from=builder --chown=reactrouter:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=reactrouter:nodejs /app/build ./build

EXPOSE 3000

# Start the React Router server
CMD ["npm", "run", "start"]