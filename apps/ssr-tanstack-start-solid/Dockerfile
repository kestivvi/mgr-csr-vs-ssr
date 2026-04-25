# Stage 1: Install dependencies
FROM node:25.9.0-alpine3.23 AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

# Stage 2: Build the application
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 3: Production image
FROM node:25.9.0-alpine3.23 AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copy the built application from Nitro
COPY --from=builder /app/.output ./

EXPOSE 3000

# Start the TanStack Start server
CMD ["node", "server/index.mjs"]


