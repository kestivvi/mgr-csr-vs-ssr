# Stage 1: Install dependencies
FROM node:22-alpine3.22 AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

# Stage 2: Build the application
FROM node:22-alpine3.22 AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# Stage 3: Production image
FROM node:22-alpine3.22 AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copy the built application
COPY --from=builder /app/.output ./

EXPOSE 3000

# Start the SolidStart server
CMD ["node", "server/index.mjs"]
