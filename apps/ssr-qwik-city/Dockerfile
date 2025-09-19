# Stage 1: Install dependencies
FROM node:22-alpine3.22 AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --only=production

# Stage 2: Build the application
FROM node:22-alpine3.22 AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build
RUN npm run build.preview

# Stage 3: Production image
FROM node:22-alpine3.22 AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT 3000

COPY --from=builder /app/package.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/server ./server
COPY --from=builder /app/src ./src
COPY --from=builder /app/vite.config.ts ./
COPY --from=builder /app/tsconfig.json ./

EXPOSE 3000

CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "3000"]
