# Stage 1: Install dependencies
FROM node:25.9.0-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

# Stage 2: Build the application
FROM node:25.9.0-slim AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build
RUN npm run build.preview

# Stage 3: Production image
FROM node:25.9.0-slim AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copy what's needed to run preview server
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/server ./server
COPY --from=builder /app/src ./src
COPY --from=builder /app/package.json ./
COPY --from=builder /app/vite.config.ts ./
COPY --from=builder /app/tsconfig.json ./

ENV PORT=3000

EXPOSE 3000

# Start the Qwik City server using vite preview
CMD ["node", "./node_modules/vite/bin/vite.js", "preview", "--host", "0.0.0.0", "--port", "3000", "--strictPort"]
