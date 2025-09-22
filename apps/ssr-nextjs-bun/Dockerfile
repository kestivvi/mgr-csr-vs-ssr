# Stage 1: Build the application with Bun
# Use the official Bun image. The default is based on Debian, which is fine for building.
FROM oven/bun:1 AS builder
WORKDIR /app

# Bun automatically uses a binary lockfile (bun.lockb) for faster installs.
# Using --frozen-lockfile is the best practice for CI/build environments.
COPY package.json bun.lockb* ./
RUN bun install --frozen-lockfile

# Copy the rest of the application source code
COPY . .

# Build the Next.js application using Bun's script runner
RUN bun --bun run build

# Stage 2: Production image
# Use the lightweight Alpine variant of the Bun image for the final stage
FROM oven/bun:1-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copy only the necessary standalone output from the builder stage
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# The official Bun images come with a non-root 'bun' user.
# Using it is a security best practice.
USER bun

EXPOSE 3000

# Start the server using the Bun runtime
CMD ["bun", "--bun", "server.js"]
