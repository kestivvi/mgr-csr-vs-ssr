# Dockerfile.prod.bun.explicit

# ---- Stage 1: Install Production Dependencies ----
# This stage creates a layer with just the production node_modules.
# It will be cached and reused if package.json or bun.lockb doesn't change.
FROM oven/bun:1-alpine AS deps
WORKDIR /app
COPY package.json bun.lockb* ./
# Install production-only dependencies
# --bun is not applicable to 'bun install'
RUN bun install --production


# ---- Stage 2: Build the SvelteKit Application ----
# This stage builds the actual application code.
# It installs all dependencies (including dev) and runs the build script.
FROM oven/bun:1-alpine AS builder
WORKDIR /app
COPY package.json bun.lockb* ./
# Install all dependencies
RUN bun install
COPY . .
# Build the SvelteKit application, explicitly using the bun runtime
RUN bun run --bun build


# ---- Stage 3: Production Runner ----
# This is the final, lean image that will be deployed.
# It copies the necessary artifacts from the previous stages.
FROM oven/bun:1-alpine AS runner
WORKDIR /app

# Set environment variables for the server
ENV NODE_ENV=production
# The adapter-node will listen on port 3000 by default, or the PORT env var
ENV PORT=3000

# Copy the pre-installed production dependencies from the 'deps' stage
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/package.json ./

# Copy the built application from the 'builder' stage.
COPY --from=builder /app/build ./build

# Expose the port the app will run on
EXPOSE 3000

# The command to start the SvelteKit server
# Explicitly use the --bun flag to run the server entrypoint
CMD ["bun", "--bun", "build/index.js"]