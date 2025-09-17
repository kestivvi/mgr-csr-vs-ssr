# Dockerfile.prod

# ---- Stage 1: Install Production Dependencies ----
# This stage creates a layer with just the production node_modules.
# It will be cached and reused if package.json doesn't change.
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
# Use --omit=dev which is the modern equivalent of --only=production
RUN npm install --omit=dev


# ---- Stage 2: Build the SvelteKit Application ----
# This stage builds the actual application code.
# It installs all dependencies (including dev) and runs the build script.
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build


# ---- Stage 3: Production Runner ----
# This is the final, lean image that will be deployed.
# It copies the necessary artifacts from the previous stages.
FROM node:20-alpine AS runner
WORKDIR /app

# Set environment variables for the Node.js server
ENV NODE_ENV=production
# The adapter-node will listen on port 3000 by default, or the PORT env var
ENV PORT=3000

# Copy the pre-installed production dependencies from the 'deps' stage
COPY --from=deps /app/package*.json ./
COPY --from=deps /app/node_modules ./node_modules

# Copy the built application from the 'builder' stage.
# For adapter-node, we need both the 'build' and '.svelte-kit' directories.
COPY --from=builder /app/build ./build
COPY --from=builder /app/.svelte-kit ./.svelte-kit

# Expose the port the app will run on
EXPOSE 3000

# The command to start the SvelteKit server
# It runs the output of the 'adapter-node' build
CMD ["node", "build"]