# Stage 1: Build the Fresh application
FROM denoland/deno:alpine-2.7.13 AS builder
WORKDIR /app

# Copy project files
COPY . .

# Build the Fresh app
RUN deno task build

# Stage 2: Run the application
FROM denoland/deno:alpine-2.7.13 AS runner
WORKDIR /app

# Set up non-root user and directories
RUN chown deno:deno /app

# Switch to non-root user
USER deno

# Copy built application from builder
COPY --from=builder --chown=deno:deno /app ./

# Expose Fresh default port
EXPOSE 8000

# Set environment variables
ENV PORT=8000 \
    DENO_DEPLOYMENT_ID=production

# Run the built server
CMD ["deno", "serve", "-A", "_fresh/server.js"]
