FROM denoland/deno:latest

# Set up work directory
WORKDIR /app

# Copy project files
COPY . .

# Build the Fresh app (produces _fresh/server.js)
RUN deno task build

# Cache the built server for faster startup
RUN deno cache _fresh/server.js

# Expose Fresh default port
EXPOSE 8000

# Allow setting deployment id for Fresh caching
ARG GIT_REVISION
ENV DENO_DEPLOYMENT_ID=${GIT_REVISION}

# Run the built server
CMD ["deno", "serve", "-A", "_fresh/server.js"]


