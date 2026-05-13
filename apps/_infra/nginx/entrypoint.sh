#!/bin/sh
set -e

# Substitute environment variables in the template
# We use a temp file to avoid overwriting the source if it's the same directory
envsubst '${APP_NAME} ${APP_PORT} ${STATIC_PATH}' < "${NGINX_TEMPLATE:-/etc/nginx/conf.d/default.conf.template}" > /etc/nginx/conf.d/default.conf

# Execute the CMD from the Dockerfile
exec "$@"
