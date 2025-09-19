# Stage 1: Build the SolidJS application
FROM node:22-alpine3.22 AS builder
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install
COPY . .
RUN pnpm run build

# Stage 2: Serve the application with Apache
FROM httpd:2.4.65-alpine3.22

# Install Apache Exporter for Prometheus monitoring
RUN apk add --no-cache wget && \
    wget -O /tmp/apache_exporter.tar.gz https://github.com/Lusitaniae/apache_exporter/releases/download/v0.11.0/apache_exporter-0.11.0.linux-amd64.tar.gz && \
    tar -xzf /tmp/apache_exporter.tar.gz -C /tmp && \
    mv /tmp/apache_exporter-0.11.0.linux-amd64/apache_exporter /usr/local/bin/ && \
    chmod +x /usr/local/bin/apache_exporter && \
    rm -rf /tmp/apache_exporter*

# Copy built application
COPY --from=builder /app/dist /var/www/html

# Copy Apache configuration
COPY apache.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf

# Enable required modules and virtual hosts
RUN echo "Include conf/extra/httpd-vhosts.conf" >> /usr/local/apache2/conf/httpd.conf && \
    echo "LoadModule rewrite_module modules/mod_rewrite.so" >> /usr/local/apache2/conf/httpd.conf && \
    echo "LoadModule status_module modules/mod_status.so" >> /usr/local/apache2/conf/httpd.conf

# Create log directories
RUN mkdir -p /var/log/apache2

# Expose ports
EXPOSE 80 8060 9117

# Create startup script
RUN echo '#!/bin/sh' > /start.sh && \
    echo 'apache_exporter --scrape_uri=http://localhost:8060/server-status?auto --telemetry.address=:9117 &' >> /start.sh && \
    echo 'httpd-foreground' >> /start.sh && \
    chmod +x /start.sh

CMD ["/start.sh"]
