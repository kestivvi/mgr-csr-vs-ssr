# Stage 1: Build the SolidJS application
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Serve the application with Apache
FROM httpd:2.4.66-alpine3.23

# Install Apache Exporter for Prometheus monitoring
RUN apk add --no-cache wget && \
    wget -O /tmp/apache_exporter.tar.gz https://github.com/Lusitaniae/apache_exporter/releases/download/v0.11.0/apache_exporter-0.11.0.linux-amd64.tar.gz && \
    tar -xzf /tmp/apache_exporter.tar.gz -C /tmp && \
    mv /tmp/apache_exporter-0.11.0.linux-amd64/apache_exporter /usr/local/bin/ && \
    chmod +x /usr/local/bin/apache_exporter && \
    rm -rf /tmp/apache_exporter*

# Copy built application to standard Apache path
COPY --from=builder /app/dist /usr/local/apache2/htdocs

# Copy Apache configuration
COPY apache.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf

# Enable virtual hosts (modules are now loaded in apache.conf)
RUN echo "Include conf/extra/httpd-vhosts.conf" >> /usr/local/apache2/conf/httpd.conf

# Create log directories and SSL dir, generate self-signed cert
RUN mkdir -p /var/log/apache2 /usr/local/apache2/ssl \
    && apk add --no-cache openssl \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /usr/local/apache2/ssl/key.pem \
        -out /usr/local/apache2/ssl/cert.pem \
        -subj "/CN=localhost"

# Expose ports
EXPOSE 80 443 8060 9117

# Create startup script
RUN echo '#!/bin/sh' > /start.sh && \
    echo 'apache_exporter --scrape_uri=http://localhost:8060/server-status?auto --telemetry.address=:9117 &' >> /start.sh && \
    echo 'httpd-foreground' >> /start.sh && \
    chmod +x /start.sh

CMD ["/start.sh"]
