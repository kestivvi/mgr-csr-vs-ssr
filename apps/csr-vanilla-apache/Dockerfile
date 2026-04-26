# Build stage: Minify assets
FROM node:25.9.0-alpine3.23 AS builder
WORKDIR /app
COPY . .

# Install minification tools
RUN npm install -g terser clean-css-cli html-minifier-terser

# Minify JavaScript
RUN terser app.js -o app.min.js -c -m

# Minify CSS
RUN cleancss -o styles.min.css styles.css

# Minify HTML
RUN html-minifier-terser -o index.min.html index.html --remove-comments --collapse-whitespace --remove-redundant-attributes --remove-script-type-attributes --remove-style-link-type-attributes --use-short-doctype --minify-css --minify-js

# Final stage: Serve with Apache
FROM httpd:2.4.66-alpine3.23

# Install Apache Exporter for Prometheus monitoring
RUN apk add --no-cache wget && \
    wget -O /tmp/apache_exporter.tar.gz https://github.com/Lusitaniae/apache_exporter/releases/download/v0.11.0/apache_exporter-0.11.0.linux-amd64.tar.gz && \
    tar -xzf /tmp/apache_exporter.tar.gz -C /tmp && \
    mv /tmp/apache_exporter-0.11.0.linux-amd64/apache_exporter /usr/local/bin/ && \
    chmod +x /usr/local/bin/apache_exporter && \
    rm -rf /tmp/apache_exporter*

COPY --from=builder /app/index.min.html /usr/local/apache2/htdocs/index.html
COPY --from=builder /app/app.min.js /usr/local/apache2/htdocs/app.js
COPY --from=builder /app/styles.min.css /usr/local/apache2/htdocs/styles.css
COPY --from=builder /app/favicon.ico /usr/local/apache2/htdocs/

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
