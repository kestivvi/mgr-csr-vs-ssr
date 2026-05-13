# Build stage: Minify assets
FROM mgr-base-node-builder AS builder
WORKDIR /app
COPY . .

RUN npm install -g terser clean-css-cli html-minifier-terser

# Minify JavaScript
RUN terser app.js -o app.min.js -c -m

# Minify CSS
RUN cleancss -o styles.min.css styles.css

# Minify HTML
RUN html-minifier-terser -o index.min.html index.html --remove-comments --collapse-whitespace --remove-redundant-attributes --remove-script-type-attributes --remove-style-link-type-attributes --use-short-doctype --minify-css --minify-js

FROM httpd:2.4.66-alpine3.23

# Apache Exporter (build-only deps in same layer, then removed)
RUN apk add --no-cache --virtual .build-deps wget tar \
    && wget -O /tmp/apache_exporter.tar.gz \
        https://github.com/Lusitaniae/apache_exporter/releases/download/v0.11.0/apache_exporter-0.11.0.linux-amd64.tar.gz \
    && tar -xzf /tmp/apache_exporter.tar.gz -C /tmp \
    && mv /tmp/apache_exporter-0.11.0.linux-amd64/apache_exporter /usr/local/bin/ \
    && chmod +x /usr/local/bin/apache_exporter \
    && rm -rf /tmp/apache_exporter* \
    && apk del .build-deps

# Self-signed cert at build time (openssl removed after)
RUN apk add --no-cache --virtual .ssl-deps openssl \
    && mkdir -p /usr/local/apache2/ssl /var/log/apache2 \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /usr/local/apache2/ssl/key.pem \
        -out /usr/local/apache2/ssl/cert.pem \
        -subj "/CN=localhost" \
    && apk del .ssl-deps

# Built app
COPY --from=builder /app/index.min.html /usr/local/apache2/htdocs/index.html
COPY --from=builder /app/app.min.js /usr/local/apache2/htdocs/app.js
COPY --from=builder /app/styles.min.css /usr/local/apache2/htdocs/styles.css
COPY --from=builder /app/favicon.ico /usr/local/apache2/htdocs/

# Configs
COPY apache.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf
COPY mpm.conf    /usr/local/apache2/conf/extra/httpd-mpm-tuning.conf

# Wire them in. Also disable the default prefork MPM line if present and
# load event MPM via mpm.conf. Raise file descriptor limit for the process.
RUN sed -i 's|^LoadModule mpm_.*|# &|' /usr/local/apache2/conf/httpd.conf \
    && echo "Include conf/extra/httpd-mpm-tuning.conf" >> /usr/local/apache2/conf/httpd.conf \
    && echo "Include conf/extra/httpd-vhosts.conf" >> /usr/local/apache2/conf/httpd.conf

EXPOSE 80 443 8060 9117

# Use exec form so signals reach httpd correctly
RUN printf '#!/bin/sh\nset -e\napache_exporter --scrape_uri=http://localhost:8060/server-status?auto --telemetry.address=:9117 &\nexec httpd-foreground\n' > /start.sh \
    && chmod +x /start.sh

CMD ["/start.sh"]
