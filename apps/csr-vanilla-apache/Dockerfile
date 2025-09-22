# Build stage: Minify assets
FROM node:22-alpine3.22 AS builder
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
FROM httpd:2.4-alpine3.22
COPY --from=builder /app/index.min.html /usr/local/apache2/htdocs/index.html
COPY --from=builder /app/app.min.js /usr/local/apache2/htdocs/app.js
COPY --from=builder /app/styles.min.css /usr/local/apache2/htdocs/styles.css
COPY --from=builder /app/favicon.ico /usr/local/apache2/htdocs/

# Copy Apache configuration
COPY apache.conf /usr/local/apache2/conf/extra/httpd-vhosts.conf

# Enable required modules and virtual hosts
RUN echo "Include conf/extra/httpd-vhosts.conf" >> /usr/local/apache2/conf/httpd.conf && \
    echo "LoadModule rewrite_module modules/mod_rewrite.so" >> /usr/local/apache2/conf/httpd.conf && \
    echo "LoadModule status_module modules/mod_status.so" >> /usr/local/apache2/conf/httpd.conf

# Create log directories
RUN mkdir -p /var/log/apache2

# Expose ports
EXPOSE 80 8060

CMD ["httpd", "-D", "FOREGROUND"]
