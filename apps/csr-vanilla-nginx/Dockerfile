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

# Final stage: Serve with Nginx
FROM nginx:1.29.1-alpine3.22
COPY --from=builder /app/index.min.html /usr/share/nginx/html/index.html
COPY --from=builder /app/app.min.js /usr/share/nginx/html/app.js
COPY --from=builder /app/styles.min.css /usr/share/nginx/html/styles.css
COPY --from=builder /app/favicon.ico /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN apk add --no-cache openssl \
    && mkdir -p /etc/nginx/ssl \
    && openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/nginx/ssl/key.pem \
        -out /etc/nginx/ssl/cert.pem \
        -subj "/CN=localhost"
EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]
