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

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/index.min.html /usr/share/nginx/html/index.html
COPY --from=builder /app/app.min.js /usr/share/nginx/html/app.js
COPY --from=builder /app/styles.min.css /usr/share/nginx/html/styles.css
COPY --from=builder /app/favicon.ico /usr/share/nginx/html/
