FROM denoland/deno:alpine-2.7.13 AS builder
WORKDIR /app
COPY deno.json ./
RUN deno install
COPY . .
# Fresh 2.0 build uses Vite to generate _fresh directory
RUN deno task build

FROM denoland/deno:alpine-2.7.13 AS runner
WORKDIR /app
# Pass build arg for deployment ID to ensure unique asset paths
ARG GIT_REVISION=1
ENV DENO_DEPLOYMENT_ID=${GIT_REVISION}
COPY --from=builder /app/deno.json /app/deno.lock ./
COPY --from=builder /app/_fresh ./_fresh
COPY --from=builder /app/static ./static
USER deno
EXPOSE 3000
CMD ["deno", "task", "start"]

FROM mgr-base-nginx AS nginx

COPY --from=builder /app/static/ /usr/share/nginx/html/static/
COPY --from=builder /app/_fresh/client/ /usr/share/nginx/html/_fresh/client/
