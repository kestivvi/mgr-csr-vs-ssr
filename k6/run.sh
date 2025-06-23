#!/bin/bash
docker run --rm -i \
  -v .:/home/k6 \
  -e CSR_URL=$1 \
  -e SSR_URL=$2 \
  -e K6_PROMETHEUS_RW_SERVER_URL=$3 \
  grafana/k6 run --out experimental-prometheus-rw /home/k6/script.js