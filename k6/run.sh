#!/bin/bash
docker run --rm -i \
  -v .:/home/k6 \
  -e TARGET_URL=$1 \
  -e SERVER_TYPE=$2 \
  -e K6_PROMETHEUS_RW_SERVER_URL=$3 \
  -e K6_PROMETHEUS_RW_TREND_AS_NATIVE_HISTOGRAM=true \
  -e K6_LOG_OUTPUT=none \
  grafana/k6 run --out experimental-prometheus-rw /home/k6/script.js