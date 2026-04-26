docker run --rm -v $(pwd):/scripts grafana/k6:latest run --insecure-skip-tls-verify \
  -e TARGET_URL=http://host.docker.internal:80 \
  -e SERVER_TYPE=manual \
  -e K6_SCENARIO=load_test \
  -e K6_RATE=1 \
  -e K6_DURATION=1s \
  /scripts/script.js