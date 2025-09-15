docker run --rm -v $(pwd):/scripts grafana/k6:latest run \
  -e TARGET_URL=http://host.docker.internal:80 \
  -e SERVER_TYPE=manual \
  -e K6_SCENARIO=constant_test \
  -e K6_RPS=1 \
  -e K6_DURATION=1s \
  /scripts/script.js