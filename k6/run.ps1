docker run --rm -i `
  -v ${PWD}:/home/k6 `
  -e CSR_URL="http://host.docker.internal:8080" `
  -e SSR_URL="http://host.docker.internal:8081" `
  grafana/k6 run /home/k6/script.js 