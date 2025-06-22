import http from 'k6/http';
import { check, group, sleep } from 'k6';

const csr_server = {
  name: 'CSR Server',
  url: __ENV.CSR_URL || 'http://localhost:8080',
  tag: 'server:csr',
};

const ssr_server = {
  name: 'SSR Server',
  url: __ENV.SSR_URL || 'http://localhost:8081',
  tag: 'server:ssr',
};

export const options = {
  stages: [
    { duration: '30s', target: 50 },
    { duration: '1m', target: 50 },
    { duration: '20s', target: 0 },
  ],

  // Thresholds are the pass/fail criteria for the test
  thresholds: {
    // A global threshold: less than 1% of all requests should fail
    // http_req_failed: ['rate<0.01'],

    // // --- Per-server thresholds using tags ---
    // // 95% of requests to the 'primary' server must be below 300ms
    // 'http_req_duration{server:csr}': ['p(95)<300'],
    // // 99% of checks for the 'primary' server must pass
    // 'checks{server:csr}': ['rate>0.99'],

    // // 95% of requests to the 'secondary' server must be below 500ms
    // 'http_req_duration{server:ssr}': ['p(95)<500'],
    // // 99% of checks for the 'secondary' server must pass
    // 'checks{server:ssr}': ['rate>0.99'],
  },
};

export default function () {

  group(csr_server.name, function () {
    const res = http.get(`${csr_server.url}/hello-world`, {
      tags: { server: 'csr' },
    });

    check(res, { 'status is 200': (r) => r.status === 200 }, { server: 'csr' });
  });

  group(ssr_server.name, function () {
    const res = http.get(`${ssr_server.url}/hello-world`, {
      tags: { server: 'ssr' },
    });

    check(res, { 'status is 200': (r) => r.status === 200 }, { server: 'ssr' });
  });

  sleep(1);
}