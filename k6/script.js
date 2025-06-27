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

const testId = new Date().toISOString().slice(0, 19).replace('T', ' ');

export const options = {
  ext: {
    loadimpact: {
      projectID: 3680373,
      // Test runs with the same name groups test runs together
      name: 'MGR_REPO',
    },
  },
  tags: {
    testid: testId,
  },
  discardResponseBodies: true,
  scenarios: {
    stress_test: {
      executor: 'ramping-arrival-rate',
      // Start at 10 requests per second
      startRate: 10,
      // The unit of time for the startRate and target
      timeUnit: '1s',
      // Pre-allocate VUs to avoid delays during ramp-up
      preAllocatedVUs: 100,
      // Set a high maximum number of VUs to handle the target rate
      maxVUs: 10000,
      // Define the stages for ramping the request rate
      stages: [
        { target: 10, duration: '1m' }, // maintain 10 RPS for 1 minute
        { target: 10000, duration: '20m' }, // ramp to 10000 RPS over 1 minute
        { target: 0, duration: '1m' },   // ramp down to 0
      ],
    },
  },

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
}