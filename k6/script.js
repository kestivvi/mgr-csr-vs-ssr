import http from 'k6/http';
import { check, group, sleep } from 'k6';

const target_server = {
  url: __ENV.TARGET_URL || 'http://localhost:8080',
  type: __ENV.SERVER_TYPE || 'unknown',
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
    server: target_server.type,
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
        { target: 10000, duration: '20m' }, // ramp up to 10000 RPS over 10 minutes
        { target: 10000, duration: '5m' }, // stay at 10000 RPS for 5 minutes
        { target: 0, duration: '1m' },   // ramp down to 0
      ],
    },
  },

  // Commented out for now, because it's a stress test and we want to see errors
  // Thresholds are the pass/fail criteria for the test
  // thresholds: {
  //   http_req_failed: ['rate<0.01'], // less than 1% of requests should fail
  //   http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
  //   checks: ['rate>0.99'], // 99% of checks must pass
  // },
};

export default function () {
  const res = http.get(`${target_server.url}/hello-world`, { timeout: 500 });

  check(res, {
    [`${target_server.type}: status is 200`]: (r) => r.status === 200,
  });
}
