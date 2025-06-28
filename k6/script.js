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
        { target: 10, duration: '2m' }, // maintain 10 RPS for 1 minute
        { target: 10000, duration: '20m' }, // ramp up to 10000 RPS over 10 minutes
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
  group('Main page', function () {
    const res = http.get(`${target_server.url}/`, { timeout: 500 });
    check(res, {
      [`${target_server.type}: status is 200`]: (r) => r.status === 200,
    });
  });

  group('Static assets', function () {
    let assets = [];
    if (target_server.type === 'csr') {
      assets = ['/assets/index-B2WQNOJE.js', '/favicon.ico'];
    } else if (target_server.type === 'ssr') {
      assets = [
        '/_next/static/chunks/webpack-5adebf9f62dc3001.js',
        '/_next/static/chunks/4bd1b696-67ee12fb04071d3b.js',
        '/_next/static/chunks/684-fa9a024d07420a1a.js',
        '/_next/static/chunks/main-app-f38f0d9153b95312.js',
        '/favicon.ico',
      ];
    }

    if (assets.length > 0) {
      const responses = http.batch(
        assets.map((asset) => {
          return {
            method: 'GET',
            url: `${target_server.url}${asset}`,
            params: { timeout: 500 },
          };
        })
      );
      responses.forEach((res) => {
        check(res, {
          'Asset downloaded successfully': (r) => r.status === 200,
        });
      });
    }
  });
}
