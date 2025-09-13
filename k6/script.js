import http from 'k6/http';
import { check } from 'k6';

// Target server URL to load test
const target_url = __ENV.TARGET_URL || 'http://localhost:8080';

// Server type label for test results (e.g., 'SSR', 'CSR')
const server_type = __ENV.SERVER_TYPE || 'unknown';

// Requests per second for constant load test
const K6_RPS = __ENV.K6_RPS || 100;

// Duration for constant load test (e.g., '5m', '10m', '1h')
const K6_DURATION = __ENV.K6_DURATION || '5m';

// Test scenario to run: 'stress_test' or 'constant_test'
const K6_SCENARIO = __ENV.K6_SCENARIO || 'stress_test';

// Test path: 'static' for "/" or 'dynamic' for "/dynamic/{incrementing-number}"
const TEST_PATH = __ENV.TEST_PATH || 'static';

// Request timeout in seconds (converted to milliseconds)
const TIMEOUT = (__ENV.TIMEOUT || 0.2) * 1000;

const testId = new Date().toISOString().slice(0, 19).replace('T', ' ');

const stressTestScenario = {
  executor: 'ramping-arrival-rate',
  startRate: 10,
  timeUnit: '1s',
  preAllocatedVUs: 200,
  maxVUs: 200,
  stages: [
    { target: 10, duration: '1m' },
    { target: 6000, duration: '10m' },
    { target: 6000, duration: '10m' },
    { target: 0, duration: '1m' },
  ],
};

const constantTestScenario = {
  executor: 'constant-arrival-rate',
  rate: K6_RPS,
  timeUnit: '1s',
  duration: K6_DURATION,
  preAllocatedVUs: 200,
  maxVUs: 200,
};

const selectedScenario = K6_SCENARIO === 'constant_test' 
  ? constantTestScenario 
  : stressTestScenario;

export const options = {
  ext: {
    loadimpact: {
      projectID: 3680373,
      name: 'MGR_REPO',
    },
  },
  tags: {
    testid: testId,
    server: server_type,
  },
  discardResponseBodies: true,
  scenarios: {
    [K6_SCENARIO]: selectedScenario,
  },
};


export default function () {
  const testPath = TEST_PATH === 'dynamic' 
    ? `/dynamic/${Math.floor(Math.random() * 1000000) + 1}`
    : '/';
  
  const res = http.get(`${target_url}${testPath}`, { timeout: TIMEOUT });
  check(res, {
    [`${server_type}: status is 200`]: (r) => r.status === 200,
  });
}
