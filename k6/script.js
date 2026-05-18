import http from 'k6/http';
import { sleep } from 'k6';
import { SharedArray } from 'k6/data';
import exec from 'k6/execution';
import { parseHTML } from 'k6/html';

/**
 * --- Block 1: Options & Configuration (Init Context) ---
 * Pre-parse environment variables for maximum performance in the hot path.
 */
const CONFIG = {
  USE_HTTPS: __ENV.K6_USE_HTTPS === 'true',
  TARGET_URL: __ENV.TARGET_URL || 'http://localhost:8080',
  SERVER_TYPE: __ENV.SERVER_TYPE || 'unknown',
  SCENARIO: __ENV.K6_SCENARIO || 'capacity_test',
  TEST_PATH: __ENV.K6_TEST_PATH || 'dynamic',
  SKIP_ASSETS: __ENV.K6_SKIP_ASSETS === 'true',
  SLIM_METRICS: __ENV.K6_SLIM_METRICS === 'true',
  TIMEOUT: (parseFloat(__ENV.TIMEOUT) || 0.4) * 1000,

  // Backoff params
  BACKOFF_TIMEOUT: parseFloat(__ENV.K6_BACKOFF_TIMEOUT_S) || 0.5,
  BACKOFF_5XX: parseFloat(__ENV.K6_BACKOFF_5XX_S) || 0.2,
  JITTER: 0.2, // 20% jitter range

  // Load params
  RATE: parseInt(__ENV.K6_RATE || 100),
  DURATION: __ENV.SCRIPT_DURATION || '5m',
  MAX_VUS: parseInt(__ENV.MAX_VUS || 200),

  // Capacity Test Params
  CAPACITY_START_RATE: parseInt(__ENV.CAPACITY_START_RATE || 10),
  CAPACITY_PEAK_RATE: parseInt(__ENV.CAPACITY_PEAK_RATE || 10000),
  CAPACITY_PEAK_RATE_2: parseInt(__ENV.CAPACITY_PEAK_RATE_2 || 0),
  CAPACITY_WARMUP_DURATION: __ENV.CAPACITY_WARMUP_DURATION || '0s',
  CAPACITY_RAMP_UP_DURATION: __ENV.CAPACITY_RAMP_UP_DURATION || '10m',
  CAPACITY_RAMP_UP_DURATION_2: __ENV.CAPACITY_RAMP_UP_DURATION_2 || '0s',
  CAPACITY_SUSTAIN_DURATION: __ENV.CAPACITY_SUSTAIN_DURATION || '5m',
  CAPACITY_RAMP_DOWN_DURATION: __ENV.CAPACITY_RAMP_DOWN_DURATION || '1m',
};

// --- Init-Context Constants (avoid property lookups in default fn) ---
const IS_DYNAMIC = CONFIG.TEST_PATH !== 'static';
const SKIP_ASSETS = CONFIG.SKIP_ASSETS;
const BACKOFF_TIMEOUT = CONFIG.BACKOFF_TIMEOUT;
const BACKOFF_5XX = CONFIG.BACKOFF_5XX;
const JITTER = CONFIG.JITTER;

// Normalize Target URL
let baseUrl = CONFIG.TARGET_URL;
if (CONFIG.USE_HTTPS) {
  baseUrl = baseUrl.replace(/^http:\/\//, 'https://').replace(/:80(?=\/|$)/, ':443');
} else {
  baseUrl = baseUrl.replace(/^https:\/\//, 'http://').replace(/:443(?=\/|$)/, ':80');
}

const testId = new Date().toISOString().slice(0, 19).replace('T', ' ');

/**
 * --- Prime Generation for Stride-Based URL Selection ---
 * Each VU walks the URL pool with its own prime stride, producing a
 * VU-specific permutation that defeats request-URL caching while staying
 * one cheap addition + bitmask per iteration in the hot path.
 */
function getPrimes(n) {
  const primes = [];
  let i = 2;
  while (primes.length < n) {
    let isPrime = true;
    for (let j = 2; j <= Math.sqrt(i); j++) {
      if (i % j === 0) { isPrime = false; break; }
    }
    if (isPrime) primes.push(i);
    i++;
  }
  return primes;
}
const VU_PRIMES = new SharedArray('vu_primes', () => getPrimes(CONFIG.MAX_VUS));

// --- Capacity Stages Construction ---
const capacityStages = [
  { target: CONFIG.CAPACITY_START_RATE, duration: CONFIG.CAPACITY_WARMUP_DURATION },
  { target: CONFIG.CAPACITY_PEAK_RATE, duration: CONFIG.CAPACITY_RAMP_UP_DURATION },
];

if (CONFIG.CAPACITY_PEAK_RATE_2 > 0) {
  capacityStages.push({ target: CONFIG.CAPACITY_PEAK_RATE_2, duration: CONFIG.CAPACITY_RAMP_UP_DURATION_2 });
}

const finalPeakRate = CONFIG.CAPACITY_PEAK_RATE_2 > 0 ? CONFIG.CAPACITY_PEAK_RATE_2 : CONFIG.CAPACITY_PEAK_RATE;
capacityStages.push({ target: finalPeakRate, duration: CONFIG.CAPACITY_SUSTAIN_DURATION });
capacityStages.push({ target: 0, duration: CONFIG.CAPACITY_RAMP_DOWN_DURATION });

// --- Scenario Definitions ---
const scenarios = {
  capacity_test: {
    executor: 'ramping-arrival-rate',
    startRate: CONFIG.CAPACITY_START_RATE,
    timeUnit: '1s',
    preAllocatedVUs: CONFIG.MAX_VUS,
    maxVUs: CONFIG.MAX_VUS,
    stages: capacityStages,
    gracefulStop: '5s',
  },
  load_test: {
    executor: 'constant-arrival-rate',
    rate: CONFIG.RATE,
    timeUnit: '1s',
    duration: CONFIG.DURATION,
    preAllocatedVUs: CONFIG.MAX_VUS,
    maxVUs: CONFIG.MAX_VUS,
    gracefulStop: '5s',
  },
};

// systemTags drive metric label cardinality. Slim mode keeps only what
// `mgr analyze` queries (status, error_code). Default mode adds `error`
// for the live Grafana error-rate panels.
const SYSTEM_TAGS = CONFIG.SLIM_METRICS
  ? ['status', 'error_code']
  : ['status', 'error_code', 'error'];

export const options = {
  tags: { testid: testId, server: CONFIG.SERVER_TYPE },
  discardResponseBodies: true,
  insecureSkipTLSVerify: CONFIG.USE_HTTPS,
  scenarios: { [CONFIG.SCENARIO]: scenarios[CONFIG.SCENARIO] || scenarios.capacity_test },
  systemTags: SYSTEM_TAGS,
};


// --- Pre-allocated Request Objects (Init Context) ---
const STATIC_URL = baseUrl + '/';
const DYNAMIC_URL_BASE = baseUrl + '/' + CONFIG.TEST_PATH + '/';

// --- URL Pool for Dynamic Paths (shared across all VUs) ---
// Power-of-two size so the hot path can use bitmask (& POOL_MASK) instead
// of modulo, which is ~10-20× cheaper on ARM.
const POOL_SIZE = 16384;
const POOL_MASK = POOL_SIZE - 1;
const DYNAMIC_URL_POOL = new SharedArray('url_pool', function () {
  const pool = new Array(POOL_SIZE);
  for (let i = 0; i < POOL_SIZE; i++) {
    pool[i] = `${DYNAMIC_URL_BASE}${100000 + i}`;
  }
  return pool;
});

const STATIC_PARAMS = {
  timeout: CONFIG.TIMEOUT,
  tags: { resource_type: 'html', name: '/' },
};

const DYNAMIC_PARAMS = {
  timeout: CONFIG.TIMEOUT,
  tags: { resource_type: 'html', name: '/dynamic/:id' },
};

const ASSET_PARAMS = {
  timeout: CONFIG.TIMEOUT,
  tags: { resource_type: 'asset', name: 'static_asset' },
};

/**
 * --- Block 3: Setup (Asset Discovery) ---
 * Returns batch tuples already in the [method, url, body, params] shape
 * expected by http.batch(), so VUs can assign them directly without a
 * per-VU .map() allocation.
 */
export function setup() {
  if (SKIP_ASSETS) return { assetBatchReqs: [] };

  const res = http.get(STATIC_URL, { responseType: 'text' });
  if (res.status !== 200) {
    console.error(`[setup] Failed to discover assets (Status: ${res.status})`);
    return { assetBatchReqs: [] };
  }

  const doc = parseHTML(res.body);
  const assets = new Set();

  doc.find('link[href], script[src], img[src]').each((i, el) => {
    let path = null;
    if (el.attributes) {
      path = el.attributes.href || el.attributes.src;
    }
    if (!path && typeof el.getAttribute === 'function') {
      path = el.getAttribute('href') || el.getAttribute('src');
    }

    if (path && !path.startsWith('data:') && !path.startsWith('#')) {
      let fullUrl;
      if (path.startsWith('http')) {
        fullUrl = path;
      } else if (path.startsWith('//')) {
        fullUrl = (baseUrl.startsWith('https') ? 'https:' : 'http:') + path;
      } else if (path.startsWith('/')) {
        const originMatch = baseUrl.match(/^(https?:\/\/[^\/]+)/);
        fullUrl = (originMatch ? originMatch[1] : baseUrl) + path;
      } else {
        fullUrl = baseUrl + (baseUrl.endsWith('/') ? '' : '/') + path;
      }
      assets.add(fullUrl);
    }
  });

  const assetBatchReqs = [];
  assets.forEach((url) => assetBatchReqs.push(['GET', url, null, ASSET_PARAMS]));
  return { assetBatchReqs };
}

/**
 * --- Block 4: Default Function ---
 * Per-VU state initialized lazily on first iteration.
 */
let myStride = 0;
let myIdx = 0;
let batchReqs = null;

export default function (data) {
  // Per-VU init (runs once per VU on its first iteration)
  if (myStride === 0) {
    myStride = VU_PRIMES[(exec.vu.idInTest - 1) % CONFIG.MAX_VUS];
    myIdx = (exec.vu.idInTest * 7919) & POOL_MASK;

    if (!SKIP_ASSETS && data.assetBatchReqs.length > 0) {
      batchReqs = data.assetBatchReqs;
    }
  }

  // Advance index by VU-specific prime stride; bitmask wraps cheaply.
  myIdx = (myIdx + myStride) & POOL_MASK;

  const res = IS_DYNAMIC
    ? http.get(DYNAMIC_URL_POOL[myIdx], DYNAMIC_PARAMS)
    : http.get(STATIC_URL, STATIC_PARAMS);

  // Conditional Asset Fetching (Nice Client Pattern)
  if (res.status === 200) {
    if (batchReqs !== null) {
      http.batch(batchReqs);
    }
  } else if (res.status === 0 || res.status >= 500) {
    const baseBackoff = res.status === 0 ? BACKOFF_TIMEOUT : BACKOFF_5XX;
    if (baseBackoff > 0) {
      sleep(baseBackoff * (1 + (Math.random() * JITTER * 2 - JITTER)));
    }
  }
}

/**
 * --- Block 5: Teardown ---
 * Dropped-iteration count is reported in k6's default end-of-test summary
 * (look for `dropped_iterations` line). No need to mirror it here.
 */
export function teardown(data) {
  console.log(`[teardown] Test completed. Scenario: ${CONFIG.SCENARIO}`);
}

