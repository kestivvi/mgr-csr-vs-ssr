import http from 'k6/http';
import { check, group } from 'k6';
import { parseHTML } from 'k6/html';
import { URL } from 'https://jslib.k6.io/url/1.0.0/index.js';

// --- K6 Test Configuration (Init Context) ---
// This code runs once per VU before the test starts. Keep it minimal.
const target_url = __ENV.TARGET_URL || 'http://localhost:8080';
const server_type = __ENV.SERVER_TYPE || 'unknown';
const K6_SCENARIO = __ENV.K6_SCENARIO || 'stress_test';
const TEST_PATH = __ENV.TEST_PATH || 'static';
const TIMEOUT = (parseFloat(__ENV.TIMEOUT) || 0.1) * 1000;
const testId = new Date().toISOString().slice(0, 19).replace('T', ' ');

// Constant Test Params
const K6_RATE = parseInt(__ENV.K6_RATE || 100);
const K6_DURATION = __ENV.SCRIPT_DURATION || '5m';

// Stress Test Params
const STRESS_START_RATE = parseInt(__ENV.STRESS_START_RATE || 10);
const STRESS_PEAK_RATE = parseInt(__ENV.STRESS_PEAK_RATE || 10000);
const STRESS_RAMP_UP_DURATION = __ENV.STRESS_RAMP_UP_DURATION || '10m';
const STRESS_SUSTAIN_DURATION = __ENV.STRESS_SUSTAIN_DURATION || '5m';
const STRESS_RAMP_DOWN_DURATION = __ENV.STRESS_RAMP_DOWN_DURATION || '1m';
const MAX_VUS = parseInt(__ENV.MAX_VUS || 200);
// ---------------------------------------------

const stressTestScenario = {
  executor: 'ramping-arrival-rate',
  startRate: STRESS_START_RATE,
  timeUnit: '1s',
  preAllocatedVUs: MAX_VUS,
  maxVUs: MAX_VUS,
  stages: [
    { target: STRESS_START_RATE, duration: '1m' }, // Brief warm-up/stabilization stage
    { target: STRESS_PEAK_RATE, duration: STRESS_RAMP_UP_DURATION },
    { target: STRESS_PEAK_RATE, duration: STRESS_SUSTAIN_DURATION },
    { target: 0, duration: STRESS_RAMP_DOWN_DURATION },
  ],
};

const constantTestScenario = {
  executor: 'constant-arrival-rate',
  rate: K6_RATE,
  timeUnit: '1s',
  duration: K6_DURATION,
  preAllocatedVUs: MAX_VUS, // Can reuse MAX_VUS for consistency
  maxVUs: MAX_VUS,
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

/**
 * A lightweight, log-free version of your original parsing function.
 * @param {string} htmlBody - The HTML content of the page.
 * @param {string} baseUrl - The base URL of the page to resolve relative paths.
 * @returns {string[]} An array of absolute URLs for the assets.
 */
function getAssetUrls(htmlBody, baseUrl) {
  const doc = parseHTML(htmlBody);
  const assetUrls = new Set();

  const selectors = [
    'link[href]',
    'script[src]',
    'img[src]',
    'source[src]',
    'video[src]',
  ];

  const foundElements = doc.find(selectors.join(','));

  foundElements.each((i, el) => {
    let assetPath = null;

    if (el && typeof el.attr === 'function') {
      assetPath = el.attr('href') || el.attr('src');
    }
    if (!assetPath && el && el.attributes) {
      assetPath = el.attributes.href || el.attributes.src;
    }
    if (!assetPath && el && typeof el.getAttribute === 'function') {
      assetPath = el.getAttribute('href') || el.getAttribute('src');
    }

    if (!assetPath || assetPath.startsWith('data:') || assetPath.startsWith('#')) {
      return;
    }

    const assetUrl = new URL(assetPath, baseUrl).toString();

    if (assetUrl.startsWith('http')) {
        assetUrls.add(assetUrl);
    }
  });

  return Array.from(assetUrls);
}

export function setup() {
  // --- All one-time logging moved here, inside setup() ---
  console.log('[config] Reading environment variables...');
  console.log(`[config] TARGET_URL: ${target_url}`);
  console.log(`[config] SERVER_TYPE: ${server_type}`);
  console.log(`[config] K6_SCENARIO: ${K6_SCENARIO}`);
  console.log(`[config] TEST_PATH: ${TEST_PATH}`);
  console.log(`[config] TIMEOUT (ms): ${TIMEOUT}`);
  console.log(`[config] Generated Test ID: ${testId}`);
  console.log('[config] Environment variable processing complete.');

  console.log('[init] Final k6 options have been assembled.');
  console.log(`[init] Running scenario: ${K6_SCENARIO}`);
  console.log(`[init] Test ID Tag: ${options.tags.testid}`);
  console.log(`[init] Server Tag: ${options.tags.server}`);
  console.log(`[init] Selected scenario configuration: ${JSON.stringify(options.scenarios[K6_SCENARIO], null, 2)}`);
  // -------------------------------------------------------

  console.log('--- Running Setup Phase ---');
  const pageToParse = `${target_url}/`;

  console.log(`[setup] Discovering assets by fetching the main page: ${pageToParse}`);

  const res = http.get(pageToParse, { responseType: 'text' });

  if (res.status !== 200 || !res.body) {
    throw new Error(`[setup] Could not fetch the page to parse assets. Status: ${res.status}. Aborting test.`);
  }

  const urls = getAssetUrls(res.body, res.url);
  console.log(`[setup] Discovered ${urls.length} assets to be used for the test.`);

  const assetRequests = urls.map(url => {
    return {
      method: 'GET',
      url: url,
      params: {
        timeout: TIMEOUT,
        tags: { resource_type: 'asset' },
        responseType: 'none' // Explicitly discard asset bodies
      },
    };
  });
  console.log('[setup] Pre-computed batch requests for all assets.');

  return { assetUrls: urls, assetRequests: assetRequests };
}

export default function (data) {
  const testPath = TEST_PATH === 'dynamic'
    ? `/dynamic/${Math.floor(Math.random() * 1000000) + 1}`
    : '/';

  const pageUrl = `${target_url}${testPath}`;

  group(`Load Page: ${testPath}`, function () {
    // --- OPTIMIZATION: STRATEGY 1 ---
    // Create the main page request object to be included in the batch.
    const mainPageRequest = {
      method: 'GET',
      url: pageUrl,
      params: {
        timeout: TIMEOUT,
        tags: { resource_type: 'html' },
        responseType: 'none'
      },
    };

    // Combine the main page request with the pre-computed asset requests.
    const allRequests = [mainPageRequest];
    if (data.assetRequests && data.assetRequests.length > 0) {
        allRequests.push(...data.assetRequests);
    }

    // Execute all requests in a single batch call.
    const responses = http.batch(allRequests);

    // The first response is always the main HTML page.
    const mainPageRes = responses[0];
    check(mainPageRes, {
      [`${server_type}: status is 200`]: (r) => r.status === 200,
    });

    // --- OPTIMIZATION: STRATEGY 2 ---
    // Use a single, efficient check for all asset responses.
    if (data.assetRequests && data.assetRequests.length > 0) {
      check(responses, {
          'all assets status is 200': (rs) =>
              // We slice(1) to skip the main page response and check the rest.
              // .every() is highly efficient for this validation.
              rs.slice(1).every((r) => r.status === 200),
      }, { resource_type: 'asset_check' });
    }
  });
}

export function teardown(data) {
  console.log('--- Running Teardown Phase ---');
  console.log(`[teardown] Test scenario '${K6_SCENARIO}' has completed.`);
  if (data && data.assetUrls) {
    console.log(`[teardown] The test was performed using ${data.assetUrls.length} assets discovered during setup.`);
  } else {
    console.log('[teardown] No asset data was passed from setup.');
  }
  console.log('----------------------------');
}
