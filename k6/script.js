import http from 'k6/http';
import { check, group } from 'k6';
import { parseHTML } from 'k6/html';
import { URL } from 'https://jslib.k6.io/url/1.0.0/index.js';

// --- K6 Test Configuration (Init Context) ---
// This code runs once per VU before the test starts. Keep it minimal.
const target_url = __ENV.TARGET_URL || 'http://localhost:8080';
const server_type = __ENV.SERVER_TYPE || 'unknown';
const K6_RATE = parseInt(__ENV.K6_RATE || 100);
const K6_DURATION = __ENV.SCRIPT_DURATION || '5m';
const K6_SCENARIO = __ENV.K6_SCENARIO || 'stress_test';
const TEST_PATH = __ENV.TEST_PATH || 'static';
const TIMEOUT = (parseFloat(__ENV.TIMEOUT) || 0.2) * 1000;
const testId = new Date().toISOString().slice(0, 19).replace('T', ' ');
// ---------------------------------------------

const stressTestScenario = {
  executor: 'ramping-arrival-rate',
  startRate: 10,
  timeUnit: '1s',
  preAllocatedVUs: 200,
  maxVUs: 200,
  stages: [
    { target: 10, duration: '1m' },
    { target: 1000, duration: '20m' },
    { target: 1000, duration: '5m' },
    { target: 0, duration: '1m' },
  ],
};

const constantTestScenario = {
  executor: 'constant-arrival-rate',
  rate: K6_RATE,
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
  discardResponseBodies: false,
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
  console.log(`[config] K6_RATE: ${K6_RATE}`);
  console.log(`[config] SCRIPT_DURATION: ${K6_DURATION}`);
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
  
  const res = http.get(pageToParse);
  
  if (res.status !== 200 || !res.body) {
    throw new Error(`[setup] Could not fetch the page to parse assets. Status: ${res.status}. Aborting test.`);
  }
  
  const urls = getAssetUrls(res.body, res.url);
  
  console.log(`[setup] Discovered ${urls.length} assets to be used for the test.`);
  
  return { assetUrls: urls };
}

export default function (data) {
  const testPath = TEST_PATH === 'dynamic' 
    ? `/dynamic/${Math.floor(Math.random() * 1000000) + 1}`
    : '/';
  
  const pageUrl = `${target_url}${testPath}`;

  group(`Load Page: ${testPath}`, function () {
    const mainPageRes = http.get(pageUrl, { 
      timeout: TIMEOUT,
      tags: { resource_type: 'html' } 
    });

    check(mainPageRes, {
      [`${server_type}: status is 200`]: (r) => r.status === 200,
    });

    const assetUrls = data.assetUrls;

    const assetRequests = assetUrls.map(url => {
      return {
        method: 'GET',
        url: url,
        params: { 
          timeout: TIMEOUT,
          tags: { resource_type: 'asset' },
          responseType: 'none' 
        },
      };
    });

    if (assetRequests.length > 0) {
      const assetResponses = http.batch(assetRequests);
      
      assetResponses.forEach((res) => {
        check(res, {
          'asset status is 200': (r) => r.status === 200,
        }, { resource_type: 'asset_check' });
      });
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