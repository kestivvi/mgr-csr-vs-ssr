// 1. Load hydration support FIRST
import '@lit-labs/ssr-client/lit-element-hydrate-support.js';

// 2. Load component definitions
import './components/counter-element.js';

console.log('Hydration complete');
