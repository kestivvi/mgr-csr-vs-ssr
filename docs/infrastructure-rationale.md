# Infrastructure Selection Rationale

This document outlines the reasoning and conclusions regarding the EC2 instance selection for the MGR thesis performance benchmarks.

## 1. The Crux: Which Instance Family?
**Decision:** Compute Optimized (`C` Series)

### Reasoning:
*   **Workload Profile:** Server-Side Rendering (SSR) is a CPU-intensive task. The process of executing JavaScript/TypeScript to generate HTML per request is limited by the processor's speed and efficiency (V8 Engine performance).
*   **Eliminating Variables:** General Purpose (`M`) or Memory Optimized (`R`) instances provide more RAM than required, adding unnecessary cost. More importantly, **Burstable (`T`) instances are avoided** because their CPU credit system introduces unpredictable performance drops that would invalidate benchmark results.
*   **Predictability:** The `C` series provides a dedicated, consistent amount of compute power, ensuring that "Run 1" and "Run 100" are comparable.

## 2. The Core: Architecture & Generation
**Decision:** AWS Graviton4 (`c8g`)

### Reasoning:
*   **Industry Relevancy:** Testing on ARM-based architecture represents the modern shift in cloud infrastructure.
*   **Architectural Benefits:** Graviton4 offers 50% more cores and 75% more memory bandwidth than Graviton3. For JIT-heavy workloads like Node.js, the improved branch prediction and L2 cache sizes provide a significant performance boost per-core.

## 3. The Scaling Question: `.medium` (1 vCPU) vs `.large` (2 vCPUs)
**Decision:** `.medium` (1 vCPU / 2 GiB RAM) for the Target Server.

### Reasoning:
*   **Single-Threaded Purity:** Node.js is primarily single-threaded. By using a 1 vCPU instance, we measure the "raw" efficiency of a framework's event loop without the noise of multi-core context switching or internal thread contention.
*   **Saturation Velocity:** A 1 vCPU instance hits its "breaking point" at lower Request-per-Second (RPS) levels. This allows for faster identification of performance plateaus and clearer "hockey stick" latency graphs for the thesis.
*   **Cost Efficiency:** Minimizes AWS spend while still providing high-quality, reproducible data.

## 4. The Critical Bottleneck: Load Generators
**Decision:** `c8g.2xlarge` (8 vCPUs / 16 GiB RAM)

### Reasoning:
*   **Load vs. Capacity Nuance:** Empirical testing shows that for steady-state **Load Tests** (e.g., constant 50 RPS), a `.medium` generator is sufficient. However, for **Capacity Tests** (ramping up to thousands of RPS to find the breaking point), a `.2xlarge` is required.
*   **Preventing Generator-Side Throttling:** During rapid ramping, a 1 vCPU generator (`.medium`) becomes the bottleneck before the target server. This results in "clipped" peaks and inaccurate latency reporting.
*   **Precision at Scale:** A `2xlarge` instance provides 8 cores, allowing the load generation tool (like k6) to use multiple threads for networking, encryption, and VU management. This ensures that the requested RPS is delivered with microsecond precision during high-load capacity tests.
*   **Clean Data:** By over-provisioning the generator for capacity runs, we guarantee that any latency measured is caused solely by the **Subject Server**, removing the "noisy generator" variable from the thesis data.

## Summary Table

| Role | Instance Type | vCPU | RAM | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Subject Server** | `c8g.medium` | 1 | 2 GiB | Measures single-core efficiency; saturates early for clear data. |
| **Load Generator** | `c8g.2xlarge` | 8 | 16 GiB | Essential for **Capacity Tests**; `.medium` is only sufficient for light **Load Tests**. |
| **Monitoring** | `t4g.small` | 2 | 2 GiB | Burstable is acceptable here as monitoring is not being measured. |

---
*Last Updated: April 2026*
