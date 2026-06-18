# CSR vs SSR Performance Benchmark

> Benchmark codebase for comparing the server-side cost of CSR and SSR web applications.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Terraform](https://img.shields.io/badge/Terraform-1.5%2B-7B42BC?logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Ansible](https://img.shields.io/badge/Ansible-2.15%2B-EE0000?logo=ansible&logoColor=white)](https://www.ansible.com/)
[![k6](https://img.shields.io/badge/k6-load%20testing-7D64FF?logo=k6&logoColor=white)](https://k6.io/)
[![AWS](https://img.shields.io/badge/AWS-Graviton%20ARM64-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/ec2/graviton/)

`csr-ssr-performance-benchmark` is the code part of a Master's thesis at Warsaw University of Life Sciences (SGGW/WULS). It supports empirical measurement of how Client-Side Rendering (CSR) and Server-Side Rendering (SSR) architectures behave under load.

It provisions AWS Graviton infrastructure, deploys standardized benchmark applications, drives traffic with k6 or wrk, collects Prometheus metrics, and generates reports and plots from the collected data.

## What it measures

- **Max RPS** from capacity tests
- **CPU and RAM utilization** during fixed-rate load tests
- **Latency percentiles** including p90, p95, and p99
- **Load-generator saturation signals** to validate whether an application, not the generator, became the bottleneck

> [!NOTE]
> In this project, SSG applications are treated as CSR for server-side cost analysis: request-time server work is static file serving.

## Features

- **Single CLI**: `mgr` drives setup, verification, tests, aggregation, and analysis.
- **31 benchmark applications**: CSR/SSG and SSR implementations across React, Angular, Vue, Svelte, Solid, Lit, Qwik, Next.js, Nuxt, Astro, SvelteKit, Fresh, Analog, and TanStack Start.
- **Runtime coverage**: Node.js, Bun, Deno, Nginx, and Apache.
- **Cloud isolation**: separate EC2 hosts for applications, load generators, and monitoring.
- **Local verification**: Docker-based smoke tests before running paid cloud experiments.
- **Analysis outputs**: Markdown reports and PNG plots generated from collected metrics.

## Architecture

```text
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ Load Generator   │   k6     │ Application SUT  │ metrics  │ Monitoring Host  │
│ EC2 / ARM64      │ ───────▶ │ EC2 / ARM64      │ ───────▶ │ Prometheus       │
└──────────────────┘          └──────────────────┘          │ Grafana          │
          ▲                            ▲                    └──────────────────┘
          │                            │                              ▲
          └────────────── mgr CLI ─────┴──────────────────────────────┘
                    Python + Terraform + Ansible
```

## Prerequisites

- Linux or WSL2
- Python **3.12+**
- Docker and Docker Compose
- Terraform **1.5+**
- Ansible **2.15+**
- AWS account, configured AWS credentials, and an EC2 key pair

## Quick start

```bash
git clone https://github.com/kestivvi/csr-ssr-performance-benchmark.git
cd csr-ssr-performance-benchmark/orchestrator

python3 -m venv venv
source venv/bin/activate
pip install -e .

cd ..
cp orchestrator/infra.example.yaml orchestrator/infra.yaml
$EDITOR orchestrator/infra.yaml
```

`orchestrator/infra.yaml` is local-only and ignored by Git. Set:

- `aws_region`: AWS region for all EC2 resources
- `key_name`: existing EC2 key pair name in that region
- `my_ip`: CIDR allowed to access SSH and monitoring endpoints
- `*_instance_type`: EC2 instance sizes for application, load-generator, and monitoring hosts

Verify selected applications locally before using AWS:

```bash
mgr verify --apps csr-react-nginx,ssr-nextjs-node
```

> [!IMPORTANT]
> `mgr setup`, `mgr test`, `mgr campaign`, and `mgr destroy` interact with paid AWS resources. Review `infra.yaml` and active applications before running them.

## Run a benchmark

Provision a cost-conscious fleet with only the applications you need:

```bash
mgr setup orchestrator/infra.yaml --apps csr-react-nginx,ssr-nextjs-node
```

Run a capacity test:

```bash
mgr test capacity --peak-rate 1000 --ramp-up 5m
```

Run a fixed-rate load test:

```bash
mgr test load --rps 200 --duration 5m --repetitions 3
```

Run a reproducible YAML scenario:

```bash
mgr test file orchestrator/test_scenarios/capacity_k6_prod_app.yaml
```

Analyze collected results:

```bash
mgr analyze capacity_k6 results/capacity_k6_2026-05-16_23-00-00
```

Tear down infrastructure:

```bash
mgr destroy
```

## CLI overview

```text
mgr setup       Provision and configure AWS infrastructure
mgr destroy     Tear down infrastructure
mgr test        Run load, capacity, file-based, or wrk tests
mgr campaign    Run a sequenced benchmark campaign
mgr aggregate   Merge partial result directories
mgr analyze     Generate reports and plots
mgr verify      Build and smoke-test applications locally
mgr preview     Run one application locally for manual inspection
```

## Repository layout

```text
.
├── orchestrator/        # Python CLI engine and test scenario definitions
├── terraform/           # AWS Graviton infrastructure
├── ansible/             # Server configuration, deployment, monitoring roles
├── applications/        # Standardized CSR and SSR benchmark applications
├── k6/                  # Load-testing scripts
│   └── fixtures/        # Captured HTML/assets used by k6 parser regression tests
├── docs/                # Infrastructure and methodology notes
└── results/             # Local experiment outputs, ignored by Git
```

Files under `k6/fixtures/` are intentionally tracked generated outputs from selected benchmark applications. They are regression fixtures for `k6/extract.test.js`, not hand-authored application source.

## Benchmark applications

Applications follow a common contract so the load generator sees comparable behavior:

- `/` renders `Hello World` plus an interactive counter.
- `/dynamic/:id` renders the dynamic ID.
- `/dynamic-app/:id` renders a larger application-like page.
- Static assets are offloaded to Nginx or Apache.
- SSR runtimes handle dynamic HTML generation only.
- Gzip compression is enabled at the web-server layer.

See [`applications/README.md`](applications/README.md) for the full application contract and registry.

## Test scenarios

Scenario files live in [`orchestrator/test_scenarios/`](orchestrator/test_scenarios/). They define the test type, repetition count, load curve, target path type, timeout, and asset-fetching behavior.

Example capacity scenario:

```yaml
test_type: capacity_k6
num_repetitions: 3

capacity_k6_options:
  start_rate: 1
  peak_rate: 1200
  ramp_up: 5m
  sustain: 2m
  ramp_down: 1m
  warmup: 1m
  max_vus: 500
  path_type: dynamic
  timeout: 0.5s
```

## Analysis outputs

`mgr analyze` writes reports and plots next to the input result directory:

- `capacity_report_k6.md` or load-test report files
- `plots/*.png` for report figures
- grouped comparisons by rendering strategy, framework, runtime, or selected champions

`mgr aggregate` can merge partial reruns into a single result set:

```bash
mgr aggregate results/run_part1 results/run_part2 -o results/aggregated_run
```

## Quality checks

Run the relevant checks before committing changes:

```bash
# Python
cd orchestrator
./venv/bin/ruff format .
./venv/bin/mypy --strict .
./venv/bin/pytest
./venv/bin/ruff check .

# Ansible
cd ../ansible
prettier --write "./**/*.{yml,yaml}"
yamllint .
ansible-lint

# Terraform
cd ../terraform
terraform fmt -recursive
terraform validate
tflint --recursive
```
