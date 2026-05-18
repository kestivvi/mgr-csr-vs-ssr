# MGR Orchestrator

> Performance benchmarking engine for the Master's Thesis *"Comparative Analysis of Server Performance for Web Applications in CSR and SSR Architectures"*.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Terraform](https://img.shields.io/badge/Terraform-1.5%2B-7B42BC?logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Ansible](https://img.shields.io/badge/Ansible-2.15%2B-EE0000?logo=ansible&logoColor=white)](https://www.ansible.com/)
[![k6](https://img.shields.io/badge/k6-load%20testing-7D64FF?logo=k6&logoColor=white)](https://k6.io/)
[![AWS](https://img.shields.io/badge/AWS-Graviton%20%28ARM64%29-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/ec2/graviton/)

The MGR Orchestrator automates the full empirical research lifecycle: provisioning isolated AWS environments, deploying web application **Subjects**, generating reproducible load with k6, and reducing the captured Prometheus time-series into publication-ready tables and plots.

## Highlights

- **Single CLI (`mgr`)** — drives infrastructure, experiments, and analysis from one Typer-based entrypoint.
- **Benchmark Subjects** — React, Vue, Svelte, Solid, Angular, Lit, Qwik, Next.js, Nuxt, Astro, SvelteKit, SolidStart, TanStack Start, Fresh, Analog… across Node.js, Bun, Deno, Nginx, and Apache.
- **AWS Graviton native** — Terraform-managed `c8g`/`m8g` ARM64 fleet with Ansible-driven server roles.
- **Strict isolation** — separate EC2 hosts for the System Under Test, the k6 load generators, and the Prometheus/Grafana monitoring stack.
- **Scientific rigor** — JIT warmup phases, configurable repetitions, schema-validated artifacts, and aggregation utilities for partial reruns.
- **Local pre-flight** — `mgr verify` exercises every Subject inside Docker before any cloud spend.

## Architecture

```
        ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
        │ Load Generator   │ ──k6──▶ │ Subject (SUT)    │ ──────▶ │ Monitoring Host  │
        │ EC2 (c8g.2xlarge)│         │ EC2 (c8g.medium) │ metrics │ Prometheus +     │
        └──────────────────┘         └──────────────────┘         │ Grafana          │
                  ▲                            ▲                  └──────────────────┘
                  │                            │                            │
                  └────────── mgr CLI ─────────┴────────────────────────────┘
                          (Terraform + Ansible + Python)
```

## Domain Vocabulary

The CLI, the codebase, and the thesis share one vocabulary. Keep these meanings straight when reading code or results.

| Term | Meaning |
| :--- | :--- |
| **Subject** | One web application under test, identified by `(strategy, framework, metaframework, runtime)`. |
| **Profile** | The load specification (rate curve, duration, repetitions). Lives under `orchestrator/test_scenarios/`. |
| **Study** | One execution of a Profile against a set of Subjects (`mgr test`). |
| **Experiment** | The atomic unit: one Subject × one Profile run, producing one Artifact. |
| **Artifact** | A structured directory in `results/` with raw Prometheus data, k6 output, logs, and metadata. |
| **Campaign** | A scripted sequence: provision → warm-up → test → rotate Subjects. |

## Prerequisites

- Linux or WSL2 (Ubuntu/Debian recommended)
- Python **3.12+**, Terraform **1.5+**, Ansible **2.15+**, Docker + Compose
- AWS account with an EC2 key pair and `aws configure` completed

## Quick Start

```bash
# 1. Install the orchestrator
cd orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -e .

# 2. Configure your AWS environment
cp infra.example.yaml infra.yaml
$EDITOR infra.yaml   # set aws_region, key_name, my_ip

# 3. Verify Subjects locally (free, runs in Docker)
mgr verify --apps csr-react-nginx,ssr-nextjs-node
```

> [!IMPORTANT]
> `mgr setup`, `mgr destroy`, and `mgr test` provision and exercise paid AWS resources. Run them yourself — don't delegate to autonomous agents.

## CLI Overview

```
mgr setup       Provision + configure AWS infrastructure
mgr destroy     Tear down all infrastructure
mgr test        Run performance experiments (load | capacity | file | wrk | stop)
mgr campaign    Sequential research campaign (provision → warmup → test → rotate)
mgr analyze     Generate statistical reports and plots from results
mgr aggregate   Merge repetition artifacts into one dataset
mgr verify      Build & smoke-test Subjects locally in Docker
mgr preview     Run a single Subject locally for manual inspection
```

### Provision the cloud environment

```bash
# Bring up only the Subjects you need (cost-conscious default)
mgr setup infra.yaml --apps csr-react-nginx,ssr-nextjs-node

# Tear everything down when you're done
mgr destroy
```

Useful flags: `-f/--force` to rebuild over an existing fleet, `--exclude` to skip Subjects, `-y/--yes` for non-interactive runs.

### Run a performance study

> [!TIP]
> `mgr test` defaults to **every active Subject** in the fleet. You normally restrict the fleet at `mgr setup` time, not at test time — restricting tests would leave provisioned instances idle.

```bash
# Capacity test: ramp RPS until the system saturates → reveals Max RPS
mgr test capacity --peak-rate 1000 --ramp-up 5m

# Load test: hold a constant rate → measures CPU, RAM, p90/p95/p99 latency
mgr test load --rps 200 --duration 5m --repetitions 3

# Reproducible study driven by a YAML profile
mgr test file orchestrator/test_scenarios/capacity_k6_prod.yaml
```

A minimal capacity profile (`test_scenarios/capacity_k6.example.yaml`):

```yaml
test_type: capacity_k6
num_repetitions: 3

capacity_k6_options:
  start_rate: 1          # starting RPS
  peak_rate: 1200        # target peak RPS
  ramp_up: 5m            # linear ramp from start_rate to peak_rate
  sustain: 2m            # hold at peak_rate
  ramp_down: 1m
  warmup: 1m             # JIT warmup before measurement begins
  max_vus: 500           # pre-allocated virtual users
  path_type: dynamic     # 'static' (fixed) or 'dynamic' (randomised paths)
  timeout: 0.5s          # requests slower than this count as errors
```

### Analyze the results

```bash
# Reduce a capacity Study into a Markdown report + LaTeX-ready plots
mgr analyze capacity_k6 results/capacity_k6_2026-05-16_23-00-00
```

Outputs land alongside the input directory:
- `capacity_report_k6.md` — Markdown scorecards
- `plots/` — PNG charts (violin, time-series, bar) sized for the thesis

Available report types: `load`, `capacity_k6`, `capacity_wrk`, `champions` (head-to-head; pass `--champions <a>,<b>`).

> [!NOTE]
> **`mgr aggregate`** merges partial reruns back into a Study — useful when a handful of Experiments fail on a transient cloud issue and you'd rather rerun those individually than redo the whole Study.
>
> ```bash
> mgr aggregate results/run_part1 results/run_part2 -o results/aggregated_run
> ```

### Inspect a Subject locally

```bash
mgr preview ssr-nextjs-node   # builds + runs the Subject; opens on localhost
```

## Repository Layout

```
mgr-code/
├── orchestrator/        # Python CLI engine (Typer, pandas, matplotlib, seaborn)
│   ├── src/orchestrator/
│   │   └── actions/     # setup · test · analyze · aggregate · campaign · verify · preview · destroy
│   ├── test_scenarios/  # YAML Profile templates (dev / preprod / prod)
│   └── infra.example.yaml
├── terraform/           # AWS Graviton infrastructure (c8g / m8g)
├── ansible/             # Roles & playbooks for SUT, load generator, monitoring host
├── subjects/            # Benchmark Subjects (csr-* / ssr-*)
├── k6/                  # k6 scripts used by the load generator
├── docs/                # Methodology notes (inventory, infra rationale, wrk usage)
└── results/             # Materialized Artifacts (git-ignored)
```

See [`subjects/README.md`](subjects/README.md) for the contract every benchmark Subject must satisfy (functional identity, static offloading, runtime versions). See [`docs/`](docs/) for methodology notes.

## Quality Control

Run before every commit. The orchestrator targets `mypy --strict` and clean Ruff output.

```bash
# Python
cd orchestrator
./venv/bin/ruff format .
./venv/bin/ruff check .
./venv/bin/mypy --strict .
./venv/bin/pytest

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
