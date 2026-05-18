# Universal Master Files (UMF)

## Problem

Application folder messy. Many `docker-compose.yml` same. Change one -> change 30. Error prone.

## Solution

Central templates. Zero `docker-compose.yml` in application folders.

## Structure

`applications/_infra/compose/`

- `ssr-nginx.yml`: SSR applications (Node/Bun/Deno) with Nginx proxy.
- `csr-nginx.yml`: Nginx static applications.
- `csr-apache.yml`: Apache static applications.

## Mechanism

Orchestrator or Ansible:

1. Detect application type (prefix `ssr-` or string `apache`).
2. Map to template.
3. Inject env vars:
   - `APPLICATION_DIR`: Application folder name (e.g. `csr-lit`).
   - `APPLICATION_ID`: Underscore name (e.g. `csr_lit`).
   - `BUILD_TARGET`: `runner` (SSR) or `nginx` (CSR).
   - `STATIC_PATH`: Framework path (e.g. `/_next/static/`, `/_nuxt/`).
4. Run: `docker-compose -f _infra/compose/[type].yml up`.

## Constraints

- Template use `../../${APPLICATION_DIR}` context.
- Application files MUST stay in `applications/[name]`.
- Templates MUST stay in `applications/_infra/compose`.

## Logic (Source of Truth)

- Local: `orchestrator/shared/infra/docker.py`
- Remote: `ansible/project/roles/application_server/tasks/main.yml`
