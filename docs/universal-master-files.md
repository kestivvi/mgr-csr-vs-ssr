# Universal Master Files (UMF)

## Problem

App folder messy. Many `docker-compose.yml` same. Change one -> change 30. Error prone.

## Solution

Central templates. Zero `docker-compose.yml` in app folders.

## Structure

`apps/_infra/compose/`

- `ssr.yml`: Node/Bun SSR apps.
- `csr.yml`: Nginx static apps.
- `apache.yml`: Apache static apps.

## Mechanism

Orchestrator or Ansible:

1. Detect app type (prefix `ssr-` or string `apache`).
2. Map to template.
3. Inject env vars:
   - `APP_DIR`: App folder name (e.g. `csr-lit`).
   - `APP_ID`: Underscore name (e.g. `csr_lit`).
   - `BUILD_TARGET`: `runner` (SSR) or `nginx` (CSR).
   - `STATIC_PATH`: Framework path (e.g. `/_next/static/`, `/_nuxt/`).
4. Run: `docker-compose -f _infra/compose/[type].yml up`.

## Constraints

- Template use `../../${APP_DIR}` context.
- App files MUST stay in `apps/[name]`.
- Templates MUST stay in `apps/_infra/compose`.

## Logic (Source of Truth)

- Local: `orchestrator/shared/infra/docker.py`
- Remote: `ansible/project/roles/app_server/tasks/main.yml`
