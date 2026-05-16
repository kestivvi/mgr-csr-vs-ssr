# Universal Master Files (UMF)

## Problem

Subject folder messy. Many `docker-compose.yml` same. Change one -> change 30. Error prone.

## Solution

Central templates. Zero `docker-compose.yml` in subject folders.

## Structure

`subjects/_infra/compose/`

- `ssr-nginx.yml`: SSR subjects (Node/Bun/Deno) with Nginx proxy.
- `csr-nginx.yml`: Nginx static subjects.
- `csr-apache.yml`: Apache static subjects.

## Mechanism

Orchestrator or Ansible:

1. Detect subject type (prefix `ssr-` or string `apache`).
2. Map to template.
3. Inject env vars:
   - `SUBJECT_DIR`: Subject folder name (e.g. `csr-lit`).
   - `SUBJECT_ID`: Underscore name (e.g. `csr_lit`).
   - `BUILD_TARGET`: `runner` (SSR) or `nginx` (CSR).
   - `STATIC_PATH`: Framework path (e.g. `/_next/static/`, `/_nuxt/`).
4. Run: `docker-compose -f _infra/compose/[type].yml up`.

## Constraints

- Template use `../../${SUBJECT_DIR}` context.
- Subject files MUST stay in `subjects/[name]`.
- Templates MUST stay in `subjects/_infra/compose`.

## Logic (Source of Truth)

- Local: `orchestrator/shared/infra/docker.py`
- Remote: `ansible/project/roles/subject_server/tasks/main.yml`
