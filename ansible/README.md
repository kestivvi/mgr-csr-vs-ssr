# Ansible

Configures the EC2 hosts provisioned by Terraform: applications (Docker + reverse proxy), load generators (k6, wrk), and the monitoring host (Prometheus, Grafana). Driven by `mgr setup` — see [mgr-code/README.md](../README.md).

The inventory is sourced from Terraform state via [`terraform-inventory`](https://github.com/adammck/terraform-inventory).

## Manual invocation (debugging)

```bash
TF_STATE=../terraform ansible-inventory -i $(which terraform-inventory) --graph
TF_STATE=../terraform ansible-playbook  -i $(which terraform-inventory) ./ping.yml
TF_STATE=../terraform ansible-playbook  -i $(which terraform-inventory) ./site.yml
```

## Quality control

```bash
prettier --check "**/*.{yml,yaml}" --write
yamllint .
ansible-lint
```
