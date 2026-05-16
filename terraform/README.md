# Terraform

Provisions the AWS Graviton (ARM64) fleet for the experiments: load generators, subject hosts, and the Prometheus/Grafana monitoring host. Driven by `mgr setup` / `mgr destroy` — see [mgr-code/README.md](../README.md).

> [!WARNING]
> Do not run `terraform apply` / `destroy` directly. Use the `mgr` CLI so the Ansible inventory and experiment state stay consistent.

## Quality control

```bash
terraform fmt -recursive
terraform validate
tflint --recursive
trivy config .
```

First-time `tflint` setup:

```bash
curl -s https://raw.githubusercontent.com/terraform-linters/tflint/master/install_linux.sh | bash
tflint --init
```
