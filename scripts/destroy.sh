#!/bin/bash
set -e
cd ../terraform
terraform init
terraform destroy -auto-approve
