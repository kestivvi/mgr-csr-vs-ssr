#!/bin/bash
set -e
cd ../terraform
terraform init
terraform apply -auto-approve
cd ..

# Wait for 5 seconds
sleep 5

cd ./ansible
export ANSIBLE_CONFIG=./ansible.cfg
ansible-playbook site.yml
