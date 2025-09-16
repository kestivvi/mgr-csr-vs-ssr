Need to install terraform-inventory tool.

Commands:
```sh
# 
TF_STATE=../terraform ansible-inventory -i $(which terraform-inventory) --graph

# 
TF_STATE=../terraform ansible-inventory -i $(which terraform-inventory) --list

#
TF_STATE=../terraform ansible-playbook -i $(which terraform-inventory) ./ping.yml

#
TF_STATE=../terraform ansible-playbook -i $(which terraform-inventory) ./site.yml
```

To check the validity of ansible:
```bash
# Format the ansible files
prettier --write "**/*.{yml,yaml}"

# Run ansible-lint
ansible-lint

# Run trivy
trivy config .
```