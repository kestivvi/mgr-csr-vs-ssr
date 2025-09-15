
### Master commands chain

```bash
# Validate the terraform code
terraform validate

# Format the terraform code
terraform fmt -recursive

# Run tflint
tflint --recursive

# Run trivy (previously tfsec)
trivy config ./
```

### Setup Tflint

```bash
# Install tflint
curl -s https://raw.githubusercontent.com/terraform-linters/tflint/master/install_linux.sh | bash

# Initialize tflint
tflint --init

# Run tflint
tflint --recursive
```

### Setup Trivy (previously tfsec)

```bash
# Install trivy
sudo apt-get install wget gnupg
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | sudo tee /usr/share/keyrings/trivy.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb generic main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy

trivy config ./
```