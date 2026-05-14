config {
  call_module_type    = "all"
  force               = false
  format              = "compact"
}

plugin "terraform" {
  enabled = true
}

plugin "aws" {
  enabled = true
  version = "0.42.0"
  source  = "github.com/terraform-linters/tflint-ruleset-aws"
}

# --- THESIS SPECIFIC OPTIMIZATIONS ---

# We use numbered files (01_..., 02_...) for step-by-step clarity in the thesis,
# so we disable the standard module structure warning to keep the output clean.
rule "terraform_standard_module_structure" {
  enabled = false
}

# Ensure we use modern instance types to minimize virtualization overhead
rule "aws_instance_invalid_type" {
  enabled = true
}

# Tagging: Ensure all resources are identifiable for the research project
rule "aws_resource_missing_tags" {
  enabled = true
  tags    = ["Name", "Role"]
}