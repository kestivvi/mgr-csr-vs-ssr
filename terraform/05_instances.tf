# trivy:ignore:AVD-AWS-0131[OK_for_thesis] EBS encryption is not required as no sensitive data is stored.
resource "aws_instance" "subject_server" {
  for_each = var.technologies

  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.subject_server_instance_type
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.subject_server.id]

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name     = "${var.project_name}-subject-server-${each.key}"
    Role     = "subject_server"
    Scenario = each.key
  }
}

# trivy:ignore:AVD-AWS-0131[OK_for_thesis] EBS encryption is not required as no sensitive data is stored.
resource "aws_instance" "load_generator" {
  for_each = var.technologies

  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.load_generator_instance_type
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.load_generator.id]

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name     = "${var.project_name}-load-generator-${each.key}"
    Role     = "load_generator"
    Scenario = each.key
  }
}

# The monitoring host remains a single, static resource
# trivy:ignore:AVD-AWS-0131[OK_for_thesis] EBS encryption is not required as no sensitive data is stored.
resource "aws_instance" "monitoring_host" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.monitoring_host_instance_type
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.monitoring_host.id]

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name = "${var.project_name}-monitoring-host"
    Role = "monitoring_host"
  }
}
