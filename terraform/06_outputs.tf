output "server_details" {
  description = "Detailed information about each server including IPs, purposes, and Terraform resource addresses"
  value = merge(
    {
      for key, server in aws_instance.app_server : "app_server_${key}" => {
        public_ip         = server.public_ip
        private_ip        = server.private_ip
        type              = var.test_scenarios[key].description
        purpose           = var.test_scenarios[key].purpose
        ssh_user          = "ec2-user"
        terraform_address = "aws_instance.app_server[\"${key}\"]"
      }
    },
    {
      for key, server in aws_instance.load_generator : "load_generator_${key}" => {
        public_ip         = server.public_ip
        private_ip        = server.private_ip
        type              = "Load Generator (${upper(key)})"
        purpose           = "Generates load for ${upper(key)} tests"
        ssh_user          = "ec2-user"
        terraform_address = "aws_instance.load_generator[\"${key}\"]"
      }
    },
    {
      "monitoring_server" = {
        public_ip         = aws_instance.monitoring_server.public_ip
        private_ip        = aws_instance.monitoring_server.private_ip
        type              = "Monitoring Server"
        purpose           = "Hosts Grafana and Prometheus monitoring stack"
        ssh_user          = "ec2-user"
        grafana           = "http://${aws_instance.monitoring_server.public_ip}:${var.grafana_port}"
        prometheus        = "http://${aws_instance.monitoring_server.public_ip}:${var.prometheus_port}"
        terraform_address = "aws_instance.monitoring_server"
      }
    }
  )
}

output "ssh_connection" {
  description = "SSH connection information"
  value = {
    command_format = "ssh -i <private_key_file> <ssh_user>@<public_ip>"
    ssh_user       = "ec2-user"
    note           = "Replace <private_key_file> with your AWS key pair file and <public_ip> with the server's public IP"
  }
}