output "server_details" {
  description = "Detailed information about each server including IPs and purposes"
  value = {
    load_generator_csr = {
      public_ip  = aws_instance.load_generator_csr.public_ip
      private_ip = aws_instance.load_generator_csr.private_ip
      type       = "Load Generator (CSR)"
      purpose    = "Generates load for Client-Side Rendering tests"
      ssh_user   = "ec2-user"
    }
    load_generator_ssr = {
      public_ip  = aws_instance.load_generator_ssr.public_ip
      private_ip = aws_instance.load_generator_ssr.private_ip
      type       = "Load Generator (SSR)"
      purpose    = "Generates load for Server-Side Rendering tests"
      ssh_user   = "ec2-user"
    }
    app_server_csr = {
      public_ip  = aws_instance.app_server_csr.public_ip
      private_ip = aws_instance.app_server_csr.private_ip
      type       = "Application Server (CSR)"
      purpose    = "Hosts Client-Side Rendered application"
      ssh_user   = "ec2-user"
    }
    app_server_ssr = {
      public_ip  = aws_instance.app_server_ssr.public_ip
      private_ip = aws_instance.app_server_ssr.private_ip
      type       = "Application Server (SSR)"
      purpose    = "Hosts Server-Side Rendered application"
      ssh_user   = "ec2-user"
    }
    monitoring_server = {
      public_ip  = aws_instance.monitoring_server.public_ip
      private_ip = aws_instance.monitoring_server.private_ip
      type       = "Monitoring Server"
      purpose    = "Hosts Grafana and Prometheus monitoring stack"
      ssh_user   = "ec2-user"
      grafana    = "http://${aws_instance.monitoring_server.public_ip}:${var.grafana_port}"
      prometheus = "http://${aws_instance.monitoring_server.public_ip}:${var.prometheus_port}"
    }
  }
}

output "ssh_connection" {
  description = "SSH connection information"
  value = {
    command_format = "ssh -i <private_key_file> <ssh_user>@<public_ip>"
    ssh_user = "ec2-user"
    note = "Replace <private_key_file> with your AWS key pair file and <public_ip> with the server's public IP"
  }
}
