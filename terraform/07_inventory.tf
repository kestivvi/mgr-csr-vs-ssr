resource "local_file" "ansible_inventory" {
  content = templatefile("${path.module}/templates/inventory.yml.tftpl", {
    subject_servers = { for k, v in aws_instance.subject_server : k => {
      tags        = v.tags
      public_ip   = v.public_ip
      private_ip  = v.private_ip
      public_dns  = v.public_dns
      private_dns = v.private_dns
    } }
    load_generators = { for k, v in aws_instance.load_generator : k => {
      tags        = v.tags
      public_ip   = v.public_ip
      private_ip  = v.private_ip
      public_dns  = v.public_dns
      private_dns = v.private_dns
    } }
    monitoring_server = {
      tags        = aws_instance.monitoring_server.tags
      public_ip   = aws_instance.monitoring_server.public_ip
      private_ip  = aws_instance.monitoring_server.private_ip
      public_dns  = aws_instance.monitoring_server.public_dns
      private_dns = aws_instance.monitoring_server.private_dns
    }
    ssh_user         = "ec2-user"
    private_key_path = "~/.ssh/${var.key_name}.pem"
    technologies     = var.technologies
  })
  filename = "${path.module}/../ansible/inventory/inventory.yml"
}