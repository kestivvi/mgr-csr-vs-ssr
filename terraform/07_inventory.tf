resource "local_file" "ansible_inventory" {
  content = templatefile("${path.module}/templates/inventory.yml.tftpl", {
    load_generator     = aws_instance.load_generator
    app_server_csr     = aws_instance.app_server_csr
    app_server_ssr     = aws_instance.app_server_ssr
    monitoring_server  = aws_instance.monitoring_server
    ssh_user           = "ec2-user"
    private_key_path   = "~/.ssh/MGR1.pem"
  })
  filename = "${path.module}/../ansible/inventory.yml"
} 