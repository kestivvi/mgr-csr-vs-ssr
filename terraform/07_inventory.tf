resource "local_file" "ansible_inventory" {
  content = templatefile("${path.module}/templates/inventory.yml.tftpl", {
    app_servers       = aws_instance.app_server
    load_generators   = aws_instance.load_generator
    monitoring_server = aws_instance.monitoring_server
    ssh_user          = "ec2-user"
    private_key_path  = "~/.ssh/MGR1.pem"
    test_scenarios    = var.test_scenarios
  })
  filename = "${path.module}/../ansible/inventory.yml"
}