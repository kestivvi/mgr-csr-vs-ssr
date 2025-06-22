resource "aws_instance" "load_generator" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "c8g.xlarge"
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.load_generator.id]

  tags = {
    Name = "${var.project_name}-load-generator"
    Role = "load_generator"
  }
}


resource "aws_instance" "app_server_csr" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "m8g.large"
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.app_server.id]

  tags = {
    Name = "${var.project_name}-app-server-csr"
    Role = "app_server"
  }
}


resource "aws_instance" "app_server_ssr" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "m8g.large"
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.app_server.id]

  tags = {
    Name = "${var.project_name}-app-server-ssr"
    Role = "app_server"
  }
}

resource "aws_instance" "monitoring_server" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "t4g.medium"
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.monitoring_server.id]

  tags = {
    Name = "${var.project_name}-monitoring-server"
    Role = "monitoring_server"
  }
}
