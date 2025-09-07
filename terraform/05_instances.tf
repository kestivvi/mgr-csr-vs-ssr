resource "aws_instance" "load_generator_csr" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "c8g.2xlarge"
  # instance_type          = "t4g.micro"
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.load_generator.id]

  tags = {
    Name = "${var.project_name}-load-generator-csr"
    Role = "load_generator_csr"
  }
}

resource "aws_instance" "load_generator_ssr" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "c8g.2xlarge"
  # instance_type          = "t4g.micro"
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.load_generator.id]

  tags = {
    Name = "${var.project_name}-load-generator-ssr"
    Role = "load_generator_ssr"
  }
}


resource "aws_instance" "app_server_csr" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = "m8g.large"
  # instance_type          = "t4g.micro"
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
  # instance_type          = "t4g.micro"
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
  instance_type          = "t4g.micro"
  # instance_type          = "t4g.micro"
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.monitoring_server.id]

  tags = {
    Name = "${var.project_name}-monitoring-server"
    Role = "monitoring_server"
  }
}

# Monitoring 2 vcpu => (t4g.micro)
# App server 2 vcpu * 2 = 4 vcpu => (m8g.large)
# Load generator 4 vcpu * 2 = 8 vcpu => (c8g.2xlarge)
# Total 14 vcpu

