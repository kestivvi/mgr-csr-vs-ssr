# Grupa pozwalająca na dostęp SSH z Twojego IP
resource "aws_security_group" "allow_ssh" {
  name        = "${var.project_name}-allow-ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Grupa dla serwerów aplikacyjnych
resource "aws_security_group" "app_server" {
  name        = "${var.project_name}-app-server-sg"
  description = "Security group for App Servers (CSR/SSR)"
  vpc_id      = aws_vpc.main.id

  # Ruch przychodzący z generatora obciążenia na porcie 80 (Nginx)
  ingress {
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.load_generator.id]
  }

  # Ruch przychodzący z serwera monitoringu (Prometheus)
  ingress {
    from_port       = 9100 # Node Exporter
    to_port         = 9100
    protocol        = "tcp"
    security_groups = [aws_security_group.monitoring_server.id]
  }
  ingress {
    from_port       = 9113 # Nginx Exporter
    to_port         = 9113
    protocol        = "tcp"
    security_groups = [aws_security_group.monitoring_server.id]
  }
}

# Grupa dla generatora obciążenia
resource "aws_security_group" "load_generator" {
  name        = "${var.project_name}-load-generator-sg"
  description = "Security group for Load Generator"
  vpc_id      = aws_vpc.main.id
}

# Grupa dla serwera monitoringu
resource "aws_security_group" "monitoring_server" {
  name        = "${var.project_name}-monitoring-server-sg"
  description = "Security group for Monitoring Server"
  vpc_id      = aws_vpc.main.id

  # Dostęp do Grafany z Twojego IP
  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Dostęp do Prometheus UI z Twojego IP
  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }
}