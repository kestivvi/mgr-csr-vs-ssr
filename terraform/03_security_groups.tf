# Grupa pozwalająca na dostęp SSH z Twojego IP
resource "aws_security_group" "allow_ssh" {
  name        = "${var.project_name}-allow-ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = var.ssh_port
    to_port     = var.ssh_port
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
    description = "Allow SSH access from my IP"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
}

# Grupa dla serwerów aplikacyjnych
resource "aws_security_group" "app_server" {
  name        = "${var.project_name}-app-server-sg"
  description = "Security group for App Servers (CSR/SSR)"
  vpc_id      = aws_vpc.main.id

  # Ruch przychodzący z generatora obciążenia na porcie 80 (Nginx)
  ingress {
    from_port       = var.app_port
    to_port         = var.app_port
    protocol        = "tcp"
    security_groups = [aws_security_group.load_generator.id]
    description     = "Allow web traffic from load generator"
  }

  ingress {
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
    description = "Allow web traffic from my IP for testing"
  }

  # Ruch przychodzący z serwera monitoringu (Prometheus)
  ingress {
    from_port       = var.node_exporter_port # Node Exporter
    to_port         = var.node_exporter_port
    protocol        = "tcp"
    security_groups = [aws_security_group.monitoring_server.id]
    description     = "Allow Prometheus to scrape node-exporter"
  }
  ingress {
    from_port       = var.nginx_exporter_port # Nginx Exporter
    to_port         = var.nginx_exporter_port
    protocol        = "tcp"
    security_groups = [aws_security_group.monitoring_server.id]
    description     = "Allow Prometheus to scrape nginx-exporter"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
}

# Grupa dla generatora obciążenia
resource "aws_security_group" "load_generator" {
  name        = "${var.project_name}-load-generator-sg"
  description = "Allow all outbound traffic from load generator"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  ingress {
    from_port       = var.node_exporter_port
    to_port         = var.node_exporter_port
    protocol        = "tcp"
    security_groups = [aws_security_group.monitoring_server.id]
    description     = "Allow Prometheus to scrape node-exporter"
  }

  tags = {
    Name = "${var.project_name}-load-generator-sg"
  }
}

# Grupa dla serwera monitoringu
resource "aws_security_group" "monitoring_server" {
  name        = "${var.project_name}-monitoring-server-sg"
  description = "Security group for Monitoring Server"
  vpc_id      = aws_vpc.main.id

  # Dostęp do Grafany z Twojego IP
  ingress {
    from_port   = var.grafana_port
    to_port     = var.grafana_port
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
    description = "Allow Grafana access from my IP"
  }

  # Dostęp do Prometheus UI z Twojego IP
  ingress {
    from_port   = var.prometheus_port
    to_port     = var.prometheus_port
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
    description = "Allow Prometheus UI access from my IP"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }
}