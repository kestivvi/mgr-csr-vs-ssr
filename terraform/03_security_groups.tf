# trivy:ignore:AVD-AWS-0104[OK_for_thesis] Egress is open to allow OS updates and software installation.
resource "aws_security_group" "allow_ssh" {
  name        = "${var.project_name}-allow-ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.main.id

  # Egress is defined here as it's the only rule and not conflicting
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name = "${var.project_name}-allow-ssh"
    Role = "security"
  }
}

resource "aws_security_group_rule" "allow_ssh_from_my_ip" {
  type              = "ingress"
  security_group_id = aws_security_group.allow_ssh.id
  from_port         = var.ssh_port
  to_port           = var.ssh_port
  protocol          = "tcp"
  cidr_blocks       = [var.my_ip]
  description       = "Allow SSH access from my IP"
}


# Grupa dla serwerów aplikacyjnych (applications)
resource "aws_security_group" "application_server" {
  name        = "${var.project_name}-application-server-sg"
  description = "Security group for Application Servers (CSR/SSR)"
  vpc_id      = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-application-server-sg"
    Role = "security"
  }
}

# trivy:ignore:AVD-AWS-0104[OK_for_thesis] Egress is open to allow OS updates and software installation.
resource "aws_security_group_rule" "application_server_egress_all" {
  type              = "egress"
  security_group_id = aws_security_group.application_server.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  ipv6_cidr_blocks  = ["::/0"]
  description       = "Allow all outbound traffic"
}

resource "aws_security_group_rule" "application_server_ingress_web_from_lg" {
  type                     = "ingress"
  security_group_id        = aws_security_group.application_server.id
  from_port                = var.application_port
  to_port                  = var.application_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.load_generator.id
  description              = "Allow web traffic from load generator"
}

resource "aws_security_group_rule" "application_server_ingress_web_from_my_ip" {
  type              = "ingress"
  security_group_id = aws_security_group.application_server.id
  from_port         = var.application_port
  to_port           = var.application_port
  protocol          = "tcp"
  cidr_blocks       = [var.my_ip]
  description       = "Allow web traffic from my IP for testing"
}

resource "aws_security_group_rule" "application_server_ingress_https_from_lg" {
  type                     = "ingress"
  security_group_id        = aws_security_group.application_server.id
  from_port                = var.application_port_https
  to_port                  = var.application_port_https
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.load_generator.id
  description              = "Allow HTTPS traffic from load generator"
}

resource "aws_security_group_rule" "application_server_ingress_https_from_my_ip" {
  type              = "ingress"
  security_group_id = aws_security_group.application_server.id
  from_port         = var.application_port_https
  to_port           = var.application_port_https
  protocol          = "tcp"
  cidr_blocks       = [var.my_ip]
  description       = "Allow HTTPS traffic from my IP for testing"
}

resource "aws_security_group_rule" "application_server_ingress_node_exporter" {
  type                     = "ingress"
  security_group_id        = aws_security_group.application_server.id
  from_port                = var.node_exporter_port
  to_port                  = var.node_exporter_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.monitoring_host.id
  description              = "Allow Prometheus to scrape node_exporter"
}

resource "aws_security_group_rule" "application_server_ingress_nginx_exporter" {
  type                     = "ingress"
  security_group_id        = aws_security_group.application_server.id
  from_port                = var.nginx_exporter_port
  to_port                  = var.nginx_exporter_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.monitoring_host.id
  description              = "Allow Prometheus to scrape nginx_log_exporter"
}

resource "aws_security_group_rule" "application_server_ingress_cadvisor" {
  type                     = "ingress"
  security_group_id        = aws_security_group.application_server.id
  from_port                = var.cadvisor_port
  to_port                  = var.cadvisor_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.monitoring_host.id
  description              = "Allow Prometheus to scrape cAdvisor"
}


# Grupa dla generatora obciążenia
resource "aws_security_group" "load_generator" {
  name        = "${var.project_name}-load-generator-sg"
  description = "Allow all outbound traffic from load generator"
  vpc_id      = aws_vpc.main.id
  tags = {
    Name = "${var.project_name}-load-generator-sg"
    Role = "security"
  }
}

# trivy:ignore:AVD-AWS-0104[OK_for_thesis] Egress is open to allow OS updates and software installation.
resource "aws_security_group_rule" "load_generator_egress_all" {
  type              = "egress"
  security_group_id = aws_security_group.load_generator.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  description       = "Allow all outbound traffic"
}

resource "aws_security_group_rule" "load_generator_ingress_node_exporter" {
  type                     = "ingress"
  security_group_id        = aws_security_group.load_generator.id
  from_port                = var.node_exporter_port
  to_port                  = var.node_exporter_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.monitoring_host.id
  description              = "Allow Prometheus to scrape node_exporter"
}


# Grupa dla monitoring host
resource "aws_security_group" "monitoring_host" {
  name        = "${var.project_name}-monitoring-host-sg"
  description = "Security group for Monitoring Host"
  vpc_id      = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-monitoring-host-sg"
    Role = "security"
  }
}

# trivy:ignore:AVD-AWS-0104[OK_for_thesis] Egress is open to allow OS updates and software installation.
resource "aws_security_group_rule" "monitoring_host_egress_all" {
  type              = "egress"
  security_group_id = aws_security_group.monitoring_host.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  ipv6_cidr_blocks  = ["::/0"]
  description       = "Allow all outbound traffic"
}

resource "aws_security_group_rule" "monitoring_host_ingress_grafana" {
  type              = "ingress"
  security_group_id = aws_security_group.monitoring_host.id
  from_port         = var.grafana_port
  to_port           = var.grafana_port
  protocol          = "tcp"
  cidr_blocks       = [var.my_ip]
  description       = "Allow Grafana access from my IP"
}

resource "aws_security_group_rule" "monitoring_host_ingress_prometheus_ui" {
  type              = "ingress"
  security_group_id = aws_security_group.monitoring_host.id
  from_port         = var.prometheus_port
  to_port           = var.prometheus_port
  protocol          = "tcp"
  cidr_blocks       = [var.my_ip]
  description       = "Allow Prometheus UI access from my IP"
}

resource "aws_security_group_rule" "monitoring_host_ingress_node_exporter_self" {
  type              = "ingress"
  security_group_id = aws_security_group.monitoring_host.id
  from_port         = var.node_exporter_port
  to_port           = var.node_exporter_port
  protocol          = "tcp"
  self              = true
  description       = "Allow Prometheus to scrape node_exporter on self"
}

resource "aws_security_group_rule" "allow_prometheus_rw_from_lg" {
  type                     = "ingress"
  from_port                = var.prometheus_port
  to_port                  = var.prometheus_port
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.load_generator.id
  security_group_id        = aws_security_group.monitoring_host.id
  description              = "Allow k6 to push metrics to Prometheus"
}