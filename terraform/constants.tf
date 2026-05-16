variable "project_name" {
  description = "Nazwa projektu używana do tagowania zasobów."
  type        = string
  default     = "mgr"
}

variable "ssh_port" {
  description = "SSH port"
  type        = number
  default     = 22
}

variable "subject_port" {
  description = "Subject web port (HTTP)"
  type        = number
  default     = 80
}

variable "subject_port_https" {
  description = "Subject HTTPS port"
  type        = number
  default     = 443
}

variable "node_exporter_port" {
  description = "Port for Node Exporter"
  type        = number
  default     = 9100
}

variable "nginx_exporter_port" {
  description = "Port for nginx log exporter"
  type        = number
  default     = 9113
}

variable "cadvisor_port" {
  description = "Port for cAdvisor"
  type        = number
  default     = 8080
}

variable "grafana_port" {
  description = "Port for Grafana"
  type        = number
  default     = 3000
}

variable "prometheus_port" {
  description = "Port for Prometheus"
  type        = number
  default     = 9090
}
