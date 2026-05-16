variable "aws_region" {
  description = "Region AWS, w którym zostaną utworzone zasoby."
  type        = string
  # default     = "eu-central-1"
  default = "ap-south-1" # Mumbai region
}

variable "key_name" {
  description = "Nazwa pary kluczy EC2 do użycia dla instancji."
  type        = string
  # default     = "MGR1" # For Frankfurt region (eu-central-1)
  default = "MGR-M" # For Mumbai region (ap-south-1)
}

variable "my_ip" {
  description = "Twój publiczny adres IP dozwolony do połączeń SSH i dostępu do paneli (Grafana, Prometheus). WAŻNE: Zmień to!"
  type        = string
  default     = "46.205.195.40/32" # https://www.whatismyip.com/ or allow all IPs with 0.0.0.0/0
}

variable "subject_server_instance_type" {
  description = "Instance type for all application servers (e.g., 2 vCPUs)."
  type        = string
  default     = "t4g.micro"
  # default = "c8g.medium"
}

variable "load_generator_instance_type" {
  description = "Instance type for all load generator servers (e.g., 4 vCPUs)."
  type        = string
  default     = "t4g.micro"
  # default     = "c8g.2xlarge"
}

variable "monitoring_server_instance_type" {
  description = "Instance type for the monitoring server."
  type        = string
  default     = "t4g.micro"
}

variable "technologies" {
  description = "A map of technologies (subjects) to provision. The key is the subject ID (e.g., 'csr-vanilla')."
  type = map(object({
    description = string
    purpose     = string
    subject_dir = string
  }))
  default = {
    "CSR-Vanilla" = {
      description = "Subject Server (CSR-Vanilla)"
      purpose     = "Hosts Client-Side Rendered subject"
      subject_dir = "subjects/csr-vanilla-nginx"
    }
  }
}

# 350 10
# 700 20
# 1400 30
# 2100 40
# 2800 50
# 3500 60
# 4200 70
# 4900 80
# 5600 90
# 6300 100
# 7000 110
# 7700 120
# 8400 130
# 9100 140
# 9800 150
# 10500 160
