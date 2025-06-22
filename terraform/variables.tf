variable "aws_region" {
  description = "Region AWS, w którym zostaną utworzone zasoby."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Nazwa projektu używana do tagowania zasobów."
  type        = string
  default     = "mgr-perf-test"
}

variable "key_name" {
  description = "Nazwa pary kluczy EC2 do użycia dla instancji."
  type        = string
  default     = "MGR1"
}

variable "my_ip" {
  description = "Twój publiczny adres IP dozwolony do połączeń SSH i dostępu do paneli (Grafana, Prometheus). WAŻNE: Zmień to!"
  type        = string
  default     = "0.0.0.0/0" # UWAGA: To jest niebezpieczne! Zmień na swój IP, np. "89.123.45.67/32". Możesz sprawdzić swój IP na stronie https://www.whatismyip.com/
}