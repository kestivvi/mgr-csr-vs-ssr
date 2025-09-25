variable "aws_region" {
  description = "Region AWS, w którym zostaną utworzone zasoby."
  type        = string
  # default     = "eu-central-1"
  default     = "ap-south-1" # Mumbai region
}

variable "project_name" {
  description = "Nazwa projektu używana do tagowania zasobów."
  type        = string
  default     = "mgr"
}

variable "key_name" {
  description = "Nazwa pary kluczy EC2 do użycia dla instancji."
  type        = string
  # default     = "MGR1" # For Frankfurt region (eu-central-1)
  default     = "MGR-M" # For Mumbai region (ap-south-1)
}

variable "my_ip" {
  description = "Twój publiczny adres IP dozwolony do połączeń SSH i dostępu do paneli (Grafana, Prometheus). WAŻNE: Zmień to!"
  type        = string
  default     = "46.205.200.250/32" # https://www.whatismyip.com/ or allow all IPs with 0.0.0.0/0
}

variable "ssh_port" {
  description = "SSH port"
  type        = number
  default     = 22
}

variable "app_port" {
  description = "Application web port (HTTP)"
  type        = number
  default     = 80
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

variable "app_server_instance_type" {
  description = "Instance type for all application servers (e.g., 2 vCPUs)."
  type        = string
  default     = "t4g.micro"
  # default = "c8g.medium"
}

variable "load_generator_instance_type" {
  description = "Instance type for all load generator servers (e.g., 4 vCPUs)."
  type        = string
  default     = "t4g.micro"
  # default = "c8g.xlarge"
}

# c8g.medium - 1 vCPUs - 2 GB RAM - $0.027/hour
# c8g.large - 2 vCPUs - 4 GB RAM - $0.054/hour
# c8g.xlarge - 4 vCPUs - 8 GB RAM - $0.108/hour
# c8g.2xlarge - 8 vCPUs - 16 GB RAM - $0.2159/hour
# c8g.4xlarge - 16 vCPUs - 32 GB RAM - $0.4318/hour

# 16 Load Generators * 4 vCPUs + 16 App Servers * 1 vCPUs + 1 Monitoring Server * 1 vCPU =
# 64 vCPUs + 16 vCPUs + 1 vCPU = 81 vCPUs

# 81 vCPUs * $0.027/hour = $2.187/hour

variable "test_scenarios" {
  description = "A map of test scenarios to provision. The key is the short name (e.g., 'csr')."
  type = map(object({
    description = string
    purpose     = string
    app_dir     = string
  }))
  default = {
    ##########################
    # Static Site
    ##########################
    # "CSR-Vanilla" = {
    #   description = "Application Server (CSR-Vanilla)"
    #   purpose     = "Hosts Client-Side Rendered application"
    #   app_dir     = "apps/csr-vanilla-nginx"
    # },
    # "CSR-Vanilla-Apache" = {
    #   description = "Application Server (CSR-Vanilla-Apache)"
    #   purpose     = "Hosts Client-Side Rendered application"
    #   app_dir     = "apps/csr-vanilla-apache"
    # },

    ##########################
    # CSR
    ##########################
    # "CSR-Angular" = {
    #   description = "Application Server (CSR-Angular-Nginx)"
    #   purpose     = "Hosts Angular Client-Side Rendered application"
    #   app_dir     = "apps/csr-angular"
    # },
    # "CSR-React" = {
    #   description = "Application Server (CSR-React-Nginx)"
    #   purpose     = "Hosts Client-Side Rendered application"
    #   app_dir     = "apps/csr-react"
    # },
    # "CSR-SolidJS" = {
    #   description = "Application Server (CSR-SolidJS-Nginx)"
    #   purpose     = "Hosts Client-Side Rendered application"
    #   app_dir     = "apps/csr-solidjs"
    # },
    # "CSR-SolidJS-Apache" = {
    #   description = "Application Server (CSR-SolidJS-Apache)"
    #   purpose     = "Hosts Client-Side Rendered application"
    #   app_dir     = "apps/csr-solidjs-apache"
    # },
    # "CSR-SvelteKit-Static" = {
    #   description = "Application Server (CSR-SvelteKit-Static-Nginx)"
    #   purpose     = "Hosts SvelteKit Client-Side Rendered application"
    #   app_dir     = "apps/csr-svelte-kit-static"
    # },
    # "CSR-Vue" = {
    #   description = "Application Server (CSR-Vue-Nginx)"
    #   purpose     = "Hosts Vue Client-Side Rendered application"
    #   app_dir     = "apps/csr-vue"
    # },

    ##########################
    # SSR
    ##########################

    # "SSR-AnalogJS" = {
    #   description = "Application Server (SSR-AnalogJS)"
    #   purpose     = "Hosts AnalogJS Server-Side Rendered application"
    #   app_dir     = "apps/ssr-analogjs"
    # },
    # "SSR-NextJS" = {
    #   description = "Application Server (SSR-NextJS)"
    #   purpose     = "Hosts Server-Side Rendered application"
    #   app_dir     = "apps/ssr-nextjs"
    # },
    # "SSR-NextJS-Bun" = {
    #   description = "Application Server (SSR-NextJS-Bun)"
    #   purpose     = "Hosts NextJS Server-Side Rendered application"
    #   app_dir     = "apps/ssr-nextjs-bun"
    # },
    # "SSR-NextJS-Deno" = {
    #   description = "Application Server (SSR-NextJS-Deno)"
    #   purpose     = "Hosts NextJS Server-Side Rendered application"
    #   app_dir     = "apps/ssr-nextjs-deno"
    # },
    # "SSR-NuxtJS" = {
    #   description = "Application Server (SSR-NuxtJS)"
    #   purpose     = "Hosts NuxtJS Server-Side Rendered application"
    #   app_dir     = "apps/ssr-nuxtjs"
    # },
    # "SSR-Qwik-City" = {
    #   description = "Application Server (SSR-Qwik-City)"
    #   purpose     = "Hosts Qwik-City Server-Side Rendered application"
    #   app_dir     = "apps/ssr-qwik-city"
    # },
    # "SSR-Solid-Start" = {
    #   description = "Application Server (SSR-Solid-Start)"
    #   purpose     = "Hosts Solid-Start Server-Side Rendered application"
    #   app_dir     = "apps/ssr-solid-start"
    # },
    # "SSR-SvelteKit" = {
    #   description = "Application Server (SSR-SvelteKit)"
    #   purpose     = "Hosts SvelteKit Server-Side Rendered application"
    #   app_dir     = "apps/ssr-svelte-kit"
    # },
    # "SSR-SvelteKit-Bun" = {
    #   description = "Application Server (SSR-SvelteKit-Bun)"
    #   purpose     = "Hosts SvelteKit Server-Side Rendered application"
    #   app_dir     = "apps/ssr-svelte-kit-bun"
    # },
    # "SSR-React-Router" = {
    #   description = "Application Server (SSR-React-Router)"
    #   purpose     = "Hosts React-Router Server-Side Rendered application"
    #   app_dir     = "apps/ssr-react-router"
    # }
    "SSR-Fresh" = {
      description = "Application Server (SSR-Fresh)"
      purpose     = "Hosts Fresh Server-Side Rendered application"
      app_dir     = "apps/ssr-fresh"
    }
  }
}