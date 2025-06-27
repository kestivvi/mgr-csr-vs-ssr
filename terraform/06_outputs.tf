output "ansible_inventory" {
  description = "Struktura danych gotowa do użycia jako inwentarz Ansible (w formacie JSON). Zawiera adresy IP i użytkownika do logowania."
  value = {
    # Grupa 'all' definiuje wszystkie hosty i ich zmienne.
    # Ansible automatycznie sprawi, że te zmienne będą dostępne dla hosta
    # w każdej innej grupie, do której należy.
    all = {
      hosts = {
        "load-generator" = {
          ansible_host = aws_instance.load_generator.public_ip,
          private_ip   = aws_instance.load_generator.private_ip,
          ansible_user = "ec2-user" # Użytkownik dla Amazon Linux
        }
        "app-server-csr" = {
          ansible_host = aws_instance.app_server_csr.public_ip,
          private_ip   = aws_instance.app_server_csr.private_ip,
          ansible_user = "ec2-user"
        }
        "app-server-ssr" = {
          ansible_host = aws_instance.app_server_ssr.public_ip,
          private_ip   = aws_instance.app_server_ssr.private_ip,
          ansible_user = "ec2-user"
        }
        "monitoring-server" = {
          ansible_host = aws_instance.monitoring_server.public_ip,
          private_ip   = aws_instance.monitoring_server.private_ip,
          ansible_user = "ec2-user"
        }
      }
    }

    # Grupy specyficzne dla ról.
    # Nie musimy tu powtarzać zmiennych, ponieważ są one dziedziczone z grupy 'all'.
    # Pusty obiekt {} jest wystarczający, aby zadeklarować przynależność hosta do grupy.
    load_generators = {
      hosts = {
        "load-generator" = {}
      }
    }
    app_servers_csr = {
      hosts = {
        "app-server-csr" = {}
      }
    }
    app_servers_ssr = {
      hosts = {
        "app-server-ssr" = {}
      }
    }
    monitoring_servers = {
      hosts = {
        "monitoring-server" = {}
      }
    }
  }
}

output "public_ips" {
  description = "Prosta lista publicznych adresów IP dla szybkiego dostępu."
  value = {
    load_generator    = aws_instance.load_generator.public_ip
    app_server_csr    = aws_instance.app_server_csr.public_ip
    app_server_ssr    = aws_instance.app_server_ssr.public_ip
    monitoring_server = aws_instance.monitoring_server.public_ip
  }
}

output "grafana_url" {
  description = "Adres URL do panelu Grafany."
  value       = "http://${aws_instance.monitoring_server.public_ip}:${var.grafana_port}"
}

output "prometheus_url" {
  description = "Adres URL do interfejsu Prometheus."
  value       = "http://${aws_instance.monitoring_server.public_ip}:${var.prometheus_port}"
}
