output "instance_ip" {
  description = "Public IP address of the GPU instance."
  value = linode_instance.gpu.ip_address
}

output "ssh_command" {
  description = "SSH command for connecting to the GPU instance."
  value = "ssh root@${linode_instance.gpu.ip_address}"
}

output "grafana_url" {
  description = "Grafana dashboard URL."
  value = "http://${linode_instance.gpu.ip_address}:3000"
}

output "streamlit_url" {
  description = "Streamlit visualization dashboard URL."

  value = "http://${linode_instance.gpu.ip_address}:8501"
}