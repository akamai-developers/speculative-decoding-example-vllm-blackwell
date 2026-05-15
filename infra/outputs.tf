output "instance_ip" {
  description = "Public IP address of the GPU instance."
  value = tolist(linode_instance.gpu.ipv4)[0]
}

output "ssh_command" {
  description = "SSH command for connecting to the GPU instance."
  value = "ssh root@${tolist(linode_instance.gpu.ipv4)[0]}"
}

output "grafana_url" {
  description = "Grafana dashboard URL."
  value = "http://${tolist(linode_instance.gpu.ipv4)[0]}:3000"
}

output "streamlit_url" {
  description = "Streamlit visualization dashboard URL."

  value = "http://${tolist(linode_instance.gpu.ipv4)[0]}:8501"
}