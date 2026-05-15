# Cloud infra setup for Linode instance with RTX 6000 Blackwell GPU 
# for running a speculative decoding examples in vllm 
#
# What this provisions: 
#
#

terraform {
  required_providers {
    linode = {
      source  = "linode/linode"
    }
  }
}

provider "linode" {
  token = var.linode_token
}

resource "linode_sshkey" "default" {
  label   = "${var.project_name}-key"
  ssh_key = chomp(file(var.ssh_public_key_path))
}

resource "linode_firewall" "fw" {
  label   = "${var.project_name}-fw"
  inbound_policy  = "DROP"
  outbound_policy = "ACCEPT"

  inbound {
    label    = "ssh"
    action   = "ACCEPT"
    protocol = "TCP"
    ports    = "22"
    ipv4     = [var.allowed_ip_cidr]
  }

  inbound {
    label    = "grafana"
    action   = "ACCEPT"
    protocol = "TCP"
    ports    = "3000"
    ipv4     = [var.allowed_ip_cidr]
  }

  inbound {
    label    = "streamlit"
    action   = "ACCEPT"
    protocol = "TCP"
    ports    = "8501"
    ipv4     = [var.allowed_ip_cidr]
  }
}

resource "linode_instance" "gpu" {
  label           = "${var.project_name}-gpu"
  region          = var.region
  type            = var.instance_type
  image           = "linode/ubuntu22.04"
  root_pass       = var.root_pass
  authorized_keys = [linode_sshkey.default.ssh_key]
  firewall_id = linode_firewall.fw.id

  metadata {
    user_data = base64encode(file("${path.module}/cloud-init.yaml"))
  }

  tags = ["vllm", "speculative-decoding", "gpu"]
}