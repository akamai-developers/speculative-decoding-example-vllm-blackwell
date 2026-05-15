variable "project_name" {
  type        = string
  description = "Prefix used for naming all resources."
  default     = "vllm-speculative-demo"
}

variable "linode_token" {
  type        = string
  description = "Linode API token used by Terraform to provision infrastructure."
  sensitive = true
}

variable "root_pass" {
  type        = string
  description = "Temporary root password for the GPU instance."
  sensitive = true
}

variable "region" {
  type        = string
  description = "Linode region where the GPU instance will be created."
  default = "id-cgk"
}

variable "instance_type" {
  type        = string
  description = "Linode GPU instance type/plan slug."
  default = "g3-gpu-rtxpro6000"
}

variable "ssh_public_key_path" {
  type        = string
  description = "Path to the local SSH public key used for VM access."
  default = "~/.ssh/id_ed25519.pub"
}

variable "allowed_ip_cidr" {
  type        = string
  description = "CIDR block allowed to access SSH/Grafana/Streamlit."
  default = "0.0.0.0/0"
}

variable "repo_url" {
  type        = string
  description = "Git repository cloned onto the GPU instance."
}

variable "hf_token" {
  type        = string
  description = "Hugging Face token."
  sensitive   = true
}

variable "github_token" {
  type      = string
  description = "GitHub access token"
  sensitive = true
}

variable "github_username" {
  type = string
  description = "Your GitHub username"
}