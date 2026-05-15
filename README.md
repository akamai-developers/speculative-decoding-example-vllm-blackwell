# vLLM Speculative Decoding Observability Demo

This project provisions an NVIDIA RTX PRO 6000 Blackwell GPU instance on Akamai Cloud / Linode using Terraform and prepares the environment for running and visualizing speculative decoding with vLLM.

The goal of the project is to compare:

* Standard target-only decoding
* Speculative decoding using a draft + target model pair

and expose:

* Throughput metrics
* Acceptance rate metrics
* GPU utilization
* Latency metrics
* Baseline vs speculative comparisons

through Grafana and a custom visualization dashboard.

---

# Architecture

```text
Terraform
  ↓
Creates RTX PRO 6000 Blackwell GPU Instance
  ↓
cloud-init bootstraps Ubuntu environment
  ↓
NVIDIA drivers installed automatically
  ↓
vLLM inference services
  ├── Baseline decoding
  └── Speculative decoding
  ↓
Prometheus scrapes metrics
  ↓
Grafana + Streamlit visualization
```

---

# Repository Structure

```text
.
├── infra/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── terraform.tfvars.example
│   └── cloud-init.yaml
│
├── scripts/
│   ├── setup-vllm.sh
│   ├── download-models.sh
│   ├── run-baseline.sh
│   └── run-speculative.sh
│
├── README.md
└── LICENSE
```

---

# Prerequisites

## Local Machine

Install:

* Terraform
* jq
* curl
* SSH client

MacOS example:

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
brew install jq
```

Verify:

```bash
terraform version
```

---

# Akamai Cloud / Linode Requirements

You must have:

* An Akamai Cloud / Linode account
* Access to the RTX PRO 6000 Blackwell GPU plan
* A Personal Access Token
* An SSH public key

---

# Creating a Linode API Token

Navigate to:

[https://cloud.linode.com/profile/tokens](https://cloud.linode.com/profile/tokens)

Create a Personal Access Token with:

* Linodes: Read/Write
* Firewalls: Read/Write
* Events: Read/Write
* Volumes: Read/Write

Export the token locally:

```bash
export TF_VAR_linode_token="YOUR_TOKEN"
```

Terraform automatically maps:

```text
TF_VAR_linode_token
```

to the Terraform variable:

```hcl
variable "linode_token"
```

---
