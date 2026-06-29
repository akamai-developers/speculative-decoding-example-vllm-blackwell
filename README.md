# vLLM Speculative Decoding Observability Demo

This project provisions an NVIDIA RTX PRO 6000 Blackwell GPU instance on Akamai Cloud / Linode using Terraform and prepares the environment for running and visualizing speculative decoding with vLLM.

This project compares:

- Standard target-only decoding
- Speculative decoding using a draft + target model pair

and visualizes:

- Throughput metrics
- Acceptance rate metrics
- GPU utilization
- Latency metrics
- Baseline vs speculative performance

through Grafana and a custom Streamlit dashboard.


# High-Level Architecture

```text
Terraform
  ↓
Creates GPU VM + networking
  ↓
cloud-init
  ↓
Installs packages + NVIDIA drivers
Clones repository
Configures shell environment
  ↓
User SSHs into VM
  ↓
Run setup_env.sh
  ↓
Configures environment variables
Creates Python virtual environment
Installs Python dependencies
Downloads models from Hugging Face
  ↓
Run start_monitor.sh
  ↓
Starts:
  - Streamlit dashboard
  - Grafana/ Prometheus docker containers
  ↓
Run vllm_server.sh
  ↓
Starts:
  - Baseline vLLM server (port 8000)
  - Speculative vLLM server (port 8001)
```


# Repository Structure

```text
speculative-decoding-example-vllm-blackwell/
├── app/
│   ├── docker-compose.yml
│   ├── grafana.json
│   ├── prometheus.yml
│   └── streamlit.py
│
├── infra/
│   ├── main.tf
│   ├── _variables.tf
│   ├── _outputs.tf
│   ├── terraform.tfvars.example
│   └── cloud-init.yaml
│
├── scripts/
    ├── helpers/
    │   ├── _baseline.sh
    │   ├── _download_models.sh
    │   ├── _speculative.sh
    │   └── _streamlit.sh
│   ├── common.sh
│   ├── setup_env.sh
│   ├── start_monitor.sh
│   └── start_vllm.sh
│
├── .gitignore
├── config.env
├── requirements.txt
└── README.md
```


# Inference Architecture & Model Configuration

This demo spins up two independent vLLM engine servers running on port 8000 (baseline) and port 8001 (speculative) to allow the Streamlit frontend to run side-by-side inference comparisons. 

| Service / Server | Port | Role / Core Purpose |
| :--- | :--- | :--- |
| **Baseline Engine** | `8000` | Standard target-only inference pass |
| **Speculative Engine** | `8001` | Accelerated draft + target speculative decoding |
| **Streamlit Dashboard** | `8501` | Interactive Frontend UI & experiment coordinator |


# Metrics and Observability
This project tracks performance from two perspectives: **Client-Side User Experience** (Streamlit) and **Server-Side Engine Efficiency** (vLLM + Grafana).

## 1. Streamlit Dashboard (Client-Side UX)
The Streamlit frontend provides real-time, user-facing experiment metrics to compare standard inference against speculative decoding side-by-side..

### **Captured UX Metrics**
* **Total Latency (s):** Wall-clock time from request initiation to the final token.
* **Generated Tokens:** The total count of tokens in the final output.
* **Throughput (tokens/sec):** The generation speed as experienced by the user.

## 2. Grafana & Prometheus (Server-Side Telemetry)
For infrastructure-level insights, server-side metrics exposed by vLLM are scraped and visualized to show exactly how the engine behaves under the hood.

```text
  [ vLLM Engine ] (Exposes /metrics)
         │
         ▼
  [ Prometheus ]  (Scrapes telemetry)
         │
         ▼
  [ Grafana Dashboard ] (Visualizes performance)
```


# How to Run the Demo

## 1. Install Local Prerequisites

Install:

- Terraform
- jq
- curl
- SSH client

MacOS example:

```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
brew install jq
```

Verify Terraform:

```bash
terraform version
```

## 2. Create an Akamai Cloud / Linode Account

You must have:

- An Akamai Cloud / Linode account
- Access to the RTX PRO 6000 Blackwell GPU plan
- A Personal Access Token to authenticate API and Terraform requests (configured in the next step).
- An SSH public key generated on your local machine and added to your Linode profile for secure instance access.

## 3. Create a Linode API Token & Environment Variables

To deploy the infrastructure, you will need a Linode Personal Access Token and a secure root password for the Ubuntu 22.04 LTS instance.

First, navigate to the [Linode Cloud Manager](https://cloud.linode.com/profile/tokens) and create a token with the following permissions:
- **Linodes:** Read/Write
- **Firewalls:** Read/Write
- **Events:** Read/Write
- **Volumes:** Read/Write

Next, export both your API token and your chosen root password as environment variables in your terminal so Terraform can securely access them:

```bash
# Paste your API token from Linode here:
export TF_VAR_linode_token="YOUR_LINODE_API_TOKEN"

# Create and type any secure password of your choice here:
export TF_VAR_root_pass="YOUR_CUSTOM_SECURE_ROOT_PASSWORD"
```

## 4. Configure Terraform Variables

Navigate to the `infra/` directory and create your local variables file from the provided template:

```bash
cd infra/
cp terraform.tfvars.example terraform.tfvars
```

Open `terraform.tfvars`. You should stick to the baseline defaults defined in the example file, but you **must** update the network access restriction:

* **Allowed IP CIDR:** Update this to your local machine's public IP address (e.g., `192.0.2.1/32`) to secure access to the instance.


## 5. Provision Infrastructure

From the `infra/` directory:

```bash
terraform init
terraform plan
terraform apply
```

Terraform will:

- create the GPU VM
- configure networking/firewall rules
- install NVIDIA drivers
- clone the repository
- reboot the machine for Blackwell driver registration

## 6. SSH into the VM

After Terraform finishes and the VM reboots:

```bash
ssh root@<instance-ip>
```

You should automatically land in:

```text
/root/vllm-speculative-demo
```

## 7. Start the Demo

Run (in this order):

```bash
export HF_TOKEN="hf_xxx"
./scripts/setup_env.sh
./scripts/start-monitor.sh
./scripts/start-vllm.sh
```

The scripts prints:

```text
Baseline vLLM:    http://<instance-ip>:8000
Speculative vLLM: http://<instance-ip>:8001
Streamlit:        http://<instance-ip>:8501
```

Open the dashboard in your browser:

```text
http://<instance-ip>:8501
```

## 8. Updating Runtime Configuration Settings

To change runtime settings such as:

- GPU memory utilization
- max model length
- speculative token count
- ports
- model selection

```text
config.env
```

Example:

```bash
GPU_MEMORY_UTILIZATION="0.46"
```

Then restart the corresponding vLLM server.