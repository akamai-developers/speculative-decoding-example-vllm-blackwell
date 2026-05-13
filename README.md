# speculative-decoding-example-vllm-blackwell

# GPU Environment Setup on Linode RTX 6000 Blackwell
This guide walks through provisioning a Linode GPU instance with an NVIDIA RTX 6000 Blackwell GPU, installing the required NVIDIA open kernel drivers, and preparing a Python development environment for running inference workloads with vLLM.


## 1. Create a Linode GPU Instance

Create a new Linode instance with the following configuration:

- **GPU:** NVIDIA RTX 6000 Blackwell
- **OS:** Ubuntu 22.04 LTS
- **Authentication:** SSH key

Add your SSH public key during instance creation to enable secure remote access to the instance. 

## 2. SSH into the Instance

After the instance is running, connect to it over SSH:

```bash
ssh root@<your-instance-ip>
```
Replace <your-instance-ip> with the public IP address of your Linode instance.


## 3. Update the System

Update and upgrade the base system packages:

```bash
apt update && apt upgrade -y
```

## 4. Install Development Tools

Install common development, monitoring, and Python environment tools:

```bash
apt install -y build-essential git curl wget htop tmux nvtop python3-pip python3-venv
``` 

These packages include:

- **build-essential** for compiling software
- **git** for working with repositories
- **curl** and wget for downloading files
- **htop** for system monitoring
- **tmux** for persistent terminal sessions
- **nvtop** for GPU monitoring
- **python3-pip** and **python3-venv** for Python environments

## 5. Install required NVIDIA Drivers

The NVIDIA RTX 6000 Blackwell requires the NVIDIA open kernel modules instead of the standard server drivers on Ubuntu 22.04.

First install the Linux kernel headers and dependencies required to build the NVIDIA kernel modules:

```bash
apt install -y linux-headers-$(uname -r) build-essential dkms
```

Install the NVIDIA open kernel driver:

```bash
apt install -y nvidia-driver-595-open
``` 

Reboot the system to load the NVIDIA kernel modules:
```bash
reboot
```

After reconnecting, verify that the GPU is detected:
```bash
nvidia-smi
```

You should see the NVIDIA RTX 6000 Blackwell listed in the output.