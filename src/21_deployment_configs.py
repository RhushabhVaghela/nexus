#!/usr/bin/env python3
"""
Deployment Configurations for Production
vLLM, Docker, Kubernetes
"""

import json
from pathlib import Path
import os

def create_vllm_config():
    """Create vLLM server config"""
    config = {
        "model": "checkpoints/stage3_grpo/final",
        "tensor-parallel-size": 1,
        "gpu-memory-utilization": 0.9,
        "max-model-len": 4096,
        "dtype": "bfloat16",
    }
    return config

def create_dockerfile():
    """Create Dockerfile for deployment"""
    dockerfile = """FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model
COPY checkpoints/ /app/checkpoints/

# Run vLLM server
EXPOSE 8000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \\
     "--model", "checkpoints/stage3_grpo/final", \\
     "--tensor-parallel-size", "1", \\
     "--gpu-memory-utilization", "0.9"]
"""
    return dockerfile

def create_docker_compose():
    """Create docker-compose.yml"""
    compose = {
        "version": "3.8",
        "services": {
            "nexus-model": {
                "build": ".",
                "ports": ["8000:8000"],
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "0"
                },
                "volumes": [
                    "./checkpoints:/app/checkpoints"
                ]
            }
        }
    }
    return compose

def create_k8s_deployment():
    """Create Kubernetes deployment"""
    k8s = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "nexus-model"},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": "nexus"}},
            "template": {
                "metadata": {"labels": {"app": "nexus"}},
                "spec": {
                    "containers": [
                        {
                            "name": "model",
                            "image": "nexus:latest",
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "limits": {"nvidia.com/gpu": "1"}
                            }
                        }
                    ]
                }
            }
        }
    }
    return k8s

def main():
    output_dir = Path("deployment")
    output_dir.mkdir(exist_ok=True)
    
    # vLLM config
    with open(output_dir / "vllm_config.json", "w") as f:
        json.dump(create_vllm_config(), f, indent=2)
    
    # Dockerfile
    with open(output_dir / "Dockerfile", "w") as f:
        f.write(create_dockerfile())
    
    # Docker-compose
    with open(output_dir / "docker-compose.yml", "w") as f:
        json.dump(create_docker_compose(), f, indent=2)
    
    # K8s
    with open(output_dir / "k8s_deployment.yaml", "w") as f:
        json.dump(create_k8s_deployment(), f, indent=2)
    
    print(f"✓ Deployment configs created in: {output_dir}")
    print("\nTo deploy:")
    print("  docker build -t nexus:latest .")
    print("  docker run -p 8000:8000 nexus:latest")

if __name__ == "__main__":
    main()
