FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "checkpoints/stage3_grpo/final", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.9"]
