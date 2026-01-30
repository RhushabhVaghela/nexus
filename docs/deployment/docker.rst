.. _docker_deployment:

Docker Deployment
=================

This guide covers deploying Nexus using Docker containers.

Prerequisites
-------------

* Docker 20.10+
* Docker Compose 2.0+
* NVIDIA Docker runtime (for GPU support)

Quick Start
-----------

Using Docker Compose:

.. code-block:: bash

   docker-compose up -d

Building the Image
------------------

.. code-block:: bash

   docker build -t nexus:latest .

Running with GPU Support
------------------------

.. code-block:: bash

   docker run --gpus all -it --rm \
       -v $(pwd)/models:/app/models \
       -v $(pwd)/data:/app/data \
       -p 8000:8000 \
       nexus:latest

Docker Compose Configuration
----------------------------

Create ``docker-compose.yml``:

.. code-block:: yaml

   version: '3.8'
   
   services:
     nexus:
       build: .
       image: nexus:latest
       container_name: nexus
       runtime: nvidia
       environment:
         - NVIDIA_VISIBLE_DEVICES=all
         - CUDA_VISIBLE_DEVICES=0
       volumes:
         - ./models:/app/models
         - ./data:/app/data
         - ./config:/app/config
       ports:
         - "8000:8000"
       command: python scripts/nexus_server.py
       
     redis:
       image: redis:7-alpine
       container_name: nexus-redis
       ports:
         - "6379:6379"

Multi-Stage Build
-----------------

Dockerfile for production:

.. code-block:: dockerfile

   # Build stage
   FROM python:3.10-slim as builder
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --user -r requirements.txt
   
   # Runtime stage
   FROM nvidia/cuda:12.1-runtime-ubuntu22.04
   
   WORKDIR /app
   COPY --from=builder /root/.local /root/.local
   COPY src/ ./src/
   COPY scripts/ ./scripts/
   
   ENV PATH=/root/.local/bin:$PATH
   EXPOSE 8000
   
   CMD ["python", "scripts/nexus_server.py"]

Environment Variables
---------------------

+----------------------------+------------------------------------------+
| Variable                   | Description                              |
+============================+==========================================+
| ``NEXUS_MODEL_PATH``       | Path to model weights                    |
+----------------------------+------------------------------------------+
| ``NEXUS_BATCH_SIZE``       | Inference batch size                     |
+----------------------------+------------------------------------------+
| ``NEXUS_GPU_MEMORY``       | GPU memory limit (GB)                    |
+----------------------------+------------------------------------------+
| ``REDIS_URL``              | Redis connection URL                     |
+----------------------------+------------------------------------------+
| ``JAEGER_ENDPOINT``        | Jaeger tracing endpoint                  |
+----------------------------+------------------------------------------+

Health Checks
-------------

.. code-block:: bash

   docker exec nexus python -c "from src.utils.health import check_health; print(check_health())"

Logs
----

.. code-block:: bash

   docker logs -f nexus
   docker-compose logs -f nexus

Troubleshooting
---------------

**GPU not detected:**

Ensure NVIDIA Docker runtime is installed:

.. code-block:: bash

   docker run --gpus all nvidia/cuda:12.1-base nvidia-smi

**Out of memory:**

Reduce batch size in config:

.. code-block:: bash

   docker run --gpus all -e NEXUS_BATCH_SIZE=1 nexus:latest
