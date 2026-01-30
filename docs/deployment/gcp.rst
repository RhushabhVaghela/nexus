.. _gcp_deployment:

GCP Deployment
==============

This guide covers deploying Nexus on Google Cloud Platform.

Architecture Overview
---------------------

.. code-block:: text

   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ Cloud DNS   │────▶| Cloud LB    │────▶| GKE         │
   └─────────────┘     └─────────────┘     │  (Nexus)    │
                                           └─────────────┘
                                                  │
                                           ┌─────────────┐
                                           │ Cloud       │
                                           │ Storage     │
                                           └─────────────┘

Prerequisites
-------------

* Google Cloud SDK installed
* kubectl configured
* Docker installed
* Billing enabled

GKE Cluster Setup
-----------------

Create cluster with GPU nodes:

.. code-block:: bash

   gcloud container clusters create nexus-cluster \
     --zone us-central1-a \
     --machine-type n1-standard-4 \
     --accelerator type=nvidia-tesla-t4,count=1 \
     --num-nodes 2 \
     --enable-autoscaling \
     --min-nodes 1 \
     --max-nodes 5

Install NVIDIA drivers:

.. code-block:: bash

   kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

Container Registry
------------------

Push image to GCR:

.. code-block:: bash

   # Configure Docker
   gcloud auth configure-docker
   
   # Tag and push
   docker tag nexus:latest gcr.io/PROJECT_ID/nexus:latest
   docker push gcr.io/PROJECT_ID/nexus:latest

Cloud Storage for Models
------------------------

Create bucket:

.. code-block:: bash

   gsutil mb -l us-central1 gs://nexus-models-bucket
   gsutil cp -r ./models/* gs://nexus-models-bucket/models/

IAM Configuration:

.. code-block:: bash

   gcloud iam service-accounts create nexus-sa \
     --display-name "Nexus Service Account"
   
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member "serviceAccount:nexus-sa@PROJECT_ID.iam.gserviceaccount.com" \
     --role "roles/storage.objectViewer"

Workloads Identity:

.. code-block:: yaml

   apiVersion: v1
   kind: ServiceAccount
   metadata:
     name: nexus-sa
     annotations:
       iam.gke.io/gcp-service-account: nexus-sa@PROJECT_ID.iam.gserviceaccount.com

Load Balancer Configuration
----------------------------

Deploy Ingress with GCE:

.. code-block:: yaml

   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: nexus-ingress
     annotations:
       kubernetes.io/ingress.class: gce
       kubernetes.io/ingress.global-static-ip-name: nexus-ip
       networking.gke.io/managed-certificates: nexus-cert
   spec:
     rules:
     - host: nexus.yourdomain.com
       http:
         paths:
         - path: /*
           pathType: ImplementationSpecific
           backend:
             service:
               name: nexus
               port:
                 number: 80

Managed Certificate:

.. code-block:: yaml

   apiVersion: networking.gke.io/v1
   kind: ManagedCertificate
   metadata:
     name: nexus-cert
   spec:
     domains:
       - nexus.yourdomain.com

Auto-scaling
------------

Horizontal Pod Autoscaler:

.. code-block:: bash

   kubectl autoscale deployment nexus \
     --cpu-percent=70 \
     --min=1 \
     --max=10

Cluster Autoscaler:

.. code-block:: bash

   gcloud container clusters update nexus-cluster \
     --enable-autoscaling \
     --min-nodes=1 \
     --max-nodes=10 \
     --zone=us-central1-a

Cloud Monitoring
----------------

Install monitoring:

.. code-block:: bash

   kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/prometheus-to-sd/kubernetes/prometheus-to-sd.yaml

Custom metrics:

.. code-block:: yaml

   apiVersion: monitoring.googleapis.com/v1
   kind: PodMonitoring
   metadata:
     name: nexus-metrics
   spec:
     selector:
       matchLabels:
         app: nexus
     endpoints:
     - port: metrics
       interval: 30s

Cost Optimization
-----------------

**Preemptible VMs:**

.. code-block:: bash

   gcloud container node-pools create gpu-preemptible \
     --cluster=nexus-cluster \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --preemptible \
     --num-nodes=0 \
     --enable-autoscaling \
     --min-nodes=0 \
     --max-nodes=10

**Cloud Storage Classes:**

.. code-block:: bash

   gsutil lifecycle set lifecycle.json gs://nexus-models-bucket

Security
--------

* Enable VPC Service Controls
* Use Workload Identity
* Enable Binary Authorization
* Configure Cloud Armor

Terraform
---------

See ``deployment/terraform/gcp/`` for IaC:

.. code-block:: bash

   cd deployment/terraform/gcp
   terraform init
   terraform apply
