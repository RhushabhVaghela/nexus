.. _kubernetes_deployment:

Kubernetes Deployment
=====================

This guide covers deploying Nexus on Kubernetes clusters.

Prerequisites
-------------

* Kubernetes 1.25+
* kubectl configured
* Helm 3.0+ (optional)
* NVIDIA GPU Operator (for GPU nodes)

Namespace Setup
---------------

.. code-block:: bash

   kubectl create namespace nexus
   kubectl config set-context --current --namespace=nexus

Basic Deployment
----------------

Deployment manifest (``k8s/deployment.yaml``):

.. code-block:: yaml

   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: nexus
     labels:
       app: nexus
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: nexus
     template:
       metadata:
         labels:
           app: nexus
       spec:
         containers:
         - name: nexus
           image: nexus:latest
           ports:
           - containerPort: 8000
           resources:
             limits:
               nvidia.com/gpu: 1
               memory: "16Gi"
               cpu: "4"
             requests:
               memory: "8Gi"
               cpu: "2"
           volumeMounts:
           - name: model-storage
             mountPath: /app/models
           env:
           - name: NEXUS_BATCH_SIZE
             value: "4"
           - name: REDIS_URL
             value: "redis://redis:6379"
         volumes:
         - name: model-storage
           persistentVolumeClaim:
             claimName: nexus-models-pvc

Service Configuration
---------------------

.. code-block:: yaml

   apiVersion: v1
   kind: Service
   metadata:
     name: nexus
   spec:
     selector:
       app: nexus
     ports:
     - port: 80
       targetPort: 8000
     type: ClusterIP

GPU Node Selector
-----------------

.. code-block:: yaml

   spec:
     nodeSelector:
       nvidia.com/gpu.present: "true"
     tolerations:
     - key: nvidia.com/gpu
       operator: Exists
       effect: NoSchedule

Horizontal Pod Autoscaler
--------------------------

.. code-block:: yaml

   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: nexus-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: nexus
     minReplicas: 1
     maxReplicas: 5
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80

Ingress Configuration
---------------------

.. code-block:: yaml

   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: nexus-ingress
     annotations:
       nginx.ingress.kubernetes.io/rewrite-target: /
       cert-manager.io/cluster-issuer: "letsencrypt-prod"
   spec:
     tls:
     - hosts:
       - nexus.yourdomain.com
       secretName: nexus-tls
     rules:
     - host: nexus.yourdomain.com
       http:
         paths:
         - path: /
           pathType: Prefix
           backend:
             service:
               name: nexus
               port:
                 number: 80

Deploy with Helm
----------------

Create ``Chart.yaml``:

.. code-block:: yaml

   apiVersion: v2
   name: nexus
   description: Nexus AI Deployment
   version: 1.0.0
   appVersion: "6.1.0"

Install:

.. code-block:: bash

   helm install nexus ./helm-chart \
     --namespace nexus \
     --set replicaCount=2 \
     --set resources.gpu=1

Monitoring
----------

Prometheus ServiceMonitor:

.. code-block:: yaml

   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: nexus-metrics
     labels:
       release: prometheus
   spec:
     selector:
       matchLabels:
         app: nexus
     endpoints:
     - port: metrics
       interval: 30s

Troubleshooting
---------------

**Pod stuck in Pending:**

.. code-block:: bash

   kubectl describe pod nexus-xxx
   kubectl get events --sort-by=.metadata.creationTimestamp

**GPU not available:**

.. code-block:: bash

   kubectl get nodes -o json | jq '.items[].status.capacity'
