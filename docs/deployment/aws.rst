.. _aws_deployment:

AWS Deployment
==============

This guide covers deploying Nexus on Amazon Web Services.

Architecture Overview
---------------------

.. code-block:: text

   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │   Route 53  │────▶|   ALB       │────▶|   EKS       │
   └─────────────┘     └─────────────┘     │  (Nexus)    │
                                           └─────────────┘
                                                  │
                                           ┌─────────────┐
                                           │  EFS/S3     │
                                           │ (Models)    │
                                           └─────────────┘

Prerequisites
-------------

* AWS CLI configured
* eksctl installed
* kubectl configured
* Docker installed

EKS Cluster Setup
-----------------

Create cluster with GPU nodes:

.. code-block:: bash

   eksctl create cluster \
     --name nexus-cluster \
     --region us-west-2 \
     --node-type g4dn.xlarge \
     --nodes 2 \
     --nodes-min 1 \
     --nodes-max 4 \
     --managed

Enable GPU support:

.. code-block:: bash

   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

ECR Repository
--------------

Create and push image:

.. code-block:: bash

   # Create repository
   aws ecr create-repository --repository-name nexus --region us-west-2
   
   # Login
   aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-west-2.amazonaws.com
   
   # Build and push
   docker build -t nexus:latest .
   docker tag nexus:latest <account>.dkr.ecr.us-west-2.amazonaws.com/nexus:latest
   docker push <account>.dkr.ecr.us-west-2.amazonaws.com/nexus:latest

S3 for Model Storage
--------------------

.. code-block:: bash

   # Create bucket
   aws s3 mb s3://nexus-models-bucket --region us-west-2
   
   # Upload models
   aws s3 sync ./models s3://nexus-models-bucket/models/

IAM Policy for S3 Access:

.. code-block:: json

   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "s3:GetObject",
           "s3:ListBucket"
         ],
         "Resource": [
           "arn:aws:s3:::nexus-models-bucket",
           "arn:aws:s3:::nexus-models-bucket/*"
         ]
       }
     ]
   }

Application Load Balancer
--------------------------

Deploy ALB Ingress Controller:

.. code-block:: bash

   helm repo add eks https://aws.github.io/eks-charts
   helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
     --set clusterName=nexus-cluster \
     --set serviceAccount.create=true

Ingress with ALB:

.. code-block:: yaml

   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: nexus-ingress
     annotations:
       alb.ingress.kubernetes.io/scheme: internet-facing
       alb.ingress.kubernetes.io/target-type: ip
       alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:...
   spec:
     ingressClassName: alb
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

Auto-scaling Configuration
--------------------------

Cluster Autoscaler:

.. code-block:: bash

   helm install cluster-autoscaler stable/cluster-autoscaler \
     --set autoDiscovery.clusterName=nexus-cluster \
     --set awsRegion=us-west-2

Cost Optimization
-----------------

**Spot Instances:**

.. code-block:: yaml

   nodeGroups:
     - name: gpu-spot
       instanceType: g4dn.xlarge
       spot: true
       minSize: 0
       maxSize: 10

**S3 Intelligent-Tiering:**

.. code-block:: bash

   aws s3api put-bucket-intelligent-tiering-configuration \
     --bucket nexus-models-bucket \
     --id nexus-tiering \
     --intelligent-tiering-configuration file://tiering-config.json

Monitoring with CloudWatch
--------------------------

.. code-block:: bash

   # Install CloudWatch agent
   kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-daemonset.yaml

Security Best Practices
-----------------------

* Use VPC endpoints for S3 access
* Enable encryption at rest (KMS)
* Configure security groups
* Use IAM roles for service accounts
* Enable CloudTrail

Terraform Deployment
--------------------

See ``deployment/terraform/aws/`` for complete infrastructure as code.

.. code-block:: bash

   cd deployment/terraform/aws
   terraform init
   terraform plan
   terraform apply
