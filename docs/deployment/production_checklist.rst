.. _production_checklist:

Production Deployment Checklist
================================

Use this checklist before deploying Nexus to production.

Pre-Deployment
--------------

Infrastructure
~~~~~~~~~~~~~~

- [ ] GPU nodes provisioned and tested
- [ ] Network security groups configured
- [ ] Load balancer SSL certificates installed
- [ ] DNS records configured
- [ ] Backup storage configured (S3/GCS/Azure Blob)
- [ ] Monitoring stack deployed (Prometheus/Grafana)
- [ ] Log aggregation configured
- [ ] Alerting rules defined

Security
~~~~~~~~

- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] Input validation implemented
- [ ] Secrets management configured (Kubernetes secrets/AWS Secrets Manager)
- [ ] Network policies defined
- [ ] Pod security policies enabled
- [ ] Container scanning completed
- [ ] Penetration testing performed

Performance
~~~~~~~~~~~

- [ ] Load testing completed
- [ ] Auto-scaling policies configured
- [ ] Resource limits set (CPU/Memory/GPU)
- [ ] Caching layer configured (Redis)
- [ ] CDN configured for static assets
- [ ] Database connection pooling configured

Deployment
----------

Container Image
~~~~~~~~~~~~~~~

- [ ] Multi-stage Dockerfile optimized
- [ ] Image scanned for vulnerabilities
- [ ] Image tagged with version
- [ ] Image pushed to registry
- [ ] Image pull secrets configured

Configuration
~~~~~~~~~~~~~

- [ ] Environment-specific configs created
- [ ] Secrets injected securely
- [ ] Feature flags configured
- [ ] Debug mode disabled
- [ ] Logging level set to INFO or WARN

Health Checks
~~~~~~~~~~~~~

- [ ] Liveness probe configured
- [ ] Readiness probe configured
- [ ] Startup probe configured (if needed)
- [ ] Health check endpoints tested

Post-Deployment
---------------

Verification
~~~~~~~~~~~~

- [ ] All pods running and ready
- [ ] Services accessible via load balancer
- [ ] SSL certificates valid
- [ ] DNS resolution working
- [ ] Health checks passing
- [ ] Metrics flowing to monitoring
- [ ] Logs appearing in aggregation system

Testing
~~~~~~~

- [ ] Smoke tests passed
- [ ] Integration tests passed
- [ ] Performance tests passed
- [ ] Failover tests completed
- [ ] Rollback procedure tested

Documentation
~~~~~~~~~~~~~

- [ ] Runbook created
- [ ] Incident response plan documented
- [ ] On-call rotation configured
- [ ] Escalation procedures defined
- [ ] Contact information updated

Capacity Planning
-----------------

+------------------+--------------+--------------+--------------+
| Metric           | Current      | Threshold    | Action       |
+==================+==============+==============+==============+
| GPU Utilization  | < 70%        | > 85%        | Scale up     |
+------------------+--------------+--------------+--------------+
| Memory Usage     | < 70%        | > 85%        | Scale up     |
+------------------+--------------+--------------+--------------+
| Request Latency  | < 200ms      | > 500ms      | Optimize     |
+------------------+--------------+--------------+--------------+
| Error Rate       | < 0.1%       | > 1%         | Investigate  |
+------------------+--------------+--------------+--------------+

Rollback Plan
-------------

1. **Immediate:**
   
   .. code-block:: bash

      kubectl rollout undo deployment/nexus

2. **Database:**
   
   - Restore from latest backup
   - Verify data integrity

3. **Communication:**
   
   - Notify stakeholders
   - Update status page
   - Create incident report

Monitoring Dashboards
---------------------

Key metrics to monitor:

* Request rate and latency
* Error rate (4xx, 5xx)
* GPU utilization
* Memory usage
* Queue depth
* Cache hit rate

Recommended alerting rules:

* High error rate (> 5% for 5 minutes)
* High latency (p99 > 1s for 10 minutes)
* GPU OOM errors
* Pod restart loops
* Disk space > 80%

Compliance
----------

- [ ] Data retention policies configured
- [ ] Audit logging enabled
- [ ] GDPR/privacy compliance verified
- [ ] SOC 2 requirements met (if applicable)
- [ ] HIPAA compliance verified (if applicable)

Sign-off
--------

| Role           | Name | Date | Signature |
|----------------|------|------|-----------|
| Tech Lead      |      |      |           |
| Security Lead  |      |      |           |
| DevOps Lead    |      |      |           |
| Product Owner  |      |      |           |
