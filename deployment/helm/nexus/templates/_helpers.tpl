{{/*
Expand the name of the chart.
*/}}
{{- define "nexus.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "nexus.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "nexus.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "nexus.labels" -}}
helm.sh/chart: {{ include "nexus.chart" . }}
{{ include "nexus.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "nexus.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nexus.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "nexus.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nexus.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL fullname
*/}}
{{- define "nexus.postgresql.fullname" -}}
{{- printf "%s-postgres" (include "nexus.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Redis fullname
*/}}
{{- define "nexus.redis.fullname" -}}
{{- printf "%s-redis-master" (include "nexus.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
MinIO fullname
*/}}
{{- define "nexus.minio.fullname" -}}
{{- printf "%s-minio" (include "nexus.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Database connection string
*/}}
{{- define "nexus.database.url" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s:%s@%s:5432/%s" 
    .Values.postgresql.auth.username 
    .Values.postgresql.auth.password 
    (include "nexus.postgresql.fullname" .) 
    .Values.postgresql.auth.database }}
{{- else }}
{{- required "A valid external database URL is required when postgresql is disabled" .Values.config.databaseUrl }}
{{- end }}
{{- end }}

{{/*
Redis connection string
*/}}
{{- define "nexus.redis.url" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://:%s@%s:6379/0" 
    .Values.redis.auth.password 
    (include "nexus.redis.fullname" .) }}
{{- else }}
{{- required "A valid external Redis URL is required when redis is disabled" .Values.config.redisUrl }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "nexus.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- else if .Values.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.imagePullSecrets }}
  - name: {{ .name }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Pod labels
*/}}
{{- define "nexus.podLabels" -}}
{{- include "nexus.selectorLabels" . }}
{{- with .Values.podLabels }}
{{- toYaml . }}
{{- end }}
{{- end }}

{{/*
Pod annotations
*/}}
{{- define "nexus.podAnnotations" -}}
{{- with .Values.podAnnotations }}
{{- toYaml . }}
{{- end }}
{{- if .Values.monitoring.enabled }}
prometheus.io/scrape: "true"
prometheus.io/port: "{{ .Values.service.port }}"
prometheus.io/path: "/metrics"
{{- end }}
{{- end }}

{{/*
Storage class
*/}}
{{- define "nexus.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- else if .Values.persistence.storageClass }}
{{- .Values.persistence.storageClass }}
{{- else }}
{{- default "" }}
{{- end }}
{{- end }}

{{/*
Config checksum
*/}}
{{- define "nexus.configChecksum" -}}
{{- include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- end }}

{{/*
Secrets checksum
*/}}
{{- define "nexus.secretsChecksum" -}}
{{- if and .Values.secrets.enabled .Values.secrets.data }}
{{- include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
{{- else }}
{{- "" }}
{{- end }}
{{- end }}

{{/*
Validation helpers
*/}}
{{- define "nexus.validate.required" -}}
{{- if not . }}
{{- fail "Required value is missing" }}
{{- end }}
{{- end }}

{{/*
Resource request/limits validation
*/}}
{{- define "nexus.validate.resources" -}}
{{- if and .Values.resources.limits.cpu .Values.resources.requests.cpu }}
{{- if gt (int (trimSuffix "m" .Values.resources.requests.cpu)) (int (trimSuffix "m" .Values.resources.limits.cpu)) }}
{{- fail "CPU request cannot be greater than CPU limit" }}
{{- end }}
{{- end }}
{{- end }}
