#!/bin/bash
set -euo pipefail
cd /home/erkin/pr-agent/banner-service
docker build -t aurora-banner-service:latest .
docker stop banner-service 2>/dev/null || true
docker rm banner-service 2>/dev/null || true
docker run -d \
  --name banner-service \
  --restart unless-stopped \
  -p 172.17.0.1:41928:41928 \
  aurora-banner-service:latest
echo 'banner-service running on 172.17.0.1:41928'
