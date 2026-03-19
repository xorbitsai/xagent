#!/usr/bin/env bash
# 生成自签名 SSL 证书（用于 HTTPS 8443）。正式环境请改用 Let's Encrypt 等。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSL_DIR="${SCRIPT_DIR}/ssl"
mkdir -p "$SSL_DIR"
CERT="${SSL_DIR}/cert.pem"
KEY="${SSL_DIR}/key.pem"
if [[ -f "$CERT" && -f "$KEY" ]]; then
  echo "证书已存在: $CERT"
  exit 0
fi
openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
  -keyout "$KEY" -out "$CERT" \
  -subj "/CN=36.139.106.105/O=Xagent"
echo "已生成自签名证书: $CERT"
echo "浏览器访问 https 会提示不安全，属正常；正式环境建议用 certbot 申请证书。"
