#!/usr/bin/env bash
# 启动 nginx 反向代理：将外网 8088(HTTP) / 9443(HTTPS) 转发到本机 xagent (127.0.0.1:80)
# 使用前请确保：1) xagent 已在运行并监听 80；2) 云主机已放行 8088、9443 端口。
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SSL_DIR="${SCRIPT_DIR}/ssl"
CERT="${SSL_DIR}/cert.pem"
KEY="${SSL_DIR}/key.pem"
GEN_CONF="${SCRIPT_DIR}/xagent-proxy.generated.conf"
PID_FILE="${SCRIPT_DIR}/nginx-proxy.pid"

# 生成自签名证书（若不存在）
"${SCRIPT_DIR}/gen-selfsigned.sh"

# 生成带实际证书路径的配置
sed -e "s|/REPLACE_SSL_CERT|${CERT}|g" -e "s|/REPLACE_SSL_KEY|${KEY}|g" \
  < "${SCRIPT_DIR}/xagent-proxy.conf" > "$GEN_CONF"

mkdir -p "${SCRIPT_DIR}/logs"
ERROR_LOG="${SCRIPT_DIR}/logs/nginx-proxy-error.log"

# 生成主配置（绝对路径，避免写 /var/log/nginx 权限错误）
MAIN_CONF="${SCRIPT_DIR}/nginx.conf.generated"
sed "s|__SCRIPT_DIR__|${SCRIPT_DIR}|g" < "${SCRIPT_DIR}/nginx.conf.template" > "$MAIN_CONF"

# 用 -g 在解析配置前指定 error_log，避免先尝试写 /var/log/nginx/error.log（Ubuntu nginx/1.18 不支持 -e 参数）
NGINX_OPTS=(-g "error_log ${ERROR_LOG};" -c "$MAIN_CONF")

# 若已在运行则 reload，否则 start
if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "nginx 代理已在运行 (pid=$PID)，执行 reload..."
    nginx "${NGINX_OPTS[@]}" -s reload
    echo "reload 完成。"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

echo "启动 nginx 反向代理（HTTP 8088, HTTPS 9443 -> 127.0.0.1:80）..."
nginx "${NGINX_OPTS[@]}"
echo "已启动。"
echo "  HTTP:  http://36.139.106.105:8088"
echo "  HTTPS: https://36.139.106.105:9443"
echo "停止: ${SCRIPT_DIR}/stop.sh"
