# Xagent 外网反向代理（HTTP + HTTPS）

将本机 xagent（localhost:80）暴露到公网两个端口，便于外网访问。

- **HTTP**: 端口 **8088** → `http://36.139.106.105:8088`
- **HTTPS**: 端口 **9443** → `https://36.139.106.105:9443`

## 前置条件

1. **xagent 已启动**并监听本机 80（例如 `docker compose up -d`）。
2. **云主机安全组/防火墙**已放行 **8088**（TCP）和 **9443**（TCP）。

## 一键启动

```bash
cd /home/shenniao/xagent/xagent/nginx-proxy
./start.sh
```

首次运行会自动生成自签名 SSL 证书（`ssl/cert.pem`、`ssl/key.pem`）。浏览器访问 HTTPS 会提示“不安全”，属正常；正式环境可用 certbot 替换为 Let's Encrypt 证书。

## 停止

```bash
./stop.sh
```

## 端口说明

| 端口 | 协议 | 说明 |
|------|------|------|
| 8088 | HTTP | 明文访问 |
| 9443 | HTTPS | TLS 加密（默认自签名证书） |

后端统一转发到 **127.0.0.1:80**（xagent）。

## 正式环境 HTTPS 证书

若已有一对证书（如 certbot 生成），可将 `ssl/cert.pem`、`ssl/key.pem` 替换为你的证书与私钥，或修改 `xagent-proxy.conf` 中 HTTPS server 的 `ssl_certificate` / `ssl_certificate_key` 路径后，重新执行 `./start.sh`（或先 `./stop.sh` 再 `./start.sh`）。
