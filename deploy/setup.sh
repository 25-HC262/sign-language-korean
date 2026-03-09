#!/bin/bash
# Ubuntu 24.04 LTS VM에 sign-language-korean 모델 서버 배포 스크립트
# 사용법: sudo bash setup.sh <도메인> (예: sudo bash setup.sh model.example.com)
set -euo pipefail

DOMAIN="${1:-}"
APP_DIR="/opt/sign-language-korean"
SERVICE_USER="ubuntu"
PYTHON="python3.11"
VENV_DIR="$APP_DIR/venv"

if [ -z "$DOMAIN" ]; then
    echo "사용법: sudo bash setup.sh <도메인>"
    echo "예시:   sudo bash setup.sh model.example.com"
    exit 1
fi

echo "==> [1/6] 시스템 패키지 설치"
apt-get update
apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    nginx certbot python3-certbot-nginx \
    git

echo "==> [2/6] 앱 디렉토리 설정"
mkdir -p "$APP_DIR"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR"

echo "==> [3/6] Python 가상환경 및 의존성 설치"
sudo -u "$SERVICE_USER" "$PYTHON" -m venv "$VENV_DIR"
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "==> [4/6] systemd 서비스 등록"
cp "$(dirname "$0")/sign-language-model.service" /etc/systemd/system/
# 도메인을 서비스 파일에 반영할 필요가 없으므로 바로 활성화
systemctl daemon-reload
systemctl enable sign-language-model
systemctl start sign-language-model

echo "==> [5/6] Nginx 설정"
cp "$(dirname "$0")/nginx-model.conf" /etc/nginx/sites-available/sign-language-model
sed -i "s/YOUR_DOMAIN/$DOMAIN/g" /etc/nginx/sites-available/sign-language-model
ln -sf /etc/nginx/sites-available/sign-language-model /etc/nginx/sites-enabled/sign-language-model
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

echo "==> [6/6] SSL 인증서 발급 (Certbot)"
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "admin@$DOMAIN"
systemctl reload nginx

echo ""
echo "배포 완료!"
echo "  WebSocket 엔드포인트: wss://$DOMAIN/ws"
echo "  서비스 상태 확인: systemctl status sign-language-model"
echo "  로그 확인: journalctl -u sign-language-model -f"
