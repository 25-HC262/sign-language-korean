#!/bin/bash
# Ubuntu 24.04 LTS GCE VM 최초 1회 설정 스크립트 (Docker 기반)
# 사용법: sudo bash setup.sh <도메인>
# 예시:   sudo bash setup.sh model.example.com
set -euo pipefail

DOMAIN="${1:-}"
APP_DIR="/opt/sign-language-korean"
SERVICE_USER="ubuntu"

if [ -z "$DOMAIN" ]; then
    echo "사용법: sudo bash setup.sh <도메인>"
    echo "예시:   sudo bash setup.sh model.example.com"
    exit 1
fi

echo "==> [1/6] 시스템 패키지 설치"
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release nginx certbot python3-certbot-nginx

echo "==> [2/6] Docker Engine 설치 (공식 Docker 레포)"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

systemctl enable docker
systemctl start docker
usermod -aG docker "$SERVICE_USER"
echo "Docker 설치 완료 (버전: $(docker --version))"

echo "==> [3/6] 앱 디렉토리 및 모델 디렉토리 생성"
mkdir -p "$APP_DIR/models/gloss_models"
mkdir -p "$APP_DIR/deploy"
touch "$APP_DIR/.last-deployed-image"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR"
echo "주의: 배포 전에 모델 파일을 복사하세요:"
echo "  $APP_DIR/models/gloss_models/<model>.keras"

echo "==> [4/6] 배포 스크립트 복사 및 실행 권한 부여"
cp "$(dirname "$0")/redeploy.sh" "$APP_DIR/deploy/"
chmod +x "$APP_DIR/deploy/redeploy.sh"

echo "==> [5/6] Nginx 설정"
cp "$(dirname "$0")/nginx-model.conf" /etc/nginx/sites-available/sign-language-model
sed -i "s/YOUR_DOMAIN/$DOMAIN/g" /etc/nginx/sites-available/sign-language-model
ln -sf /etc/nginx/sites-available/sign-language-model \
       /etc/nginx/sites-enabled/sign-language-model
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable nginx
systemctl reload nginx

echo "==> [6/6] SSL 인증서 발급 (Certbot)"
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "admin@$DOMAIN"
systemctl reload nginx

echo ""
echo "VM 초기 설정 완료!"
echo "  다음 단계:"
echo "  1. 모델 파일 복사: $APP_DIR/models/gloss_models/"
echo "  2. GitHub Secrets 등록 (GCP_WORKLOAD_IDENTITY_PROVIDER, GCP_SERVICE_ACCOUNT_EMAIL, GCP_PROJECT_ID)"
echo "  3. main 브랜치에 푸시하면 자동 배포 시작"
echo "  WebSocket 엔드포인트: wss://$DOMAIN/ws"
