# VM 신규 세팅 Runbook

새 GCE VM에 모델 서버를 세팅하는 절차.

---

## 전제 조건

- GCP 프로젝트 및 Artifact Registry(`asia-northeast3-docker.pkg.dev`) 설정 완료
- GitHub Secrets 등록 완료
  - `GCP_PROJECT_ID`
  - `GCP_WORKLOAD_IDENTITY_PROVIDER`
  - `GCP_SERVICE_ACCOUNT_EMAIL`
- 사용할 도메인의 A 레코드가 VM 공개 IP를 가리키고 있을 것

---

## Step 1. GCE VM 생성

GCP Console 또는 gcloud CLI로 생성.

```bash
gcloud compute instances create sign-language-model \
  --zone=asia-northeast3-a \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --tags=http-server,https-server
```

> VM 이름(`sign-language-model`)과 존(`asia-northeast3-a`)은 `deploy.yml`에 하드코딩되어 있으므로 동일하게 사용.

---

## Step 2. GCP 방화벽 규칙 확인

아래 두 규칙이 없으면 생성.

```bash
# HTTP (Certbot 인증 및 HTTPS 리다이렉트용)
gcloud compute firewall-rules create allow-http \
  --direction=INGRESS --action=ALLOW \
  --rules=tcp:80 --target-tags=http-server

# HTTPS (실제 서비스)
gcloud compute firewall-rules create allow-https \
  --direction=INGRESS --action=ALLOW \
  --rules=tcp:443 --target-tags=https-server
```

> 포트 8000은 열지 않는다. Nginx가 앞에서 받아 localhost:8000으로 전달.

---

## Step 3. 도메인 DNS 확인

```bash
dig <YOUR_DOMAIN> +short
# → VM 공개 IP가 나와야 함
```

DNS가 전파되지 않으면 Certbot SSL 발급이 실패하므로 반드시 먼저 확인.

---

## Step 4. VM SSH 접속

```bash
gcloud compute ssh sign-language-model \
  --zone=asia-northeast3-a \
  --tunnel-through-iap
```

---

## Step 5. Docker 설치

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
```

---

## Step 6. 앱 디렉토리 및 모델 파일 준비

```bash
sudo mkdir -p /opt/sign-language-korean/models/gloss_models
sudo touch /opt/sign-language-korean/.last-deployed-image
sudo chown -R ubuntu:ubuntu /opt/sign-language-korean
```

모델 파일은 CI/CD 파이프라인이 구성되어 있으니 배포에 참고

---

## Step 7. Nginx + Certbot 설치 및 HTTP 설정

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

HTTP 블록만 먼저 적용 (Certbot 인증서 발급 전에 443 블록을 올리면 오류 발생):

```bash
sudo tee /etc/nginx/sites-available/sign-language-model > /dev/null << 'EOF'
server {
    listen 80;
    server_name <YOUR_DOMAIN>;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/sign-language-model \
            /etc/nginx/sites-enabled/sign-language-model
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl enable nginx && sudo systemctl start nginx
```

---

## Step 8. SSL 인증서 발급 (Certbot)

```bash
sudo certbot --nginx -d <YOUR_DOMAIN> \
  --non-interactive --agree-tos -m admin@<YOUR_DOMAIN>
```

> Let's Encrypt는 `kro.kr` 같은 공유 도메인에 주당 50개 발급 제한이 있음.
> 실패 시 잠시 대기 후 재시도하거나 다른 도메인 서비스(duckdns.org 등) 사용.

---

## Step 9. Nginx 프록시 설정 추가

Certbot이 SSL을 추가한 뒤, 프록시 블록을 포함한 최종 설정으로 덮어씀:

```bash
sudo tee /etc/nginx/sites-available/sign-language-model > /dev/null << 'EOF'
server {
    listen 80;
    server_name <YOUR_DOMAIN>;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name <YOUR_DOMAIN>;

    ssl_certificate /etc/letsencrypt/live/<YOUR_DOMAIN>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<YOUR_DOMAIN>/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 10s;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
EOF

sudo nginx -t && sudo systemctl reload nginx
```

---

## Step 10. 최초 배포 트리거

`main` 브랜치에 푸시하면 GitHub Actions가 자동으로:
1. Docker 이미지 빌드 → Artifact Registry 푸시
2. VM에 IAP SSH로 접속
3. `redeploy.sh` 실행 (이미지 pull → 컨테이너 교체 → 헬스체크)

```bash
git push origin main
```

---

## Step 11. 동작 확인

```bash
# VM에서
curl https://<YOUR_DOMAIN>/health

# 로컬 PowerShell
Test-NetConnection <YOUR_DOMAIN> -Port 443

# 로컬 터미널
nc -zv <YOUR_DOMAIN> 443
```

