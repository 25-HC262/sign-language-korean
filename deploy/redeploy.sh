#!/bin/bash
# VM에서 GitHub Actions가 IAP SSH로 호출하는 재배포 스크립트
# 사용법: bash redeploy.sh <image:tag>
set -euo pipefail

IMAGE="${1:?사용법: redeploy.sh <image:tag>}"
CONTAINER_NAME="sign-language-korean"
MODEL_DIR="/opt/sign-language-korean/models"
PORT="8000"
LAST_IMAGE_FILE="/opt/sign-language-korean/.last-deployed-image"

echo "==> [1/5] Artifact Registry 인증"
gcloud auth configure-docker asia-northeast3-docker.pkg.dev --quiet

echo "==> [2/5] 새 이미지 pull: $IMAGE"
docker pull "$IMAGE"

echo "==> [3/5] GCS에서 모델 파일 다운로드"
mkdir -p "${MODEL_DIR}/gloss_models"
gsutil cp gs://trout-models/gloss_models/gloss_transformer_2026_03_25_08-55.keras \
  "${MODEL_DIR}/gloss_models/gloss_transformer_2026_03_25_08-55.keras"

echo "==> [4/5] 기존 컨테이너 중지 및 제거"
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm   "$CONTAINER_NAME" 2>/dev/null || true

echo "==> [5/5] 새 컨테이너 시작"
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 127.0.0.1:${PORT}:8000 \
  -v "${MODEL_DIR}:/app/models" \
  -e KSL_VISUALIZE=1 \
  --memory="6g" \
  --cpus="2" \
  "$IMAGE"

echo "==> [5/5] 헬스체크 (최대 120초 대기, 모델 로딩 포함)"
for i in $(seq 1 24); do
  sleep 5
  # 컨테이너가 여전히 실행 중인지 확인
  if ! docker ps --filter "name=${CONTAINER_NAME}" --filter "status=running" --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: 컨테이너가 실행 중이 아닙니다. 로그:"
    docker logs "$CONTAINER_NAME" --tail 30
    break
  fi
  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health 2>/dev/null) || true
  if [ "$HTTP_STATUS" = "200" ]; then
    echo "헬스체크 통과 ($((i * 5))초 경과)"
    echo "$IMAGE" > "$LAST_IMAGE_FILE"
    echo "배포 완료: $IMAGE"
    exit 0
  fi
  echo "  대기 중... ($((i * 5))초 경과, HTTP 상태: ${HTTP_STATUS:-000})"
done

# 헬스체크 실패 → 이전 이미지로 롤백
echo "ERROR: 헬스체크 실패 (120초 초과). 롤백을 시도합니다."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm   "$CONTAINER_NAME" 2>/dev/null || true

PREVIOUS_IMAGE=$(cat "$LAST_IMAGE_FILE" 2>/dev/null || echo "")
if [ -n "$PREVIOUS_IMAGE" ]; then
  echo "이전 이미지로 롤백: $PREVIOUS_IMAGE"
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p 127.0.0.1:${PORT}:8000 \
    -v "${MODEL_DIR}:/app/models" \
    --memory="6g" \
    --cpus="2" \
    "$PREVIOUS_IMAGE"
  echo "롤백 완료. 파이프라인을 실패 처리합니다."
else
  echo "롤백할 이전 이미지가 없습니다."
fi
exit 1
