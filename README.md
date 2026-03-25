# 한국 수어 인식 시스템 (Korean Sign Language Recognition)

실시간 영상 스트리밍을 통해 한국 수어를 인식하고 텍스트로 변환하는 딥러닝 기반 시스템입니다.

---

## 기술 스택

### 딥러닝 프레임워크

| 기술 | 버전 | 용도 |
|------|------|------|
| **TensorFlow** | 2.20.0 | 메인 딥러닝 프레임워크, 모델 학습 및 추론 |
| **Keras** | 3.11.3 | 고수준 신경망 API |
| **PyTorch** | 2.1+ | Mamba SSM 아키텍처 실험용 |

혼합 정밀도 학습(FP16)을 지원하여 GPU 연산을 가속화합니다.

### 포즈 추정 및 컴퓨터 비전

| 기술 | 버전 | 용도 |
|------|------|------|
| **MediaPipe** | 0.10.32 | 실시간 신체/손 랜드마크 감지 |
| **OpenCV** | 4.12.0 | 이미지 디코딩 및 전처리 |
| **Pillow** | 10+ | 이미지 처리 |

MediaPipe Tasks API를 사용하여 다음 랜드마크를 추출합니다:
- `PoseLandmarker`: 신체 키포인트 25개
- `HandLandmarker`: 손 랜드마크 21개 (양손 지원)

### 웹 서버 및 실시간 통신

| 기술 | 버전 | 용도 |
|------|------|------|
| **FastAPI** | 0.116.1 | REST API 및 WebSocket 서버 |
| **uvicorn** | 0.35.0 | ASGI 서버 |
| **websockets** | 12.0+ | 실시간 양방향 스트리밍 |

클라이언트로부터 JPEG 프레임을 WebSocket으로 수신하고, 인식 결과를 실시간으로 반환합니다.

### 데이터 처리 및 ML 유틸리티

| 기술 | 버전 | 용도 |
|------|------|------|
| **NumPy** | 1.26.0+ | 키포인트 행렬 연산 |
| **scikit-learn** | 1.4+ | 전처리 및 ML 유틸리티 |
| **pandas** | 2.1+ | 데이터 관리 |
| **umap-learn** | - | 차원 축소 (98차원 → 32차원) |
| **Optuna** | 3.6+ | 하이퍼파라미터 자동 탐색 |
| **WandB** | 0.16+ | 실험 추적 및 시각화 |
| **huggingface_hub** | 0.23+ | 모델 허브 연동 |

### 클라우드 스토리지

| 기술 | 용도 |
|------|------|
| **AWS S3** (boto3) | 데이터셋 저장 및 다운로드 |
| **Google Cloud Storage** | 데이터셋 저장 및 다운로드 |

### 배포 인프라

| 기술 | 용도 |
|------|------|
| **Docker** | 컨테이너화 (Python 3.11-slim 기반) |
| **GitHub Actions** | CI/CD 파이프라인 |
| **Google Compute Engine** | 모델 서버 호스팅 |
| **Google Artifact Registry** | Docker 이미지 저장소 |
| **Nginx** | 리버스 프록시 |
| **systemd** | 서비스 프로세스 관리 |

---

## 모델 아키텍처

### 1. Gloss Transformer (주력 모델)

`src/backbone.py`에 구현된 커스텀 아키텍처입니다.

```
Input (batch, 125 frames, 32 dims)
  └─ Dense Stem + BatchNorm
       └─ [Conv1D Block × 3 + Transformer Block] × N stages
            ├─ CausalDWConv1D  : 인과적 뎁스와이즈 시계열 컨볼루션
            ├─ ECA             : Efficient Channel Attention
            └─ MultiHeadSelfAttention + FFN
  └─ GlobalAveragePooling
  └─ Output (num_classes)
```

- **핵심 아이디어**: Conv1D 블록으로 지역적 시계열 패턴을 포착하고, Transformer 블록으로 장거리 의존성을 학습합니다.
- **LateDropout**: N 스텝 이후에 활성화되는 커스텀 드롭아웃으로 학습 안정성을 높입니다.

### 2. Mamba SSM (실험적)

`src/mamba_backbone.py`에 구현된 Selective State Space Model입니다.

- Mamba / Mamba2 논문 기반 순수 PyTorch 구현
- 인과적 컨볼루션과 선택적 상태 전이(selective state transition) 사용
- Transformer 대비 선형 복잡도로 긴 시퀀스 처리

### 3. UMAP 차원 축소 인코더

```
49 keypoints × 2D = 98 dims → UMAP Encoder → 32 dims
```

- 신체 7개 + 왼손 21개 + 오른손 21개 = 49 키포인트 선택
- 모델 입력 차원을 줄여 학습 효율 향상
- `models/umap_models/`에 별도 저장

---

## 추론 파이프라인

```
WebSocket 수신 (JPEG 프레임)
  └─ MediaPipe 랜드마크 추출 (신체 + 양손)
       └─ OpenPose 포맷 변환
            └─ 키포인트 정규화 (중심점 + 스케일)
                 └─ 49개 키포인트 선택 → UMAP 인코딩 (32차원)
                      └─ 60프레임 슬라이딩 윈도우 누적
                           └─ Gloss Transformer 분류
                                └─ 신뢰도 ≥ 0.5 → WebSocket 반환
```

- 다중 사용자 지원: 사용자 ID 기반 세션 관리
- 슬라이딩 윈도우 60프레임(약 2초) 단위로 수어 인식

---

## 인식 대상 수어 (5개 클래스)

| 레이블 | 수어 문장 |
|--------|-----------|
| NIA_SL_SEN0354 | 안녕하세요 |
| NIA_SL_SEN0355 | 감사합니다 |
| NIA_SL_SEN0356 | 죄송합니다 |
| NIA_SL_SEN0181 | 도와주세요 |
| NIA_SL_SEN2000 | 수고하셨습니다 |

데이터셋: [국립국어원 한국수어 말뭉치(NIA)](https://www.korean.go.kr/)

---

## 학습 설정

| 항목 | 기본값 |
|------|--------|
| 옵티마이저 | AdamW |
| 학습률 | 0.0001 |
| 배치 크기 | 32 |
| 최대 시퀀스 길이 | 125 프레임 |
| 에폭 | 158+ |
| L2 정규화 | 0.23+ |
| 조기 종료 | patience=10 |
| 학습률 감소 | ReduceLROnPlateau (patience=5, factor=0.5) |

하이퍼파라미터 탐색은 **Optuna**로 자동화하며, 실험 결과는 **Weights & Biases**로 추적합니다.

---

## 환경 설정

### 가상 환경 생성 (Windows)
```
python -m venv venv
.venv/Scripts/activate
pip install -r requirements-win.txt
```

### 가상 환경 생성 (macOS)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r documents/requirements-mac.txt
```

### 가상 환경 생성 (Linux)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-linux.txt
```

---

## 학습 진행

드라이브를 참고해서 root 경로에 `/data` 디렉터리를 만들고 `/train`, `/val` 데이터를 넣은 뒤 학습을 진행합니다.

- download/upload options: `-s`(storage), `-u`(upload)
- model options: `--gm`(gloss_model), `--gmt`(gloss_model_type), `--umap`
- training options: `--bs`(batch_size), `--lr`(learning_rate), `--epochs`, `--wd`(weight_decay), `--msl`(max_sequence_len)

```bash
python -m train.gloss_transformer_train -s L -u G --lr 0.0001178136471332758 --bs 32 --epochs 158 --wd 0.23042807878441396 --msl 281
```

---

## 배포

GitHub Actions가 `main` 브랜치 푸시 시 자동으로:
1. Docker 이미지 빌드 (MediaPipe 모델 파일 포함)
2. Google Artifact Registry에 푸시
3. GCE VM에 SSH 접속 (IAP 터널) 후 컨테이너 재배포
4. systemd 서비스로 컨테이너 수명 주기 관리

```bash
# 로컬 실행
docker build -t sign-language-korean .
docker run -p 8000:8000 sign-language-korean

# 헬스 체크
curl http://localhost:8000/health
```

---

## 프로젝트 구조

```
sign-language-korean/
├── main.py                    # FastAPI WebSocket 서버 (추론 진입점)
├── requirements.txt
├── Dockerfile
├── src/
│   ├── backbone.py            # Gloss Transformer 아키텍처
│   ├── mamba_backbone.py      # Mamba SSM 아키텍처
│   ├── utils.py               # 전처리 레이어 (Preprocess)
│   ├── config.py              # 설정, 하이퍼파라미터
│   └── primary_label_map.json
├── train/
│   ├── gloss_transformer_train.py
│   ├── mamba_train.py
│   └── best_params_search.py  # Optuna 하이퍼파라미터 탐색
├── load_data/
│   ├── create_dataset.py      # 데이터셋 생성 및 클라우드 동기화
│   └── inference.py           # 추론용 전처리
├── validation/                # 평가 및 테스트 스크립트
├── tools/                     # 시각화, 업로드 유틸리티
├── models/
│   ├── gloss_models/          # 학습된 분류 모델
│   └── umap_models/           # UMAP 인코더
├── deploy/                    # 배포 스크립트 (nginx, systemd)
└── .github/workflows/
    └── deploy.yml             # CI/CD 파이프라인
```
