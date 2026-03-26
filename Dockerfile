FROM python:3.11-slim

# MediaPipe/OpenCV 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MediaPipe Tasks 모델 파일 다운로드
RUN curl -fsSL \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
    -o /app/pose_landmarker_full.task && \
    curl -fsSL \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
    -o /app/hand_landmarker.task && \
    curl -fsSL \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
    -o /app/face_landmarker.task

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main_asl_finetune:app", "--host", "0.0.0.0", "--port", "8000"]
