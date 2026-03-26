"""
ISLR (Isolated Sign Language Recognition) 추론 서버.

sign-language/src/ 의 백본 구조를 sign-language-korean 인프라에서 제공합니다.
MediaPipe Tasks API (face/pose/hand landmarker)를 재사용해 (543, 3) 랜드마크를 추출합니다.

실행:
  ISLR_MODEL=models/islr/islr-fp16-192-8-seed42-fold0-best.h5 \
  SERVER_MODULE=main_islr uvicorn main_islr:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.backbones.islr_backbone import get_islr_model, ISLRTFLiteModel, ISLR_ROWS_PER_FRAME
from load_data.islr_inference import extract_coordinates_for_islr

logger = logging.getLogger("islr")
logging.basicConfig(level=logging.INFO)

PREDICTION_STRIDE = 30   # sign-language SEQ_LEN과 동일 (30프레임마다 예측)
SMOOTHING_WINDOW = 3
THRESHOLD = 0.5

# ── 라벨 맵 ──────────────────────────────────────────────────────────────────
_SIGN_MAP_PATH = os.path.join(os.path.dirname(__file__), "src", "islr_sign_map.json")
with open(_SIGN_MAP_PATH, "r") as f:
    _sign_to_idx = json.load(f)
idx_to_label = {v: k for k, v in _sign_to_idx.items()}

# ── MediaPipe Tasks API 초기화 ────────────────────────────────────────────────
pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
    mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="/app/pose_landmarker_full.task"),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="/app/hand_landmarker.task"),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
)

_face_model_path = "/app/face_landmarker.task"
face_landmarker = None
if os.path.exists(_face_model_path):
    try:
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_face_model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
        )
        print("[*] FaceLandmarker 초기화 완료")
    except Exception as e:
        print(f"[!] FaceLandmarker 초기화 실패: {e}")
else:
    print(f"[!] face_landmarker.task 없음. 얼굴 키포인트 비활성화.")

# ── ISLR 모델 로드 ─────────────────────────────────────────────────────────────
_default_model_path = "models/islr/islr-fp16-192-8-seed42-fold0-best.h5"
ISLR_MODEL_PATH = os.environ.get("ISLR_MODEL", _default_model_path)

print(f"[*] ISLR 모델 로딩 중: {ISLR_MODEL_PATH}")
tf.get_logger().setLevel("ERROR")

model = None
try:
    _backbone = get_islr_model()
    _backbone.load_weights(ISLR_MODEL_PATH, by_name=True, skip_mismatch=True)
    model = ISLRTFLiteModel(islr_models=[_backbone])
    print(f"[*] ISLR 모델 로딩 완료. 클래스 수: {len(idx_to_label)}")
except Exception as e:
    print(f"[!] ISLR 모델 로딩 실패: {e}")

# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "islr",
        "model_loaded": model is not None,
        "face_landmarker": face_landmarker is not None,
        "num_classes": len(idx_to_label),
    }


def _init_user(uid, seqs, last_label, frame_cnt, prob_hist):
    if uid not in seqs:
        seqs[uid]       = deque(maxlen=PREDICTION_STRIDE)
        last_label[uid] = ""
        frame_cnt[uid]  = 0
        prob_hist[uid]  = deque(maxlen=SMOOTHING_WINDOW)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[*] WebSocket 연결됨 (ISLR 모드)")

    seqs:       dict[str, deque] = {}
    last_label: dict[str, str]   = {}
    frame_cnt:  dict[str, int]   = {}
    prob_hist:  dict[str, deque] = {}

    try:
        while True:
            message = await websocket.receive()

            # JSON 제어 메시지
            if message.get("text"):
                try:
                    ctrl = json.loads(message["text"])
                    uid  = ctrl.get("userId")
                    if ctrl.get("type") == "stream_config" and uid:
                        _init_user(uid, seqs, last_label, frame_cnt, prob_hist)
                    elif ctrl.get("type") == "stop_stream" and uid:
                        for d in (seqs, last_label, frame_cnt, prob_hist):
                            d.pop(uid, None)
                except Exception as e:
                    print(f"제어 메시지 파싱 오류: {e}")
                continue

            # 바이너리 프레임
            raw = message.get("bytes")
            if not raw or len(raw) < 4:
                continue

            user_id_len = int.from_bytes(raw[:4], "big")
            if len(raw) < 4 + user_id_len:
                continue
            user_id    = raw[4:4 + user_id_len].decode("utf-8")
            frame_data = raw[4 + user_id_len:]

            if user_id not in seqs:
                _init_user(user_id, seqs, last_label, frame_cnt, prob_hist)

            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_result = pose_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image) if face_landmarker else None

            coords = extract_coordinates_for_islr(pose_result, hand_result, face_result)
            seqs[user_id].append(coords)
            frame_cnt[user_id] += 1

            # PREDICTION_STRIDE 프레임마다 예측
            if (len(seqs[user_id]) == PREDICTION_STRIDE
                    and model is not None
                    and frame_cnt[user_id] % PREDICTION_STRIDE == 0):
                try:
                    seq_arr = np.array(list(seqs[user_id]), dtype=np.float32)  # (30, 543, 3)
                    prediction = model(seq_arr)["outputs"].numpy()              # (250,)

                    prob_hist[user_id].append(prediction)
                    smoothed   = np.mean(list(prob_hist[user_id]), axis=0)
                    confidence = float(np.max(smoothed))
                    pred_idx   = int(np.argmax(smoothed))

                    if confidence >= THRESHOLD:
                        pred_sign = idx_to_label.get(pred_idx, "unknown")
                        if pred_sign != last_label[user_id]:
                            last_label[user_id] = pred_sign
                            await websocket.send_json({"userId": user_id, "text": pred_sign})

                except Exception as e:
                    print(f"[{user_id}] 예측 오류: {e}")
                    await websocket.send_json({"userId": user_id, "text": "예측 중 오류 발생"})

    except WebSocketDisconnect:
        print("[*] WebSocket 연결 종료 (ISLR)")
    except Exception as e:
        print(f"[!] 오류: {e}")
