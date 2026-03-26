"""
ASL 파인튜닝 모델용 추론 서버 (main.py 복사 후 수정).
변경 사항:
  - FaceLandmarker 추가 (얼굴 키포인트 추출)
  - asl_inference.py의 mediapipe_to_openpose_keypoints_with_face + asl_preprocess_sequence 사용
  - ASL 파인튜닝 모델 로드

실행:
  KSL_FINETUNE_MODEL=models/gloss_models/ksl_asl_finetuned_final.keras uvicorn main_asl_finetune:app
"""
import asyncio
import json
import logging
import os
import time
os.environ["KERAS_BACKEND"] = "tensorflow"
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import keras

from load_data.asl_inference import (
    mediapipe_to_openpose_keypoints_with_face,
    asl_preprocess_sequence,
)
from src.backbones.transformer_backbone import CausalDWConv1D, ECA, LateDropout, MultiHeadSelfAttention
from src.asl_finetune_config import CHANNELS_ASL, MAX_LEN
from src.config import SEQ_LEN, THRESHOLD, KSL_SENTENCES

logger = logging.getLogger("ksl_finetune")
logging.basicConfig(level=logging.INFO)

_debug = {"enabled": os.environ.get("KSL_DEBUG", "0") == "1"}

PREDICTION_COOLDOWN_SEC = 1.5
PREDICTION_STRIDE       = 15
SMOOTHING_WINDOW        = 3

# ── 이전 버전 Keras 호환 ──────────────────────────────────────────────────────
_orig_dense_from_config = keras.layers.Dense.from_config.__func__
@classmethod
def _compat_dense_from_config(cls, config):
    config = {k: v for k, v in config.items() if k != 'quantization_config'}
    return _orig_dense_from_config(cls, config)
keras.layers.Dense.from_config = _compat_dense_from_config

# ── 레이블 맵 ──────────────────────────────────────────────────────────────────
LABEL_MAP = KSL_SENTENCES
idx_to_label = {i: v for i, (k, v) in enumerate(LABEL_MAP.items())}

# ── MediaPipe 초기화 ──────────────────────────────────────────────────────────
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

# FaceLandmarker (face_landmarker.task 필요)
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
        print(f"[!] FaceLandmarker 초기화 실패: {e}. 얼굴 키포인트 없이 진행합니다.")
else:
    print(f"[!] face_landmarker.task 없음 ({_face_model_path}). 얼굴 키포인트 비활성화.")

# ── ASL 파인튜닝 모델 로드 ────────────────────────────────────────────────────
_default_model_path = "models/gloss_models/ksl_asl_finetuned_final.keras"
FINETUNE_MODEL_PATH = os.environ.get("KSL_FINETUNE_MODEL", _default_model_path)

print(f"[*] 모델 로딩 중: {FINETUNE_MODEL_PATH}")
tf.get_logger().setLevel('ERROR')
custom_objects = {
    'CausalDWConv1D': CausalDWConv1D, 'ECA': ECA,
    'LateDropout': LateDropout, 'MultiHeadSelfAttention': MultiHeadSelfAttention,
}
try:
    model = keras.models.load_model(FINETUNE_MODEL_PATH, custom_objects=custom_objects)
    print(f"[*] 모델 로딩 완료. 입력: {model.input_shape}")
except Exception as e:
    print(f"[!] 모델 로딩 실패: {e}")
    try:
        model = keras.models.load_model(FINETUNE_MODEL_PATH, custom_objects=custom_objects, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("[*] 모델 로딩 완료 (compile=False 재시도)")
    except Exception as e2:
        print(f"[!] 최종 모델 로딩 실패: {e2}")
        model = None

# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "asl_finetune",
        "model_loaded": model is not None,
        "face_landmarker": face_landmarker is not None,
        "input_channels": CHANNELS_ASL,
    }


def _init_user(user_id, user_sequences, user_last_label, user_last_emit_time,
               user_frame_count, user_prob_history):
    if user_id not in user_sequences:
        user_sequences[user_id]     = deque(maxlen=SEQ_LEN)
        user_last_label[user_id]    = ""
        user_last_emit_time[user_id]= 0.0
        user_frame_count[user_id]   = 0
        user_prob_history[user_id]  = deque(maxlen=SMOOTHING_WINDOW)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[*] WebSocket 연결됨 (ASL finetune 모드)")

    user_sequences:      dict[str, deque] = {}
    user_last_label:     dict[str, str]   = {}
    user_last_emit_time: dict[str, float] = {}
    user_frame_count:    dict[str, int]   = {}
    user_prob_history:   dict[str, deque] = {}

    try:
        while True:
            message = await websocket.receive()

            # JSON 제어 메시지
            if message.get("text"):
                try:
                    ctrl = json.loads(message["text"])
                    msg_type = ctrl.get("type")
                    if msg_type == "stream_config":
                        uid = ctrl.get("userId")
                        if uid:
                            _init_user(uid, user_sequences, user_last_label,
                                       user_last_emit_time, user_frame_count, user_prob_history)
                    elif msg_type == "stop_stream":
                        uid = ctrl.get("userId")
                        for d in (user_sequences, user_last_label, user_last_emit_time,
                                  user_frame_count, user_prob_history):
                            d.pop(uid, None)
                except Exception as e:
                    print(f"제어 메시지 파싱 오류: {e}")
                continue

            # 바이너리 프레임
            raw = message.get("bytes")
            if not raw or len(raw) < 4:
                continue

            user_id_len = int.from_bytes(raw[:4], 'big')
            if len(raw) < 4 + user_id_len:
                continue
            user_id    = raw[4:4 + user_id_len].decode('utf-8')
            frame_data = raw[4 + user_id_len:]

            if user_id not in user_sequences:
                _init_user(user_id, user_sequences, user_last_label,
                           user_last_emit_time, user_frame_count, user_prob_history)

            # 이미지 디코딩
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # MediaPipe 처리 (pose + hand + face)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_result = pose_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image) if face_landmarker else None

            # 키포인트 추출 (얼굴 포함)
            keypoints = mediapipe_to_openpose_keypoints_with_face(
                pose_result, hand_result, face_result, w, h
            )  # (137, 3)
            user_sequences[user_id].append(keypoints)
            user_frame_count[user_id] += 1

            # 예측
            if (len(user_sequences[user_id]) == SEQ_LEN
                    and model is not None
                    and user_frame_count[user_id] % PREDICTION_STRIDE == 0):
                try:
                    seq_array = np.array(list(user_sequences[user_id]))  # (SEQ_LEN, 137, 3)
                    processed = asl_preprocess_sequence(
                        seq_array, max_len=MAX_LEN, use_asl_norm=True
                    )  # (MAX_LEN, 540)

                    if _debug["enabled"]:
                        logger.info(
                            f"[{user_id}] shape={processed.shape} "
                            f"mean={processed.mean():.3f} std={processed.std():.3f}"
                        )

                    input_batch = np.expand_dims(processed, axis=0)  # (1, MAX_LEN, 540)
                    prediction  = model.predict(input_batch, verbose=0)

                    user_prob_history[user_id].append(prediction[0])
                    smoothed    = np.mean(list(user_prob_history[user_id]), axis=0)
                    confidence  = float(np.max(smoothed))
                    pred_idx    = int(np.argmax(smoothed))

                    if confidence >= THRESHOLD:
                        pred_sign = idx_to_label.get(pred_idx, "알 수 없음")
                        now = time.time()
                        should_emit = (
                            pred_sign != user_last_label[user_id] or
                            (now - user_last_emit_time[user_id]) >= PREDICTION_COOLDOWN_SEC
                        )
                        if should_emit:
                            user_last_label[user_id]     = pred_sign
                            user_last_emit_time[user_id] = now
                            result_text = f"{pred_sign} (정확도: {confidence:.0%})"
                            await websocket.send_json({"userId": user_id, "text": result_text})

                except Exception as e:
                    print(f"[{user_id}] 예측 오류: {e}")
                    await websocket.send_json({"userId": user_id, "text": "예측 중 오류 발생"})

    except WebSocketDisconnect:
        print("[*] WebSocket 연결 종료")
    except Exception as e:
        print(f"[!] 오류: {e}")
