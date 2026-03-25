import json
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import keras
from load_data.inference import mediapipe_to_openpose_keypoints, \
    main_preprocess_sequence
from src.backbone import CausalDWConv1D, ECA, LateDropout, \
    MultiHeadSelfAttention
from src.config import SEQ_LEN, THRESHOLD, KSL_SENTENCES, GM_LOAD_PATH, CROP_LEN, get_config_args, PathConfig

logger = logging.getLogger("ksl")
logging.basicConfig(level=logging.INFO)

_debug = {"enabled": os.environ.get("KSL_DEBUG", "0") == "1"}

# 중복 출력 방지 설정
PREDICTION_COOLDOWN_SEC = 1.5   # 같은 단어 재출력까지 최소 대기 시간(초)
PREDICTION_STRIDE = 15          # N 프레임마다 1회 예측
SMOOTHING_WINDOW = 3            # 최근 N개 예측 확률 평균 후 argmax

# 이전 버전 Keras 호환: quantization_config이 포함된 모델 로딩 지원
# custom_objects는 Keras 내장 레이어에 적용되지 않아 from_config를 직접 패치
_orig_dense_from_config = keras.layers.Dense.from_config.__func__

@classmethod
def _compat_dense_from_config(cls, config):
    config = {k: v for k, v in config.items() if k != 'quantization_config'}
    return _orig_dense_from_config(cls, config)

keras.layers.Dense.from_config = _compat_dense_from_config

# 수어 레이블 정의
LABEL_MAP = KSL_SENTENCES
idx_to_label = {i: v for i, (k, v) in enumerate(LABEL_MAP.items())}

# MediaPipe 초기화 (0.10.x Tasks API)
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

print("모델 로딩 중...")
tf.get_logger().setLevel('ERROR')
custom_objects = {
    'CausalDWConv1D': CausalDWConv1D, 'ECA': ECA,
    'LateDropout': LateDropout, 'MultiHeadSelfAttention': MultiHeadSelfAttention,
}
try:
    model = keras.models.load_model(GM_LOAD_PATH, custom_objects=custom_objects)
    print("커스텀 모델 로딩 완료")
except Exception as e:
    print(f"모델 로딩 실패. 컴파일 없이 다시 시도합니다. 오류: {e}")
    try:
        model = keras.models.load_model(GM_LOAD_PATH, custom_objects=custom_objects, compile=False)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("커스텀 모델 (비컴파일) 로딩 완료")
    except Exception as e2:
        print(f"최종 모델 로딩 실패: {e2}")
        model = None

print("UMAP 인코더 로딩 중...")
try:
    _umap_path = PathConfig(get_config_args()).UMAP_LOAD_PATH
    umap_encoder = keras.models.load_model(_umap_path)
    print(f"UMAP 인코더 로딩 완료: {_umap_path}")
except Exception as e:
    print(f"UMAP 인코더 로딩 실패: {e}")
    umap_encoder = None

# -- FastAPI 앱 및 WebSocket 엔드포인트 --
app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("스트리밍 서버가 연결되었습니다.")

    # userId별 시퀀스 deque (MediaPipe는 전역 인스턴스 공유)
    user_sequences: dict[str, deque] = {}

    try:
        while True:
            message = await websocket.receive()

            # JSON 제어 메시지 처리
            if message.get("text"):
                try:
                    ctrl = json.loads(message["text"])
                    msg_type = ctrl.get("type")

                    if msg_type == "stream_config":
                        user_id = ctrl.get("userId")
                        if user_id and user_id not in user_sequences:
                            user_sequences[user_id] = deque(maxlen=SEQ_LEN)
                            print(f"[{user_id}] 시퀀스 초기화")

                    elif msg_type == "stop_stream":
                        user_id = ctrl.get("userId")
                        user_sequences.pop(user_id, None)
                        print(f"[{user_id}] 리소스 정리 완료")

                except Exception as e:
                    print(f"제어 메시지 파싱 오류: {e}")
                continue

            # 바이너리 프레임 처리
            raw = message.get("bytes")
            if not raw or len(raw) < 4:
                continue

            # 4바이트 userId 헤더 파싱
            user_id_len = int.from_bytes(raw[:4], 'big')
            if len(raw) < 4 + user_id_len:
                continue
            user_id = raw[4:4 + user_id_len].decode('utf-8')
            frame_data = raw[4 + user_id_len:]

            # stream_config 없이 프레임이 먼저 도착한 경우 lazy 초기화
            if user_id not in user_sequences:
                user_sequences[user_id] = deque(maxlen=SEQ_LEN)
                print(f"[{user_id}] lazy 초기화")

            # 이미지 디코딩
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # MediaPipe 처리
            image_height, image_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_result = pose_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)

            # 키포인트 추출 및 시퀀스 추가
            keypoints = mediapipe_to_openpose_keypoints(pose_result, hand_result, image_width, image_height)
            user_sequences[user_id].append(keypoints)

            # 60프레임 시퀀스가 쌓이면 예측
            if len(user_sequences[user_id]) == SEQ_LEN and model:
                try:
                    processed_seq = main_preprocess_sequence(
                        np.array(list(user_sequences[user_id])), max_len=CROP_LEN
                    )
                    input_batch = np.expand_dims(processed_seq, axis=0)

                    prediction = model.predict(input_batch, verbose=0)
                    confidence = np.max(prediction[0])

                    if confidence >= THRESHOLD:
                        predicted_idx = np.argmax(prediction[0])
                        predicted_sign = idx_to_label.get(predicted_idx, "알 수 없음")
                        result_text = f"{predicted_sign} (정확도: {confidence:.0%})"
                    else:
                        result_text = "인식 결과 없음"

                    # userId 포함하여 응답 → 스트리밍 서버가 해당 사용자에게만 전달
                    await websocket.send_json({"userId": user_id, "text": result_text})

                except Exception as e:
                    print(f"[{user_id}] 예측 오류: {e}")
                    await websocket.send_json({"userId": user_id, "text": "예측 중 오류 발생"})

    except WebSocketDisconnect:
        print("스트리밍 서버 연결이 끊겼습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")