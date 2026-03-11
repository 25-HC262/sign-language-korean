import json
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import keras
from load_data.inference import mediapipe_to_openpose_keypoints, \
    main_preprocess_sequence
from src.backbone import CausalDWConv1D, ECA, LateDropout, \
    MultiHeadSelfAttention
from src.config import SEQ_LEN, THRESHOLD, KSL_SENTENCES, GM_LOAD_PATH, CROP_LEN

# 수어 레이블 정의
LABEL_MAP = KSL_SENTENCES
idx_to_label = {i: v for i, (k, v) in enumerate(LABEL_MAP.items())}

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic

print("모델 로딩 중...")
tf.get_logger().setLevel('ERROR')
custom_objects = {
    'CausalDWConv1D': CausalDWConv1D, 'ECA': ECA,
    'LateDropout': LateDropout, 'MultiHeadSelfAttention': MultiHeadSelfAttention
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

# -- FastAPI 앱 및 WebSocket 엔드포인트 --
app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


def _make_holistic():
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("스트리밍 서버가 연결되었습니다.")

    # userId별 시퀀스 deque와 MediaPipe 인스턴스를 독립적으로 유지
    user_sequences: dict[str, deque] = {}
    user_holistics: dict[str, any] = {}

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
                            user_holistics[user_id] = _make_holistic()
                            print(f"[{user_id}] 시퀀스 및 MediaPipe 초기화")

                    elif msg_type == "stop_stream":
                        user_id = ctrl.get("userId")
                        if user_id in user_holistics:
                            user_holistics[user_id].close()
                            del user_holistics[user_id]
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
                user_holistics[user_id] = _make_holistic()
                print(f"[{user_id}] lazy 초기화")

            # 이미지 디코딩
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # MediaPipe 처리
            image_height, image_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = user_holistics[user_id].process(rgb_frame)

            # 키포인트 추출 및 시퀀스 추가
            keypoints = mediapipe_to_openpose_keypoints(results, image_width, image_height)
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
    finally:
        for uid, holistic in user_holistics.items():
            holistic.close()
            print(f"[{uid}] MediaPipe 리소스 정리 완료")