import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from collections import deque
import mediapipe as mp

# -- 1. Custom Keras Layers (main_video.py에서 가져옴) --
from tensorflow.keras.layers import Layer, Conv1D, Dense, Dropout, Add, Input, GlobalAveragePooling1D, Activation, BatchNormalization, Multiply, Reshape, Lambda
from tensorflow.keras import backend as K
from src.backbone import get_model, CausalDWConv1D, ECA, LateDropout, MultiHeadSelfAttention
from src.config import SEQ_LEN, THRESHOLD, MAX_LEN, POINT_LANDMARKS, KSL_SENTENCES, GLOSS_TRANSFORMER_MODEL_PATH
from load_data.create_dataset import mediapipe_to_openpose_keypoints, preprocess_sequence

# 수어 레이블 정의
LABEL_MAP = KSL_SENTENCES
idx_to_label = {i: v for i, (k, v) in enumerate(LABEL_MAP.items())}

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic

# -- 4. 모델 로딩 --
print("모델 로딩 중...")
tf.get_logger().setLevel('ERROR')
custom_objects = {
    'CausalDWConv1D': CausalDWConv1D, 'ECA': ECA,
    'LateDropout': LateDropout, 'MultiHeadSelfAttention': MultiHeadSelfAttention
}
try:
    model = tf.keras.models.load_model(GLOSS_TRANSFORMER_MODEL_PATH, custom_objects=custom_objects)
    print("커스텀 모델 로딩 완료")
except Exception as e:
    print(f"모델 로딩 실패. 컴파일 없이 다시 시도합니다. 오류: {e}")
    try:
        model = tf.keras.models.load_model(GLOSS_TRANSFORMER_MODEL_PATH, custom_objects=custom_objects, compile=False)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("커스텀 모델 (비컴파일) 로딩 완료")
    except Exception as e2:
        print(f"최종 모델 로딩 실패: {e2}")
        model = None

# -- 5. FastAPI 앱 및 WebSocket 엔드포인트 --
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("클라이언트가 연결되었습니다.")

    # WebSocket 연결마다 고유한 시퀀스 데이터와 MediaPipe 인스턴스를 가집니다.
    sequence_data = deque(maxlen=SEQ_LEN)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    try:
        while True:
            # 클라이언트로부터 영상 프레임 데이터 수신
            data = await websocket.receive_bytes()
            
            # 받은 바이트 데이터를 이미지로 변환
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # MediaPipe 처리
                image_height, image_width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = holistic.process(rgb_frame)
                
                # 키포인트 추출 및 시퀀스에 추가
                keypoints = mediapipe_to_openpose_keypoints(results, image_width, image_height)
                sequence_data.append(keypoints)
                
                # 시퀀스가 충분히 쌓이면 예측 수행
                if len(sequence_data) == SEQ_LEN and model:
                    try:
                        processed_seq = preprocess_sequence(list(sequence_data))
                        input_batch = np.expand_dims(processed_seq, axis=0)
                        
                        prediction = model.predict(input_batch, verbose=0)
                        confidence = np.max(prediction[0])
                        
                        if confidence >= THRESHOLD:
                            predicted_idx = np.argmax(prediction[0])
                            predicted_sign = idx_to_label.get(predicted_idx, "알 수 없음")
                            result_text = f"{predicted_sign} (정확도: {confidence:.0%})"
                        else:
                            result_text = "인식 결과 없음"
                        
                        # 결과를 클라이언트로 전송
                        await websocket.send_json({"text": result_text})
                    
                    except Exception as e:
                        print(f"예측 오류 발생: {e}")
                        await websocket.send_json({"text": "예측 중 오류 발생"})
            
    except WebSocketDisconnect:
        print("클라이언트 연결이 끊겼습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 연결 종료 시 리소스 정리
        holistic.close()
        print("MediaPipe 리소스를 정리했습니다.")