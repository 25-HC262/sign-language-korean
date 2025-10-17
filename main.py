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
from src.config import SEQ_LEN, THRESH_HOLD, MAX_LEN, POINT_LANDMARKS, KSL_SENTENCES

# 수어 레이블 정의
LABEL_MAP = KSL_SENTENCES
idx_to_label = {i: v for i, (k, v) in enumerate(LABEL_MAP.items())}

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic

# -- 3. 전처리 및 키포인트 변환 함수 --
def mediapipe_hands_to_openpose_format(mp_hand_landmarks, image_width, image_height):
    hand_keypoints = np.zeros((21, 3))
    if mp_hand_landmarks:
        for i, landmark in enumerate(mp_hand_landmarks.landmark):
            hand_keypoints[i] = [landmark.x * image_width, landmark.y * image_height, 1.0]
    return hand_keypoints

def mediapipe_to_openpose_keypoints(results, image_width, image_height):
    pose = np.zeros((25, 3)); face = np.zeros((70, 3)); 
    left_hand = np.zeros((21, 3)); right_hand = np.zeros((21, 3))
    def to_pixel_coords(landmark):
        return [landmark.x * image_width, landmark.y * image_height, landmark.visibility if hasattr(landmark, 'visibility') else 1.0]
    if results.pose_landmarks:
        mp_pose = results.pose_landmarks.landmark
        pose[0] = to_pixel_coords(mp_pose[0]) # Nose
        pose[1] = [(to_pixel_coords(mp_pose[11])[0] + to_pixel_coords(mp_pose[12])[0]) / 2, (to_pixel_coords(mp_pose[11])[1] + to_pixel_coords(mp_pose[12])[1]) / 2, 1.0] # Neck
        pose[2] = to_pixel_coords(mp_pose[12]); pose[3] = to_pixel_coords(mp_pose[14]); pose[4] = to_pixel_coords(mp_pose[16]) # Right Arm
        pose[5] = to_pixel_coords(mp_pose[11]); pose[6] = to_pixel_coords(mp_pose[13]); pose[7] = to_pixel_coords(mp_pose[15]) # Left Arm
    left_hand = mediapipe_hands_to_openpose_format(results.left_hand_landmarks, image_width, image_height)
    right_hand = mediapipe_hands_to_openpose_format(results.right_hand_landmarks, image_width, image_height)
    return np.concatenate([pose, face, left_hand, right_hand], axis=0)

def preprocess_sequence(sequence):
    sequence = np.array(sequence)
    if len(sequence) > MAX_LEN: sequence = sequence[:MAX_LEN]
    else: sequence = np.concatenate([sequence, np.zeros((MAX_LEN - len(sequence), sequence.shape[1], sequence.shape[2]))], axis=0)
    selected_seq = sequence[:, POINT_LANDMARKS, :]
    neck_pos = sequence[:, 1:2, :2]
    neck_mean = np.nanmean(neck_pos, axis=(0, 1), keepdims=True)
    if np.isnan(neck_mean).any(): neck_mean = np.array([[[0.5, 0.5]]])
    selected_xy = selected_seq[:, :, :2]
    std = np.nanstd(selected_xy)
    if std == 0: std = 1.0
    selected_xy = (selected_xy - neck_mean) / std
    x_flat = selected_xy.reshape(MAX_LEN, -1)
    processed = np.nan_to_num(x_flat, 0)
    return processed

# -- 4. 모델 로딩 --
MODEL_PATH = 'models/gloss_transformer_models/sign_language_v1.h5'
print("모델 로딩 중...")
tf.get_logger().setLevel('ERROR')
custom_objects = {
    'CausalDWConv1D': CausalDWConv1D, 'ECA': ECA,
    'LateDropout': LateDropout, 'MultiHeadSelfAttention': MultiHeadSelfAttention
}
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("커스텀 모델 로딩 완료")
except Exception as e:
    print(f"모델 로딩 실패. 컴파일 없이 다시 시도합니다. 오류: {e}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
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
                        
                        if confidence >= THRESH_HOLD:
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