"""
Real-time Korean Sign Language Recognition using MediaPipe
(Converting to OpenPose format for compatibility with trained model)
Upper body focused version for video conferencing

This script uses MediaPipe for keypoint extraction but converts the output
to OpenPose format to be compatible with the model trained on OpenPose data.

Date: June 2025
"""

import os
import json
import numpy as np
import cv2
import time
from collections import deque
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import platform

# Import custom modules
from src.backbone import get_model, CausalDWConv1D, ECA, LateDropout, MultiHeadSelfAttention
from src.config import SEQ_LEN, THRESH_HOLD, MAX_LEN, POINT_LANDMARKS

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Label mapping
LABEL_MAP = {
    'NIA_SL_SEN0181': '도와주세요',
    'NIA_SL_SEN0354': '안녕하세요',
    'NIA_SL_SEN0355': '감사합니다',
    'NIA_SL_SEN0356': '죄송합니다',
    'NIA_SL_SEN2000': '수고하셨습니다'
}

# Create reverse mapping for decoding
idx_to_label = {i: v for i, (k, v) in enumerate(LABEL_MAP.items())}
label_to_idx = {v: i for i, v in idx_to_label.items()}


def put_korean_text(image, text, position, font_size=30, color=(255, 255, 255)):
    """
    OpenCV 이미지에 한글 텍스트를 추가하는 함수
    """
    # Convert to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 폰트 설정 (시스템에 따라 다른 폰트 경로 사용)
    try:
        if platform.system() == 'Darwin':  # macOS
            font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        elif platform.system() == 'Windows':
            font_path = 'C:/Windows/Fonts/malgun.ttf'
        else:  # Linux
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        
        font = ImageFont.truetype(font_path, font_size)
    except:
        # 폰트를 찾을 수 없는 경우 기본 폰트 사용
        font = ImageFont.load_default()
    
    # Draw text
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def mediapipe_to_openpose_keypoints(results, image_width, image_height):
    """
    Convert MediaPipe results to OpenPose format (137 keypoints)
    MediaPipe와 OpenPose의 키포인트를 매칭하여 변환
    Upper body focused - 하체 키포인트는 0으로 설정
    """
    # Initialize arrays with zeros
    pose = np.zeros((25, 3))
    face = np.zeros((70, 3))
    left_hand = np.zeros((21, 3))
    right_hand = np.zeros((21, 3))
    
    # Convert normalized coordinates to pixel coordinates
    def to_pixel_coords(landmark):
        return [
            landmark.x * image_width,
            landmark.y * image_height,
            landmark.visibility if hasattr(landmark, 'visibility') else 1.0
        ]
    
    # Extract pose landmarks (33 MediaPipe → 25 OpenPose mapping)
    # 상반신만 추출 (하반신 관련 포인트는 0으로 유지)
    if results.pose_landmarks:
        mp_pose = results.pose_landmarks.landmark
        
        # 상반신 키포인트만 추출
        pose[0] = to_pixel_coords(mp_pose[0])     # Nose
        pose[1] = [(to_pixel_coords(mp_pose[11])[0] + to_pixel_coords(mp_pose[12])[0]) / 2,
                   (to_pixel_coords(mp_pose[11])[1] + to_pixel_coords(mp_pose[12])[1]) / 2, 1.0]  # Neck
        pose[2] = to_pixel_coords(mp_pose[12])    # Right shoulder
        pose[3] = to_pixel_coords(mp_pose[14])    # Right elbow
        pose[4] = to_pixel_coords(mp_pose[16])    # Right wrist
        pose[5] = to_pixel_coords(mp_pose[11])    # Left shoulder
        pose[6] = to_pixel_coords(mp_pose[13])    # Left elbow
        pose[7] = to_pixel_coords(mp_pose[15])    # Left wrist
        
        # 중요한 상반신 포인트들
        pose[15] = to_pixel_coords(mp_pose[2])    # Right eye
        pose[16] = to_pixel_coords(mp_pose[5])    # Left eye
        pose[17] = to_pixel_coords(mp_pose[8])    # Right ear
        pose[18] = to_pixel_coords(mp_pose[7])    # Left ear
        
        # 하반신 포인트들(8-14, 19-24)은 0으로 유지
        # 이는 모델이 OpenPose 형식을 기대하기 때문에 필요
    
    # Extract face landmarks (468 MediaPipe → 70 OpenPose)
    if results.face_landmarks:
        mp_face = results.face_landmarks.landmark
        # Sample 70 evenly distributed points from 468 MediaPipe face landmarks
        indices = np.linspace(0, 467, 70, dtype=int)
        for i, idx in enumerate(indices):
            face[i] = to_pixel_coords(mp_face[idx])
    
    # Extract hand landmarks (21 points each, same as OpenPose)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            left_hand[i] = to_pixel_coords(lm)
    
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            right_hand[i] = to_pixel_coords(lm)
    
    # Concatenate all keypoints in OpenPose order
    all_keypoints = np.concatenate([pose, face, left_hand, right_hand], axis=0)
    
    return all_keypoints


def preprocess_sequence(sequence):
    """Preprocess a sequence of keypoints for the model"""
    sequence = np.array(sequence)
    
    # Pad or truncate to MAX_LEN
    if len(sequence) > MAX_LEN:
        sequence = sequence[:MAX_LEN]
    else:
        pad_length = MAX_LEN - len(sequence)
        padding = np.zeros((pad_length, sequence.shape[1], sequence.shape[2]))
        sequence = np.concatenate([sequence, padding], axis=0)
    
    # Extract selected landmarks
    selected_seq = sequence[:, POINT_LANDMARKS, :]
    
    # Normalize using neck as reference
    neck_pos = sequence[:, 1:2, :2]  # Neck is at index 1
    neck_mean = np.nanmean(neck_pos, axis=(0, 1), keepdims=True)
    if np.isnan(neck_mean).any():
        neck_mean = np.array([[[0.5, 0.5]]])
    
    selected_xy = selected_seq[:, :, :2]
    std = np.nanstd(selected_xy)
    if std == 0:
        std = 1.0
    selected_xy = (selected_xy - neck_mean) / std
    
    # Calculate derivatives
    dx = np.zeros_like(selected_xy)
    dx[1:] = selected_xy[1:] - selected_xy[:-1]
    
    dx2 = np.zeros_like(selected_xy)
    dx2[2:] = selected_xy[2:] - selected_xy[:-2]
    
    # Flatten and concatenate
    x_flat = selected_xy.reshape(MAX_LEN, -1)
    dx_flat = dx.reshape(MAX_LEN, -1)
    dx2_flat = dx2.reshape(MAX_LEN, -1)
    
    processed = np.concatenate([x_flat, dx_flat, dx2_flat], axis=-1)
    processed = np.nan_to_num(processed, 0)
    
    return processed


def draw_styled_landmarks(image, results):
    """Draw landmarks with MediaPipe styling - 상반신 중심"""
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # Draw pose landmarks (상반신만)
    if results.pose_landmarks:
        # 상반신 연결선만 그리기 위한 커스텀 연결
        upper_body_connections = [
            (0, 1),   # nose to neck
            (1, 2),   # neck to right_shoulder
            (2, 3),   # right_shoulder to right_elbow
            (3, 4),   # right_elbow to right_wrist
            (1, 5),   # neck to left_shoulder
            (5, 6),   # left_shoulder to left_elbow
            (6, 7),   # left_elbow to left_wrist
            (0, 15),  # nose to right_eye
            (0, 16),  # nose to left_eye
            (15, 17), # right_eye to right_ear
            (16, 18), # left_eye to left_ear
        ]
        
        # 상반신 포인트만 그리기
        for connection in upper_body_connections:
            start_idx, end_idx = connection
            if start_idx < len(results.pose_landmarks.landmark) and end_idx < len(results.pose_landmarks.landmark):
                start = results.pose_landmarks.landmark[start_idx]
                end = results.pose_landmarks.landmark[end_idx]
                
                # 두 점이 모두 visible한 경우에만 연결선 그리기
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                    end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # 상반신 랜드마크 점 그리기
        upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
        for idx in upper_body_indices:
            if idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    cx = int(landmark.x * image.shape[1])
                    cy = int(landmark.y * image.shape[0])
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    
    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


def real_time_ksl():
    """
    Perform real-time Korean Sign Language recognition using webcam feed with MediaPipe.
    Upper body focused version for video conferencing.
    """
    print("Starting Korean Sign Language Recognition...")
    print("Using MediaPipe for keypoint extraction (OpenPose format conversion)")
    print("Upper body focused mode for video conferencing")
    
    print("\nLoading trained model...")
    
    # TensorFlow import
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    # Custom objects for model loading
    custom_objects = {
        'CausalDWConv1D': CausalDWConv1D,
        'ECA': ECA,
        'LateDropout': LateDropout,
        'MultiHeadSelfAttention': MultiHeadSelfAttention
    }
    
    # Load the trained model
    model_path = 'checkpoints/best_model.h5'
    if not os.path.exists(model_path):
        model_path = 'models/ksl_model_final.h5'
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize MediaPipe with upper body focus
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,  # 0, 1, or 2. Higher = more accurate but slower
        enable_segmentation=False,  # 배경 분할 비활성화로 성능 향상
        refine_face_landmarks=True  # 얼굴 랜드마크 정밀도 향상
    )
    
    # Initialize variables
    sequence_data = deque(maxlen=SEQ_LEN)
    recognized_signs = []
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Set camera properties for better upper body capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get actual camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nCamera resolution: {frame_width}x{frame_height}")
    print("\nStarting real-time recognition...")
    print("Controls:")
    print("  'q' - 종료")
    print("  'c' - 기록 지우기")
    print("  's' - 현재 화면 저장")
    print("  'p' - 일시정지/재개")
    print("  'r' - 카메라 위치 가이드 재설정")
    
    frame_count = 0
    last_prediction_time = time.time()
    current_sign = ""
    confidence_threshold = THRESH_HOLD
    paused = False
    show_guide = True
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
        
        frame_count += 1
        fps_frame_count += 1
        
        # Calculate FPS
        if time.time() - fps_start_time > 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        if not paused:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process with MediaPipe
            results = holistic.process(rgb_frame)
            
            # Convert back to BGR for OpenCV
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            draw_styled_landmarks(frame, results)
            
            # Convert MediaPipe keypoints to OpenPose format
            keypoints = mediapipe_to_openpose_keypoints(results, frame_width, frame_height)
            
            # Add keypoints to sequence
            sequence_data.append(keypoints)
            
            # Make prediction when we have enough frames
            if len(sequence_data) == SEQ_LEN and time.time() - last_prediction_time > 0.5:
                try:
                    # Preprocess sequence
                    processed_seq = preprocess_sequence(list(sequence_data))
                    input_batch = np.expand_dims(processed_seq, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(input_batch, verbose=0)
                    confidence = np.max(prediction[0])
                    
                    if confidence > confidence_threshold:
                        predicted_idx = np.argmax(prediction[0])
                        predicted_sign = idx_to_label[predicted_idx]
                        
                        if predicted_sign != current_sign:
                            current_sign = predicted_sign
                            recognized_signs.append(current_sign)
                            # Keep only last 5 signs
                            if len(recognized_signs) > 5:
                                recognized_signs.pop(0)
                            
                            print(f"✓ 인식됨: {current_sign} (정확도: {confidence:.2%})")
                    else:
                        current_sign = ""
                    
                    last_prediction_time = time.time()
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
        
        # Create display
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # Add position guide for upper body (선택적)
        if show_guide and frame_count < 150:  # 처음 5초 동안만 표시
            # 상반신 위치 가이드 그리기
            guide_color = (200, 200, 200)
            cv2.rectangle(display_frame, 
                         (width//4, height//8), 
                         (3*width//4, 3*height//4), 
                         guide_color, 2)
            display_frame = put_korean_text(display_frame, "상반신을 프레임 안에 위치시켜주세요", 
                                          (width//4, height//8 - 20), 
                                          font_size=25, color=guide_color)
        
        # Add semi-transparent overlay for text background
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.8, overlay, 0.2, 0)
        
        # Add status information
        status_color = (0, 255, 0) if not paused else (0, 165, 255)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.putText(display_frame, f"Frames: {len(sequence_data)}/{SEQ_LEN}", 
                   (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Upper Body Mode", 
                   (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if paused:
            cv2.putText(display_frame, "PAUSED", 
                       (width - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # 한글 텍스트 표시
        if current_sign:
            display_frame = put_korean_text(display_frame, f"현재: {current_sign}", 
                                          (10, 60), font_size=35, color=(0, 255, 0))
        
        if recognized_signs:
            history_text = "기록: " + " > ".join(recognized_signs[-3:])
            display_frame = put_korean_text(display_frame, history_text, 
                                          (10, 100), font_size=25, color=(255, 255, 0))
        
        # Add confidence bar
        if current_sign and 'confidence' in locals():
            bar_width = int(200 * confidence)
            cv2.rectangle(display_frame, (10, 130), (210, 140), (100, 100, 100), -1)
            cv2.rectangle(display_frame, (10, 130), (10 + bar_width, 140), (0, 255, 0), -1)
            cv2.putText(display_frame, f"{confidence:.1%}", 
                       (220, 139), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Korean Sign Language Recognition - Video Conference Mode', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            recognized_signs.clear()
            current_sign = ""
            print("기록이 지워졌습니다")
        elif key == ord('s'):
            filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"화면이 {filename}으로 저장되었습니다")
        elif key == ord('p'):
            paused = not paused
            print("일시정지" if paused else "재개")
        elif key == ord('r'):
            show_guide = True
            frame_count = 0
            print("위치 가이드를 재설정했습니다")
    
    # Cleanup
    holistic.close()
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n\n인식이 중지되었습니다.")
    if recognized_signs:
        print(f"최종 인식된 수어: {', '.join(recognized_signs)}")


if __name__ == "__main__":
    # Check if MediaPipe is installed
    try:
        import mediapipe
        print("MediaPipe version:", mediapipe.__version__)
    except ImportError:
        print("MediaPipe not installed. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
        print("MediaPipe installed. Please run the script again.")
        exit()
    
    real_time_ksl()