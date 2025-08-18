"""
Video-based Korean Sign Language Recognition using MediaPipe
(Converting to OpenPose format for compatibility with trained model)

This script processes video files for sign language recognition instead of real-time webcam.
Supports various video formats and provides analysis results.

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
import argparse
from datetime import datetime
import csv

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

# 정확도 임계값 설정 (70%)
CONFIDENCE_MIN_THRESHOLD = 0.70


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


def mediapipe_hands_to_openpose_format(mp_hand_landmarks, image_width, image_height):
    """
    Convert MediaPipe hand landmarks directly to OpenPose format
    Both use 21 keypoints, so mapping is straightforward
    """
    hand_keypoints = np.zeros((21, 3))
    
    if mp_hand_landmarks:
        for i, landmark in enumerate(mp_hand_landmarks.landmark):
            # Convert normalized coordinates to pixel coordinates
            hand_keypoints[i] = [
                landmark.x * image_width,
                landmark.y * image_height,
                1.0  # MediaPipe doesn't provide confidence for hands, so we use 1.0
            ]
    
    return hand_keypoints


def mediapipe_to_openpose_keypoints(results, image_width, image_height):
    """
    Convert MediaPipe results to OpenPose format (137 keypoints)
    Upper body focused with enhanced arm tracking
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
    
    # Extract pose landmarks with enhanced arm tracking
    if results.pose_landmarks:
        mp_pose = results.pose_landmarks.landmark
        
        # 상반신 키포인트 추출 (팔 관련 포인트 강조)
        pose[0] = to_pixel_coords(mp_pose[0])     # Nose
        pose[1] = [(to_pixel_coords(mp_pose[11])[0] + to_pixel_coords(mp_pose[12])[0]) / 2,
                   (to_pixel_coords(mp_pose[11])[1] + to_pixel_coords(mp_pose[12])[1]) / 2, 1.0]  # Neck
        
        # 오른팔 관련 키포인트
        pose[2] = to_pixel_coords(mp_pose[12])    # Right shoulder
        pose[3] = to_pixel_coords(mp_pose[14])    # Right elbow
        pose[4] = to_pixel_coords(mp_pose[16])    # Right wrist
        
        # 왼팔 관련 키포인트
        pose[5] = to_pixel_coords(mp_pose[11])    # Left shoulder
        pose[6] = to_pixel_coords(mp_pose[13])    # Left elbow
        pose[7] = to_pixel_coords(mp_pose[15])    # Left wrist
        
        # 추가 상반신 포인트들
        pose[8] = [(to_pixel_coords(mp_pose[23])[0] + to_pixel_coords(mp_pose[24])[0]) / 2,
                   (to_pixel_coords(mp_pose[23])[1] + to_pixel_coords(mp_pose[24])[1]) / 2, 
                   min(mp_pose[23].visibility, mp_pose[24].visibility)]  # Mid hip
        
        # 얼굴 관련 포인트
        pose[15] = to_pixel_coords(mp_pose[2])    # Right eye
        pose[16] = to_pixel_coords(mp_pose[5])    # Left eye
        pose[17] = to_pixel_coords(mp_pose[8])    # Right ear
        pose[18] = to_pixel_coords(mp_pose[7])    # Left ear
        
        # 손가락 끝 포인트
        if hasattr(mp_pose[19], 'visibility'):
            pose[20] = to_pixel_coords(mp_pose[19])
        if hasattr(mp_pose[20], 'visibility'):
            pose[21] = to_pixel_coords(mp_pose[20])
    
    # Extract face landmarks
    if results.face_landmarks:
        mp_face = results.face_landmarks.landmark
        indices = np.linspace(0, 467, 70, dtype=int)
        for i, idx in enumerate(indices):
            face[i] = to_pixel_coords(mp_face[idx])
    
    # Extract hand landmarks
    left_hand = mediapipe_hands_to_openpose_format(
        results.left_hand_landmarks, image_width, image_height
    )
    right_hand = mediapipe_hands_to_openpose_format(
        results.right_hand_landmarks, image_width, image_height
    )
    
    # Concatenate all keypoints
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
    neck_pos = sequence[:, 1:2, :2]
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
    """
    Draw landmarks with MediaPipe styling
    """
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
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


def process_video(video_path, model_path='models/ksl_model_2025_08_17_17-21-36.h5', output_video=None, show_display=True):
    """
    Process a video file for sign language recognition
    
    Args:
        video_path: Path to input video file
        model_path: Path to trained model
        output_video: Path to save output video (optional)
        show_display: Whether to show real-time display
    
    Returns:
        List of recognition results with timestamps
    """
    print(f"\n처리할 비디오: {video_path}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return None
    
    # Load model
    print("모델 로딩 중...")
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    custom_objects = {
        'CausalDWConv1D': CausalDWConv1D,
        'ECA': ECA,
        'LateDropout': LateDropout,
        'MultiHeadSelfAttention': MultiHeadSelfAttention
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        if 'custom_loss' in str(e):
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            print(f"❌ 모델 로딩 실패: {e}")
            return None
    
    print("✓ 모델 로딩 완료")
    
    # Initialize MediaPipe
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        smooth_landmarks=True,
        smooth_segmentation=False
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 오류: 비디오를 열 수 없습니다")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n비디오 정보:")
    print(f"  해상도: {frame_width}x{frame_height}")
    print(f"  FPS: {fps}")
    print(f"  총 프레임: {total_frames}")
    print(f"  재생 시간: {duration:.2f}초")
    
    # Initialize video writer if output is requested
    video_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
        print(f"\n출력 비디오 저장 경로: {output_video}")
    
    # Initialize variables
    sequence_data = deque(maxlen=SEQ_LEN)
    recognition_results = []
    frame_count = 0
    last_prediction_time = 0
    current_sign = ""
    current_confidence = 0.0
    
    # Progress tracking
    last_progress = 0
    
    print("\n비디오 처리 중...")
    print("0%", end='', flush=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        # Update progress
        progress = int((frame_count / total_frames) * 100)
        if progress > last_progress:
            print(f"\r{progress}%", end='', flush=True)
            last_progress = progress
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with MediaPipe
        results = holistic.process(rgb_frame)
        
        # Convert back to BGR
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        draw_styled_landmarks(frame, results)
        
        # Convert keypoints to OpenPose format
        keypoints = mediapipe_to_openpose_keypoints(results, frame_width, frame_height)
        
        # Add keypoints to sequence
        sequence_data.append(keypoints)
        
        # Make prediction when we have enough frames
        if len(sequence_data) == SEQ_LEN and current_time - last_prediction_time > 0.5:
            try:
                # Preprocess and predict
                processed_seq = preprocess_sequence(list(sequence_data))
                input_batch = np.expand_dims(processed_seq, axis=0)
                
                prediction = model.predict(input_batch, verbose=0)
                confidence = np.max(prediction[0])
                
                if confidence >= CONFIDENCE_MIN_THRESHOLD:
                    predicted_idx = np.argmax(prediction[0])
                    predicted_sign = idx_to_label[predicted_idx]
                    
                    if predicted_sign != current_sign:
                        # New sign detected
                        recognition_results.append({
                            'time': current_time,
                            'frame': frame_count,
                            'sign': predicted_sign,
                            'confidence': float(confidence)
                        })
                        current_sign = predicted_sign
                        current_confidence = confidence
                else:
                    current_sign = ""
                    current_confidence = confidence
                
                last_prediction_time = current_time
                
            except Exception as e:
                print(f"\n예측 오류: {e}")
        
        # Add overlay information
        if output_video or show_display:
            display_frame = frame.copy()
            
            # Add semi-transparent overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, 120), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(display_frame, 0.8, overlay, 0.2, 0)
            
            # Add text information
            cv2.putText(display_frame, f"Time: {current_time:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                       (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add current sign if detected
            if current_sign and current_confidence >= CONFIDENCE_MIN_THRESHOLD:
                display_frame = put_korean_text(display_frame, 
                                              f"인식: {current_sign} ({current_confidence:.0%})", 
                                              (10, 60), font_size=35, color=(0, 255, 0))
            
            # Write frame to output video
            if video_writer:
                video_writer.write(display_frame)
            
            # Show display if requested
            if show_display:
                cv2.imshow('Video Sign Language Recognition', display_frame)
                
                # Allow user to control playback
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\n사용자에 의해 중단됨")
                    break
                elif key == ord(' '):  # Space to pause
                    cv2.waitKey(0)  # Wait indefinitely until another key is pressed
    
    print("\r100%")
    print("\n✓ 비디오 처리 완료")
    
    # Cleanup
    holistic.close()
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Print results summary
    print(f"\n인식 결과 요약:")
    print(f"총 {len(recognition_results)}개의 수어가 인식되었습니다.")
    
    if recognition_results:
        print("\n시간별 인식 결과:")
        for result in recognition_results:
            print(f"  {result['time']:6.2f}초 - {result['sign']:15s} (신뢰도: {result['confidence']:.0%})")
        
        # Count occurrences
        sign_counts = {}
        for result in recognition_results:
            sign = result['sign']
            sign_counts[sign] = sign_counts.get(sign, 0) + 1
        
        print("\n수어별 인식 횟수:")
        for sign, count in sorted(sign_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sign}: {count}회")
    
    return recognition_results


def save_results_to_csv(results, output_path):
    """Save recognition results to CSV file"""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['time', 'frame', 'sign', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n결과가 CSV 파일로 저장되었습니다: {output_path}")


def save_results_to_json(results, output_path):
    """Save recognition results to JSON file"""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_recognitions': len(results),
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 JSON 파일로 저장되었습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Korean Sign Language Recognition from Video')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--model', type=str, default='models/ksl_model_2025_08_17_17-21-36.h5',
                       help='Path to trained model (default: models/ksl_model_2025_08_17_17-21-36.h5)')
    parser.add_argument('--output-video', type=str, help='Path to save output video with annotations')
    parser.add_argument('--output-csv', type=str, help='Path to save results as CSV')
    parser.add_argument('--output-json', type=str, help='Path to save results as JSON')
    parser.add_argument('--no-display', action='store_true', help='Do not show video display')
    
    args = parser.parse_args()
    
    # Process video
    results = process_video(
        args.video_path,
        model_path=args.model,
        output_video=args.output_video,
        show_display=not args.no_display
    )
    
    if results:
        # Save results if requested
        if args.output_csv:
            save_results_to_csv(results, args.output_csv)
        
        if args.output_json:
            save_results_to_json(results, args.output_json)
        
        # Generate default output names if not specified
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not args.output_csv and not args.output_json:
            # Save at least one output format by default
            default_csv = f"{base_name}_results_{timestamp}.csv"
            save_results_to_csv(results, default_csv)


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
    
    main()