"""
서빙 전용 전처리 함수 모음.
학습 의존성(boto3, tqdm, gcs 등) 없이 동작합니다.
"""
import numpy as np
import keras
from src.config import POINT_LANDMARKS, UMAP_LOAD_PATH


def mediapipe_hands_to_openpose_format(mp_hand_landmarks, image_width, image_height):
    hand_keypoints = np.zeros((21, 3))
    if mp_hand_landmarks:
        for i, landmark in enumerate(mp_hand_landmarks.landmark):
            hand_keypoints[i] = [landmark.x * image_width, landmark.y * image_height, 1.0]
    return hand_keypoints


def mediapipe_to_openpose_keypoints(results, image_width, image_height):
    pose = np.zeros((25, 3)); face = np.zeros((70, 3))
    left_hand = np.zeros((21, 3)); right_hand = np.zeros((21, 3))
    def to_pixel_coords(landmark):
        return [landmark.x * image_width, landmark.y * image_height, landmark.visibility if hasattr(landmark, 'visibility') else 1.0]
    if results.pose_landmarks:
        mp_pose = results.pose_landmarks.landmark
        pose[0] = to_pixel_coords(mp_pose[0])
        pose[1] = [(to_pixel_coords(mp_pose[11])[0] + to_pixel_coords(mp_pose[12])[0]) / 2, (to_pixel_coords(mp_pose[11])[1] + to_pixel_coords(mp_pose[12])[1]) / 2, 1.0]
        pose[2] = to_pixel_coords(mp_pose[12]); pose[3] = to_pixel_coords(mp_pose[14]); pose[4] = to_pixel_coords(mp_pose[16])
        pose[5] = to_pixel_coords(mp_pose[11]); pose[6] = to_pixel_coords(mp_pose[13]); pose[7] = to_pixel_coords(mp_pose[15])
    left_hand = mediapipe_hands_to_openpose_format(results.left_hand_landmarks, image_width, image_height)
    right_hand = mediapipe_hands_to_openpose_format(results.right_hand_landmarks, image_width, image_height)
    return np.concatenate([pose, face, left_hand, right_hand], axis=0)


def main_preprocess_sequence(sequence: np.ndarray, max_len: int) -> np.ndarray:
    sequence = np.array(sequence)
    original_len = len(sequence)

    if original_len > max_len:
        sequence = sequence[:max_len]
    else:
        padding = np.zeros((max_len - original_len, sequence.shape[1], sequence.shape[2]))
        sequence = np.concatenate([sequence, padding], axis=0)

    valid_frames = sequence[:original_len]
    all_points = valid_frames.reshape(-1, 2)
    all_x = all_points[:, 0]
    all_y = all_points[:, 1]

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    scale = max(x_max - x_min, y_max - y_min) / 2

    if scale < 1e-6:
        scale = 1.0

    normalized_sequence = (sequence - [center_x, center_y]) / scale
    normalized_sequence = np.clip(normalized_sequence, -2, 2)
    selected_seq = normalized_sequence[:, POINT_LANDMARKS, :]
    selected_seq = selected_seq.reshape(max_len, -1)
    selected_seq = np.nan_to_num(selected_seq, 0)

    umap_encoder = keras.models.load_model(UMAP_LOAD_PATH)
    embedding = umap_encoder.predict(selected_seq)

    return embedding
