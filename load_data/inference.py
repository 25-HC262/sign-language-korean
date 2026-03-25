"""
서빙 전용 전처리 함수 모음.
학습 의존성(boto3, tqdm, gcs 등) 없이 동작합니다.
"""
import keras
import numpy as np

from src.config import POINT_LANDMARKS, get_config_args, PathConfig


def mediapipe_to_openpose_keypoints(pose_result, hand_result, image_width, image_height):
    pose = np.zeros((25, 3)); face = np.zeros((70, 3))
    left_hand = np.zeros((21, 3)); right_hand = np.zeros((21, 3))

    def to_pixel_coords(landmark):
        return [landmark.x * image_width, landmark.y * image_height, landmark.visibility if hasattr(landmark, 'visibility') else 1.0]

    # Pose (PoseLandmarker 결과: pose_landmarks[0] = 첫 번째 사람의 랜드마크 리스트)
    if pose_result.pose_landmarks:
        mp_pose = pose_result.pose_landmarks[0]
        pose[0] = to_pixel_coords(mp_pose[0])
        pose[1] = [(to_pixel_coords(mp_pose[11])[0] + to_pixel_coords(mp_pose[12])[0]) / 2, (to_pixel_coords(mp_pose[11])[1] + to_pixel_coords(mp_pose[12])[1]) / 2, 1.0]
        pose[2] = to_pixel_coords(mp_pose[12]); pose[3] = to_pixel_coords(mp_pose[14]); pose[4] = to_pixel_coords(mp_pose[16])
        pose[5] = to_pixel_coords(mp_pose[11]); pose[6] = to_pixel_coords(mp_pose[13]); pose[7] = to_pixel_coords(mp_pose[15])

    # Hands (HandLandmarker 결과: hand_landmarks[i] = i번째 손, handedness[i][0].category_name = "Left"/"Right")
    if hand_result.hand_landmarks:
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            hand_keypoints = np.array([[lm.x * image_width, lm.y * image_height, 1.0] for lm in landmarks])
            if handedness[0].category_name == "Left":
                left_hand = hand_keypoints
            else:
                right_hand = hand_keypoints

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

    args = get_config_args()
    pc = PathConfig(args)

    umap_encoder = keras.models.load_model(pc.UMAP_LOAD_PATH)
    embedding = umap_encoder.predict(selected_seq)

    return embedding
