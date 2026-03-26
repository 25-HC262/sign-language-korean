"""
ASL 파인튜닝 모델용 추론 전처리 함수.
inference.py를 복사 후 수정:
  - POINT_LANDMARKS_ASL (90개, 얼굴 포함) 사용
  - ASL 스타일 시퀀스 레벨 정규화 지원
"""
import numpy as np

from src.asl_finetune_config import (
    POINT_LANDMARKS_ASL, CHANNELS_ASL, NECK_IDX_IN_SELECTED,
)


def mediapipe_to_openpose_keypoints_with_face(
    pose_result, hand_result, face_result,
    image_width: int, image_height: int,
) -> np.ndarray:
    """
    MediaPipe PoseLandmarker + HandLandmarker + FaceLandmarker 결과를
    OpenPose 137포인트 배열(pose 25 + face 70 + lhand 21 + rhand 21)로 변환.

    ASL 파인튜닝 모델은 face keypoints가 필요하므로 FaceLandmarker도 사용.
    face_result가 None이면 face 영역을 0으로 채웁니다.

    Returns:
        np.ndarray shape (137, 3): x, y, confidence
    """
    pose  = np.zeros((25, 3), dtype=np.float32)
    face  = np.zeros((70, 3), dtype=np.float32)
    lhand = np.zeros((21, 3), dtype=np.float32)
    rhand = np.zeros((21, 3), dtype=np.float32)

    def to_px(landmark):
        return [
            landmark.x * image_width,
            landmark.y * image_height,
            getattr(landmark, 'visibility', 1.0),
        ]

    # ── Pose (OpenPose BODY_25 매핑) ──────────────────────────────────────────
    if pose_result and pose_result.pose_landmarks:
        mp_pose = pose_result.pose_landmarks[0]
        pose[0]  = to_px(mp_pose[0])   # Nose
        # Neck = 양쪽 어깨 중간점
        ls = to_px(mp_pose[11]); rs = to_px(mp_pose[12])
        pose[1]  = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, 1.0]
        pose[2]  = to_px(mp_pose[12])  # RShoulder
        pose[3]  = to_px(mp_pose[14])  # RElbow
        pose[4]  = to_px(mp_pose[16])  # RWrist
        pose[5]  = to_px(mp_pose[11])  # LShoulder
        pose[6]  = to_px(mp_pose[13])  # LElbow
        pose[7]  = to_px(mp_pose[15])  # LWrist

    # ── Face (MediaPipe FaceLandmarker → OpenPose Dlib 70점 근사 매핑) ────────
    # FaceLandmarker가 없거나 미탐지 시 0 유지
    if face_result and hasattr(face_result, 'face_landmarks') and face_result.face_landmarks:
        mp_face = face_result.face_landmarks[0]  # 첫 번째 사람

        def face_to_px(lm):
            return [lm.x * image_width, lm.y * image_height, 1.0]

        # MediaPipe Face Mesh (478개) → OpenPose Dlib 68+2점 근사 매핑
        # 입술 (OpenPose 48-67): MediaPipe lip landmarks
        mp_lip_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # outer upper
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146  # outer lower
        ]
        for i, idx in enumerate(mp_lip_indices):
            if idx < len(mp_face):
                face[48 + i] = face_to_px(mp_face[idx])

        # 눈 (OpenPose 36-41 right, 42-47 left): MediaPipe eye landmarks
        mp_reye_indices = [33, 7, 163, 144, 145, 133]
        mp_leye_indices = [263, 249, 390, 373, 374, 362]
        for i, idx in enumerate(mp_reye_indices):
            if idx < len(mp_face):
                face[36 + i] = face_to_px(mp_face[idx])
        for i, idx in enumerate(mp_leye_indices):
            if idx < len(mp_face):
                face[42 + i] = face_to_px(mp_face[idx])

        # 코 (OpenPose 27-35): MediaPipe nose bridge
        mp_nose_indices = [1, 2, 98, 327, 4, 5, 195, 197, 6]
        for i, idx in enumerate(mp_nose_indices):
            if idx < len(mp_face):
                face[27 + i] = face_to_px(mp_face[idx])

    # ── Hands ─────────────────────────────────────────────────────────────────
    if hand_result and hand_result.hand_landmarks:
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            kp = np.array([[lm.x * image_width, lm.y * image_height, 1.0] for lm in landmarks])
            if handedness[0].category_name == "Left":
                lhand = kp
            else:
                rhand = kp

    return np.concatenate([pose, face, lhand, rhand], axis=0)  # (137, 3)


def asl_preprocess_sequence(
    sequence: np.ndarray,
    max_len: int,
    use_asl_norm: bool = True,
) -> np.ndarray:
    """
    실시간 추론용 전처리 파이프라인 (ASL 파인튜닝 모델).

    Args:
        sequence: (T, 137, 3) OpenPose 137점 + confidence
        max_len: 최대 시퀀스 길이 (380)
        use_asl_norm: True이면 시퀀스 레벨 ASL 정규화, False이면 bounding box 정규화

    Returns:
        np.ndarray (max_len, 540)
    """
    seq_xy = np.array(sequence)[..., :2]   # (T, 137, 2) — x, y만 사용
    original_len = min(len(seq_xy), max_len)

    # 길이 조정
    if len(seq_xy) > max_len:
        seq_xy = seq_xy[:max_len]
    else:
        pad = np.zeros((max_len - len(seq_xy), seq_xy.shape[1], 2), dtype=np.float32)
        seq_xy = np.concatenate([seq_xy, pad], axis=0)  # (max_len, 137, 2)

    valid = seq_xy[:original_len]  # 실제 데이터 부분만

    if use_asl_norm:
        # ── ASL 스타일: 시퀀스 레벨 정규화 ──────────────────────────────────
        # POINT_LANDMARKS_ASL로 90개 선택
        selected_valid = valid[:, POINT_LANDMARKS_ASL, :]  # (valid_T, 90, 2)
        full_selected = seq_xy[:, POINT_LANDMARKS_ASL, :]  # (max_len, 90, 2)
        full_selected[original_len:] = 0.0

        # 목 포인트 중심화
        neck = selected_valid[:, NECK_IDX_IN_SELECTED:NECK_IDX_IN_SELECTED + 1, :]  # (valid_T, 1, 2)
        neck_mean = np.mean(neck, axis=0, keepdims=True)  # (1, 1, 2)
        full_selected = full_selected - neck_mean

        # 시퀀스 전체 std 정규화
        valid_vals = full_selected[:original_len]
        std = np.std(valid_vals[~np.isnan(valid_vals)])
        if std > 1e-6:
            full_selected = full_selected / std

        selected = full_selected  # (max_len, 90, 2)
    else:
        # ── 기존 bounding box 정규화 ─────────────────────────────────────────
        all_pts = valid.reshape(-1, 2)
        x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
        y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
        cx = (x_max + x_min) / 2
        cy = (y_max + y_min) / 2
        scale = max(x_max - x_min, y_max - y_min) / 2
        if scale < 1e-6:
            scale = 1.0
        norm_seq = (seq_xy - [cx, cy]) / scale
        norm_seq = np.clip(norm_seq, -2, 2)
        selected = norm_seq[:, POINT_LANDMARKS_ASL, :]  # (max_len, 90, 2)

    selected = np.nan_to_num(selected, nan=0.0)

    # ── velocity + acceleration 추가 ─────────────────────────────────────────
    dx = np.zeros_like(selected)
    dx[:-1] = selected[1:] - selected[:-1]

    dx2 = np.zeros_like(selected)
    dx2[:-2] = selected[2:] - selected[:-2]

    # (max_len, 90, 6) → (max_len, 540)
    combined = np.concatenate([selected, dx, dx2], axis=-1)
    return combined.reshape(max_len, -1).astype(np.float32)
