"""
ISLR 추론을 위한 랜드마크 추출 유틸리티.

MediaPipe Tasks API (pose/hand/face landmarker) 결과를
ISLR 모델 입력 형식 (543, 3)으로 변환합니다.

레이아웃: face(468) + lhand(21) + pose(33) + rhand(21) = 543
  → sign-language/src/landmarks_extraction.py:extract_coordinates() 와 동일한 레이아웃
"""
import numpy as np


def extract_coordinates_for_islr(pose_result, hand_result, face_result) -> np.ndarray:
    """
    Tasks API 결과 → (543, 3) float32 배열.

    - face_result.face_landmarks[0][:468]: 표준 face mesh 468 포인트
      (FaceLandmarker는 478개를 반환하며 마지막 10개가 iris)
    - hand_result: handedness로 left/right 구분
    - pose_result.pose_landmarks[0]: 33개 포즈 랜드마크

    좌표 값은 모두 정규화된 [0, 1] 범위 (x, y, z).
    랜드마크 미검출 시 해당 영역은 np.nan으로 채움.
    """
    # ── Face (0:468) ──────────────────────────────────────────────────────────
    if face_result and face_result.face_landmarks:
        lms = face_result.face_landmarks[0]
        face = np.array([[lm.x, lm.y, lm.z] for lm in lms[:468]], dtype=np.float32)
        if len(face) < 468:
            pad = np.full((468 - len(face), 3), np.nan, dtype=np.float32)
            face = np.vstack([face, pad])
    else:
        face = np.full((468, 3), np.nan, dtype=np.float32)

    # ── Hands (lhand 468:489, rhand 522:543) ─────────────────────────────────
    lhand = np.full((21, 3), np.nan, dtype=np.float32)
    rhand = np.full((21, 3), np.nan, dtype=np.float32)
    if hand_result and hand_result.hand_landmarks:
        for lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            hand_arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            # MediaPipe Tasks API handedness: "Left" / "Right" (카메라 기준)
            if handedness[0].category_name == "Left":
                lhand = hand_arr
            else:
                rhand = hand_arr

    # ── Pose (489:522) ────────────────────────────────────────────────────────
    if pose_result and pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]],
                        dtype=np.float32)
    else:
        pose = np.full((33, 3), np.nan, dtype=np.float32)

    return np.concatenate([face, lhand, pose, rhand])  # (543, 3)
