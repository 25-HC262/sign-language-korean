"""
임시 입력 검증용 시각화 유틸리티.
KSL_VISUALIZE=1 환경변수로 활성화되며, 불필요 시 이 파일과 main.py의
# === VISUALIZATION START/END === 블록을 함께 삭제하면 됩니다.

제공 기능:
  1. draw_mediapipe_frame  - MediaPipe 감지 결과를 실제 프레임에 오버레이
  2. draw_openpose_skeleton - OpenPose 변환 결과를 빈 캔버스에 스켈레톤으로 그림
  3. make_validation_image  - 두 이미지를 좌우로 합쳐 JPEG 바이트로 반환
"""

import cv2
import numpy as np

# ── skeleton-visualization.py에서 가져온 스켈레톤 연결 정의 ──────────────────
POSE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # nose→neck→RShoulder→RElbow→RWrist
    (1, 5), (5, 6), (6, 7),            # neck→LShoulder→LElbow→LWrist
]
HAND_PAIRS = [(i, i + 1) for i in range(0, 20)]

# ── 색상 ─────────────────────────────────────────────────────────────────────
COLOR_POSE       = (0, 255, 0)    # 초록
COLOR_LEFT_HAND  = (255, 80, 80)  # 파랑
COLOR_RIGHT_HAND = (80, 80, 255)  # 빨강
COLOR_LABEL      = (255, 255, 0)  # 노랑


def draw_mediapipe_frame(frame: np.ndarray, pose_result, hand_result) -> np.ndarray:
    """MediaPipe 감지 결과를 실제 프레임에 오버레이한 이미지를 반환합니다."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Pose 랜드마크
    if pose_result.pose_landmarks:
        mp_pose = pose_result.pose_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in mp_pose]

        for i, j in POSE_PAIRS:
            if mp_pose[i].visibility > 0.3 and mp_pose[j].visibility > 0.3:
                cv2.line(vis, pts[i], pts[j], COLOR_POSE, 2)
        for i, pt in enumerate(pts):
            if mp_pose[i].visibility > 0.3:
                cv2.circle(vis, pt, 4, COLOR_POSE, -1)

    # Hand 랜드마크
    if hand_result.hand_landmarks:
        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            color = COLOR_LEFT_HAND if label == "Left" else COLOR_RIGHT_HAND
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            wrist_x = int(landmarks[0].x * w)
            wrist_y = int(landmarks[0].y * h)

            for i, j in HAND_PAIRS:
                cv2.line(vis, pts[i], pts[j], color, 2)
            for pt in pts:
                cv2.circle(vis, pt, 3, color, -1)

            # 손 레이블 + wrist_x 표시
            cv2.putText(vis, f"{label} (wx:{landmarks[0].x:.2f})",
                        (wrist_x, wrist_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)

    # 감지 요약 텍스트
    n_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
    pose_ok = bool(pose_result.pose_landmarks)
    cv2.putText(vis, f"pose={'OK' if pose_ok else 'NG'}  hands={n_hands}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LABEL, 2, cv2.LINE_AA)
    cv2.putText(vis, "MediaPipe", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return vis


def draw_openpose_skeleton(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    mediapipe_to_openpose_keypoints()의 출력 (137, 3)을 받아
    빈 캔버스에 스켈레톤을 그린 이미지를 반환합니다.
    keypoints 좌표는 픽셀 단위 (변환 직후, 정규화 전).
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    pose       = keypoints[:25]       # OpenPose BODY_25 (0-24)
    left_hand  = keypoints[95:116]    # LHAND (95-115)
    right_hand = keypoints[116:137]   # RHAND (116-136)

    def pt(kp):
        return (int(kp[0]), int(kp[1]))

    def visible(kp):
        return kp[2] > 0.1 and not (kp[0] == 0 and kp[1] == 0)

    # Pose
    for i, j in POSE_PAIRS:
        if visible(pose[i]) and visible(pose[j]):
            cv2.line(canvas, pt(pose[i]), pt(pose[j]), COLOR_POSE, 2)
    for kp in pose:
        if visible(kp):
            cv2.circle(canvas, pt(kp), 5, COLOR_POSE, -1)

    # Hands
    for hand_pts, color, label in [
        (left_hand,  COLOR_LEFT_HAND,  "L"),
        (right_hand, COLOR_RIGHT_HAND, "R"),
    ]:
        for i, j in HAND_PAIRS:
            if visible(hand_pts[i]) and visible(hand_pts[j]):
                cv2.line(canvas, pt(hand_pts[i]), pt(hand_pts[j]), color, 2)
        for kp in hand_pts:
            if visible(kp):
                cv2.circle(canvas, pt(kp), 3, color, -1)
        if visible(hand_pts[0]):
            cv2.putText(canvas, label, (pt(hand_pts[0])[0], pt(hand_pts[0])[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, "OpenPose (converted)", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def make_validation_image(frame: np.ndarray, pose_result, hand_result,
                          keypoints: np.ndarray) -> bytes:
    """
    좌: MediaPipe 오버레이 / 우: OpenPose 변환 스켈레톤
    두 이미지를 가로로 합쳐 JPEG 바이트로 반환합니다.
    """
    h, w = frame.shape[:2]
    left_img  = draw_mediapipe_frame(frame, pose_result, hand_result)
    right_img = draw_openpose_skeleton(keypoints, w, h)

    combined = np.concatenate([left_img, right_img], axis=1)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
