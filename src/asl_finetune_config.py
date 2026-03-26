"""
ASL 파인튜닝용 설정.
기존 src/config.py에서 import한 뒤, 얼굴 랜드마크 확장 및 파인튜닝 상수를 추가합니다.
"""
from pathlib import Path

# 기존 config의 모든 상수 임포트
from src.config import (
    POSE, LHAND, RHAND, KEYPOINT_DIM, L_CKPT, L_GM, )

# ============= ASL 사전학습 모델 경로 =============
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASL_WEIGHTS_PATH = str(
    _PROJECT_ROOT.parent / "sign-language-korean" / "models" /
    "islr-fp16-192-8-seed_all42-foldall-last.h5"
)

# ============= 파인튜닝 아키텍처 상수 =============
# ASL 백본과 동일한 latent dim (192는 4 heads로 나누어 떨어짐)
KSL_FINETUNE_DIM = 192

# ============= 확장된 랜드마크 선택 =============
# all_keypoints 배열 레이아웃: pose(0-24) + face(25-94) + lhand(95-115) + rhand(116-136)
_FACE_OFFSET = 25  # face가 all_keypoints에서 시작하는 인덱스

# OpenPose face 랜드마크 → ASL MediaPipe 대응 선택
FACE_NOSE_OP = list(range(_FACE_OFFSET + 27, _FACE_OFFSET + 36))   # 9개 → ASL NOSE(4) 대응
FACE_REYE_OP = list(range(_FACE_OFFSET + 36, _FACE_OFFSET + 42))   # 6개 → ASL REYE(16) 대응
FACE_LEYE_OP = list(range(_FACE_OFFSET + 42, _FACE_OFFSET + 48))   # 6개 → ASL LEYE(16) 대응
FACE_LIP_OP  = list(range(_FACE_OFFSET + 48, _FACE_OFFSET + 68))   # 20개 → ASL LIP(40) 대응
FACE_LANDMARKS_OPENPOSE = FACE_NOSE_OP + FACE_REYE_OP + FACE_LEYE_OP + FACE_LIP_OP  # 41개

# 파인튜닝 모델 랜드마크: pose(7) + face(41) + lhand(21) + rhand(21) = 90개
POINT_LANDMARKS_ASL = POSE + FACE_LANDMARKS_OPENPOSE + LHAND + RHAND

NUM_NODES_ASL = len(POINT_LANDMARKS_ASL)           # 90
CHANNELS_ASL = KEYPOINT_DIM * NUM_NODES_ASL * 3   # 90 * 2 * 3 = 540

# 목(Neck) 랜드마크: OpenPose BODY_25의 Neck = index 1
# POINT_LANDMARKS_ASL에서 Neck의 위치 (POSE[0] = 1 = Neck)
NECK_IDX_IN_SELECTED = POINT_LANDMARKS_ASL.index(1)  # = 0 (POSE의 첫 번째)

# ============= 파인튜닝 학습 파라미터 =============
FINETUNE_STAGE1_LR = 1e-3
FINETUNE_STAGE2_LR = 1e-4
FINETUNE_STAGE3_LR = 5e-6
FINETUNE_STAGE1_EPOCHS = 10
FINETUNE_STAGE2_EPOCHS = 20
FINETUNE_STAGE3_EPOCHS = 30
FINETUNE_LABEL_SMOOTHING = 0.1

# ============= 저장 경로 =============
FINETUNE_CKPT_DIR = L_CKPT
FINETUNE_MODEL_DIR = L_GM

if __name__ == "__main__":
    print(f"NUM_NODES_ASL: {NUM_NODES_ASL}")
    print(f"CHANNELS_ASL:  {CHANNELS_ASL}")
    print(f"NECK_IDX_IN_SELECTED: {NECK_IDX_IN_SELECTED}")
    print(f"POINT_LANDMARKS_ASL (first 10): {POINT_LANDMARKS_ASL[:10]}")
    print(f"ASL_WEIGHTS_PATH: {ASL_WEIGHTS_PATH}")
    print(f"KSL_FINETUNE_DIM: {KSL_FINETUNE_DIM}")
