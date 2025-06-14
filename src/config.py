# src/config.py - OpenPose 버전
import numpy as np

# Model parameters
THRESH_HOLD = 0.5
SEQ_LEN = 30
ROWS_PER_FRAME = 137  # OpenPose: 25(body) + 70(face) + 21×2(hands)
MAX_LEN = 384
CROP_LEN = MAX_LEN
NUM_CLASSES = 5  # Korean sign language sentences
PAD = -100.

# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 100
VALIDATION_SPLIT = 0.2

# OpenPose keypoint structure:
# Body: 0-24 (25 points)
# Face: 25-94 (70 points)  
# Left Hand: 95-115 (21 points)
# Right Hand: 116-136 (21 points)

# Body keypoints (important for sign language)
BODY_KEYPOINTS = {
    'Nose': 0, 'Neck': 1,
    'RShoulder': 2, 'RElbow': 3, 'RWrist': 4,
    'LShoulder': 5, 'LElbow': 6, 'LWrist': 7,
    'MidHip': 8,
    'RHip': 9, 'RKnee': 10, 'RAnkle': 11,
    'LHip': 12, 'LKnee': 13, 'LAnkle': 14,
    'REye': 15, 'LEye': 16,
    'REar': 17, 'LEar': 18,
    'LBigToe': 19, 'LSmallToe': 20, 'LHeel': 21,
    'RBigToe': 22, 'RSmallToe': 23, 'RHeel': 24
}

# 선택적 랜드마크들
# Focus on upper body, face, and hands

# Upper body points (more relevant for sign language)
POSE = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]  # Nose, neck, shoulders, arms, eyes, ears
LPOSE = [5, 6, 7]  # Left arm
RPOSE = [2, 3, 4]  # Right arm

# Face points - all are important for Korean sign language
# Select key facial features
FACE_FEATURES = list(range(25, 95))  # All face points (70 points)

# For efficiency, we can select subset of face points
# Mouth region (important for Korean sign language)
LIP = list(range(73, 93))  # Approximate mouth region (20 points)

# Eye regions
REYE = list(range(61, 67))  # Right eye region (6 points)
LEYE = list(range(67, 73))  # Left eye region (6 points)

# Nose region
NOSE_FACE = list(range(52, 61))  # Nose region in face keypoints (9 points)

# Hand points - all are crucial
LHAND = list(range(95, 116))   # All left hand: 21 points
RHAND = list(range(116, 137))  # All right hand: 21 points

# All selected landmarks for the model
# Total: 12 (pose) + 20 (lip) + 6 (reye) + 6 (leye) + 9 (nose) + 21 (lhand) + 21 (rhand) = 95 points
POINT_LANDMARKS = (
    POSE +          # Upper body (12 points)
    LIP +           # Mouth expressions (20 points)
    REYE +          # Right eye (6 points)
    LEYE +          # Left eye (6 points)
    NOSE_FACE +     # Nose region (9 points)
    LHAND +         # Left hand (21 points)
    RHAND           # Right hand (21 points)
)

# Verify all indices are valid
assert all(0 <= idx < 137 for idx in POINT_LANDMARKS), "Invalid landmark indices!"
assert len(set(POINT_LANDMARKS)) == len(POINT_LANDMARKS), "Duplicate landmarks!"

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES  # x, y, dx, dy, dx2, dy2 for each point

# Data paths
KSL_DATA_PATH = "data"
KSL_TRAIN_PATH = "data/train"
KSL_VAL_PATH = "data/val"

# Korean Sign Language sentences
KSL_SENTENCES = {
    'NIA_SL_SEN0181': '도와주세요',
    'NIA_SL_SEN0354': '안녕하세요',
    'NIA_SL_SEN0355': '감사합니다',
    'NIA_SL_SEN0356': '죄송합니다',
    'NIA_SL_SEN2000': '수고하셨습니다'
}

if __name__ == "__main__":
    print(f"Number of selected keypoints: {NUM_NODES}")
    print(f"Feature dimension: {CHANNELS}")
    print(f"Number of classes: {len(KSL_SENTENCES)}")
    print(f"Expected model input shape: ({MAX_LEN}, {CHANNELS})")