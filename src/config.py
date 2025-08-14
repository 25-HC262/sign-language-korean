import numpy as np

# Model parameters
THRESH_HOLD = 0.5
SEQ_LEN = 60
ROWS_PER_FRAME = 137
MAX_LEN = 384
CROP_LEN = MAX_LEN
NUM_CLASSES = 5
PAD = -100.

# Training parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 300
#VALIDATION_SPLIT = 0.2

# ============= POSE KEYPOINTS (0-24) =============
# OpenPose BODY_25 model keypoints
BODY_KEYPOINTS = {
    'Nose': 0,
    'Neck': 1,
    'RShoulder': 2,
    'RElbow': 3,
    'RWrist': 4,
    'LShoulder': 5,
    'LElbow': 6,
    'LWrist': 7,
    'MidHip': 8,
    'RHip': 9,
    'RKnee': 10,
    'RAnkle': 11,
    'LHip': 12,
    'LKnee': 13,
    'LAnkle': 14,
    'REye': 15,
    'LEye': 16,
    'REar': 17,
    'LEar': 18,
    'LBigToe': 19,
    'LSmallToe': 20,
    'LHeel': 21,
    'RBigToe': 22,
    'RSmallToe': 23,
    'RHeel': 24
}

# 상체 포인트
POSE = [
    #0,   # Nose
    1,   # Neck (normalization reference)
    2,   # RShoulder
    3,   # RElbow
    4,   # RWrist
    5,   # LShoulder
    6,   # LElbow
    7,   # LWrist
    #15,  # REye
    #16,  # LEye
    #17,  # REar
    #18   # LEar
]

# 팔 관련 포인트
LPOSE = [5, 6, 7]  # Left arm (LShoulder, LElbow, LWrist)
RPOSE = [2, 3, 4]  # Right arm (RShoulder, RElbow, RWrist)

'''
# ============= FACE KEYPOINTS (25-94) =============
# 0-16: 턱선 (jaw line)
# 17-21: 오른쪽 눈썹
# 22-26: 왼쪽 눈썹
# 27-35: 코 (nose bridge + nostrils)
# 36-41: 오른쪽 눈
# 42-47: 왼쪽 눈
# 48-67: 입술 (outer + inner lips)

# 얼굴 전체 포인트
FACE_FEATURES = list(range(25, 95))  # 모든 얼굴 포인트 (70개)

# 입 영역
LIP = list(range(25 + 48, 25 + 68))  # 입술 포인트 (20개)

# 눈 영역
REYE = list(range(25 + 36, 25 + 42))  # 오른쪽 눈 (6개)
LEYE = list(range(25 + 42, 25 + 48))  # 왼쪽 눈 (6개)

# 코 영역
NOSE_FACE = list(range(25 + 27, 25 + 36))  # 코 (9개)

# 눈썹 영역
REYEBROW = list(range(25 + 17, 25 + 22))  # 오른쪽 눈썹 (5개)
LEYEBROW = list(range(25 + 22, 25 + 27))  # 왼쪽 눈썹 (5개)
'''

# ============= HAND KEYPOINTS =============
# 왼손 (95-115): 21개 포인트
LHAND = list(range(95, 116))

# 오른손 (116-136): 21개 포인트  
RHAND = list(range(116, 137))

# Hand keypoint 상세
HAND_LANDMARKS = {
    'WRIST': 0,
    'THUMB_CMC': 1,
    'THUMB_MCP': 2,
    'THUMB_IP': 3,
    'THUMB_TIP': 4,
    'INDEX_MCP': 5,
    'INDEX_PIP': 6,
    'INDEX_DIP': 7,
    'INDEX_TIP': 8,
    'MIDDLE_MCP': 9,
    'MIDDLE_PIP': 10,
    'MIDDLE_DIP': 11,
    'MIDDLE_TIP': 12,
    'RING_MCP': 13,
    'RING_PIP': 14,
    'RING_DIP': 15,
    'RING_TIP': 16,
    'PINKY_MCP': 17,
    'PINKY_PIP': 18,
    'PINKY_DIP': 19,
    'PINKY_TIP': 20
}

# ============= SELECTED LANDMARKS FOR MODEL =============
# 한국 수어 인식에 최적화된 키포인트 선택
# 총 95개 포인트 선택

'''POINT_LANDMARKS = (
    # 상체 포즈 (12개)
    POSE +
    
    # 얼굴 표정
    LIP +           # 입 모양 (20개) - 의문문/평서문 구분
    REYE +          # 오른쪽 눈 (6개)
    LEYE +          # 왼쪽 눈 (6개)
    REYEBROW +      # 오른쪽 눈썹 (5개) - 감정/의문 표현
    LEYEBROW +      # 왼쪽 눈썹 (5개)
    
    # 손
    LHAND +         # 왼손 전체 (21개)
    RHAND           # 오른손 전체 (21개)
)'''

POINT_LANDMARKS = (
    POSE +
    LHAND +
    RHAND
)

assert all(0 <= idx < 137 for idx in POINT_LANDMARKS), "Invalid landmark indices!"
assert len(set(POINT_LANDMARKS)) == len(POINT_LANDMARKS), "Duplicate landmarks!"

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES  # x, y, dx, dy, dx2, dy2 for each point

# ============= DATA PATHS =============
KSL_DATA_PATH = "data"
KSL_TRAIN_PATH = "data/train"
KSL_VAL_PATH = "data/val"

# ============= KOREAN SIGN LANGUAGE SENTENCES =============
KSL_SENTENCES = {
    'NIA_SL_SEN0181': '도와주세요',
    'NIA_SL_SEN0354': '안녕하세요',
    'NIA_SL_SEN0355': '감사합니다',
    'NIA_SL_SEN0356': '죄송합니다',
    'NIA_SL_SEN2000': '수고하셨습니다'
}
DIRECTIONS = ['D', 'F', 'L', 'R', 'U']

if __name__ == "__main__":
    print(f"Number of selected keypoints: {NUM_NODES}")
    print(f"Feature dimension: {CHANNELS}")
    print(f"Number of classes: {len(KSL_SENTENCES)}")
    print(f"Expected model input shape: ({MAX_LEN}, {CHANNELS})")