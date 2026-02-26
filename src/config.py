import argparse
import datetime
import json
import os

# Model parameters
THRESHOLD = 0.5
SEQ_LEN = 60
ROWS_PER_FRAME = 137 # 제거 필요.
MAX_LEN = 125
CROP_LEN = MAX_LEN
NUM_CLASSES = 5
PAD = 0. #-100.

# Training parameters
LEARNING_RATE = 0.0001
LEARNING_RATE_FOR_UMAP = 0.001
BATCH_SIZE = 32
BATCH_SIZE_FOR_UMAP = 1024
WEIGHT_DECAY = 0.01
EPOCHS = 300
EPOCHS_FOR_UMAP = 100
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.05
UMAP_OUTPUT_DIM = 32
OUTPUT_DIM = 98

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
DIM = 2 # 현재 2D이므로
CHANNELS = DIM * NUM_NODES  # x, y for each point

# ==========================================
# 전역 패스(Path) 변수 설정
# ==========================================
"""
1. 다운로드/업로드 저장소 선택
    - 다운로드: -s 혹은 --storage 뒤에 L,S,G 중 하나를 받도록 설정
        명령어 예시: `python -m model.gloss_transformer -s L` 혹은 `-storage L`
    - 업로드: -u 혹은 --upload 뒤에 L,S,G 중 하나를 받도록 설정
        명령어 예시: `python -m model.gloss_transformer -u G` 혹은 `-upload G`
2. umap 모델 선택
    -u 혹은 --umap 뒤에 모델명
    명령어 예시: `python -m model.gloss_transformer -u "umap.keras"` 혹은 `--umap "umap.keras"`
3. gloss 모델 선택
    -g 혹은 --gm 혹은 --gloss_model 뒤에 모델명
    명령어 예시: `python -m model.gloss_transformer -g "gloss_transformer.keras"` 혹은 `--gt "gloss_transformer.keras"` 혹은 `--gloss_model "gloss_transformer.keras"`
4. gloss 모델 종류 선택
    --gmt 혹은 --gloss_model_type으로 모델 종류 선택
    명령어 예시: `python -m model.gloss_transformer --gmt "transformer"` 혹은 `python -m model.gloss_transformer --gloss_model_type "transformer"`
"""
def get_config_args():
    parser = argparse.ArgumentParser()

    # 저장소 선택 옵션 - args.storage에 저장
    parser.add_argument(
        "-s", "--storage",
        choices=["L", "S", "G"],
        default="L",
        help="Storage type: L(Local), S(S3), G(GCS)"
    )

    # 업로드 선택 옵션 - args.upload에 저장
    parser.add_argument(
        "-u", "--upload",
        choices=["L", "S", "G"],
        default="L",
        help="Storage type: L(Local), S(S3), G(GCS)"    )

    # umap 모델 선택 옵션 - args.umap에 저장
    parser.add_argument(
        "-u", "--umap",
        default="encoder.keras"
    )
    # gloss 모델 선택 옵션 - args.gm에 저장
    parser.add_argument(
        "-g", "--gm", "--gloss_model",
        default="sign_language_v1.h5"
    )

    # gloss 모델 종류 선택 옵션
    parser.add_argument(
        "--gmt", "--gloss_model_type",
        default="transformer"
    )

    args, _ = parser.parse_known_args()
    print(*(f"   > {'[default]' if v==parser.get_default(k) else ''} {k}: {v} selected." for k,v in vars(args).items()), sep='\n')
    return args

args = get_config_args()
STORAGE_MODE = args.storage
UPLOAD_MODE = args.upload
SELECTED_GM_TYPE = args.gmt

# 경로 안정화
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONFIG_DIR, ".."))

date_idx = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")

"""TO-DO: 기본 버킷명으로 통일"""
base_map = {
    "G": ("gs://openpose-keypoint", "gs://trout-models/umap_models", "gs://trout-models/gloss_models", "gs://trout-models/checkpoints"),
    "S": ("s3://openpose-keypoints", "s3://trout-model/umap_models", "s3://trout-model/gloss_models", "s3://trout-model/checkpoints"),
    "L": ("data/openpose_keypoints", "models/umap_models", "models/gloss_models", "models/checkpoints")
}
# 새로운 모델 저장
names = {
    "gm": f"gloss_{SELECTED_GM_TYPE}_{date_idx}",
    "umap": f"umap_encoder_{date_idx}"
}
files = {
    "gm_keras": f"{names['gm']}.keras", # .h5보다 .keras가 권장되므로 .keras로 통일할 것.
    "gm_tflite": f"{names['gm']}.tflite",
    "umap_keras": f"{names['umap']}.keras"
}

# 로컬 베이스는 항상 필요
L_DATA, L_UMAP, L_GM, L_CKPT = (os.path.join(PROJECT_ROOT, L_PATH) for L_PATH in base_map['L'])
L_TOOLS = os.path.join(PROJECT_ROOT, "tools")
for path in [L_CKPT, L_GM, L_UMAP, L_TOOLS]:
    os.makedirs(path, exist_ok=True)
print(f"[*] Local Project Path Initialized at: {PROJECT_ROOT}")

# LOAD_BASE: 사용자 선택 모드(UPLOAD_MODE)에서 가져옴
LOAD_DATA, LOAD_UMAP, LOAD_GM, _ = base_map.get(UPLOAD_MODE, base_map["L"])

# 최종 경로
UMAP_LOAD_PATH = f'{LOAD_UMAP}/{args.umap}' # GM 학습 & 프로젝트에 사용되는 최적 UMAP MODEL
GM_LOAD_PATH = f'{LOAD_GM}/{args.gm}'       # 프로젝트에 사용되는 최적 GLOSS MODEL

# 저장 경로
LOCAL_PATHS = {
    "gm_ckpt": f"{L_CKPT}/{files['gm_keras']}",
    "gm_final": f"{L_GM}/{files['gm_keras']}",
    "gm_tflite": f"{L_GM}/{files['gm_tflite']}",
    "umap_ckpt": f"{L_UMAP}/{files['umap_keras']}",
    "umap_final": f"{L_UMAP}/{files['umap_keras']}"
}

# ============== WandB 설정 ==============
WANDB_GM_PROJECT = f"grad-gloss-{SELECTED_GM_TYPE}-training"
WANDB_GM_NAME = f"gloss-{SELECTED_GM_TYPE}-{date_idx}"
WANDB_UMAP_PROJECT = f"grad-umap-training"
WANDB_UMAP_NAME = f"umap-{date_idx}"

# ========= WandB run organization (grouping & tagging) 설정 =========
WANDB_GM_GROUP = SELECTED_GM_TYPE
WANDB_UMAP_GROUP = "umap"
WANDB_GM_TAGS = [SELECTED_GM_TYPE, f"dim{OUTPUT_DIM}", f"classes{NUM_CLASSES}"]
WANDB_UMAP_TAGS = ["umap", f"dim{OUTPUT_DIM}"]
# ============== optuna 설정 ==============
OPTUNA_TRIALS_PATH = "sqlite:///optuna_trials.db" # 로컬 수정 필요
SUBSET_RATIO = 0.05
OPTUNA_STUDY_NAME = "transformers_optuna_study"
OPTUNA_MODEL = "transformer"
BEST_PARAMS_PATH = f"{L_GM}/best_params-{date_idx}.json"
N_TRIALS = 20

# ============= KOREAN SIGN LANGUAGE SENTENCES =============
try:
    with open('src/primary_label_map.json', 'r', encoding='utf-8') as f:
        KSL_SENTENCES = json.load(f)
except FileNotFoundError:
    print("Warning: 'label_map.json' not found.")
DIRECTIONS = ['D', 'F', 'L', 'R', 'U']

if __name__ == "__main__":
    print(f"Number of selected keypoints: {NUM_NODES}")
    print(f"Feature dimension: {CHANNELS}")
    print(f"Number of classes: {len(KSL_SENTENCES)}")
    print(f"Expected model input shape: ({MAX_LEN}, {CHANNELS})")