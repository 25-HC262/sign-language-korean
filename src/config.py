import argparse
import datetime
import json
from enum import Enum
from pathlib import Path
from typing import List, Tuple

# ============= 데이터를 위한 클래스 =============
class DataType(Enum):
    GLOSS = "gloss"
    SENTENCE = "sentence"

class DataDim(Enum):
    _2D = "2d"
    _3D = "3d"
    @classmethod
    def _missing_(cls, value):
        if value == 2:
            return cls._2D
        if value == 3:
            return cls._3D
        return super()._missing_(value)

class Trainer(Enum):
    GM = "gm"
    UMAP = "umap"

# TO-DO: 기본 버킷명으로 통일
# 경로 관련

# 경로 안정화
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"[*] Local Project Path Initialized at: {PROJECT_ROOT}")

base_map = {
    "G": ("gs://openpose-keypoint", "gs://trout-models/umap_models", "gs://trout-models/gloss_models", "gs://trout-models/checkpoints", "gs://test-openpose-keypoint", "gs://trout-models/data/tools", "gs://trout-models/data/processed"),
    "S": ("s3://openpose-keypoints", "s3://trout-model/umap_models", "s3://trout-model/gloss_models", "s3://trout-model/checkpoints", "s3://test-openpose-keypoints", "s3://trout-models/data/tools", "s3://trout-models/data/processed"),
    "L": ("data/openpose-keypoints", "models/umap_models", "models/gloss_models", "models/checkpoints", "data/test-openpose-keypoints", "tools/", "data/processed")
}
date_idx = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")

L_DATA, L_UMAP, L_GM, L_CKPT, L_TEST, L_TOOLS, L_PREPROCESSED_DATA = (PROJECT_ROOT / p for p in base_map['L']) # 로컬 베이스는 항상 필요
L_PARAMS = L_GM / "best_params"
for p in [L_CKPT, L_GM, L_UMAP, L_TOOLS, L_TEST, L_PARAMS]:
    p.mkdir(parents=True, exist_ok=True)

class PathConfig:
    def __init__(self, args):
        # 1. 실행 시점의 인자값 확정
        self.TRAINER = args.trainer
        self.STORAGE_MODE = args.storage
        self.UPLOAD_MODE = args.upload
        self.UMAP_MODEL = getattr(args, 'umap', 'encoder.keras')
        self.KEYPOINT_DIM = getattr(args, 'kpt_dim', 2) # default 2
        self.GM_MODEL = getattr(args, 'gm', "gloss_transformer_2026_03_25_16-22.keras")
        self.SELECTED_GM_TYPE = getattr(args, 'gmt', "transformer")
        self.UMAP_OUTPUT_DIM = getattr(args, 'u_dim', 32)
        self.UMAP_MODEL = getattr(args, 'umap', 'encoder.keras')
        self.USE_UMAP = getattr(args, 'use_umap', False)

        # 새로운 모델 저장
        self.names = {
            "gm": f"gloss_{self.SELECTED_GM_TYPE}_{date_idx}",
            "umap": f"umap_encoder_{date_idx}-dim-{self.UMAP_OUTPUT_DIM}"
        }
        files = {
            "gm_keras": f"{self.names['gm']}.keras", # .h5보다 .keras가 권장되므로 .keras로 통일할 것.
            "optuna_gm_keras": f"optuna-{self.names['gm']}.keras",
            "gm_tflite": f"{self.names['gm']}.tflite",
            "umap_ckpt": f"{self.names['umap']}-dim-{self.UMAP_OUTPUT_DIM}.weights.h5",
            "umap_keras": f"{self.names['umap']}.keras"
        }

        # LOAD_BASE: 사용자 선택 모드(UPLOAD_MODE)에서 가져옴
        self.LOAD_DATA, self.LOAD_UMAP, self.LOAD_GM, _, self.LOAD_TEST, self.LOAD_TOOLS, self.LOAD_PREPROCESSED_DATA = base_map.get(self.UPLOAD_MODE, base_map['L'])

        # 최종 경로
        self.UMAP_LOAD_PATH = f'{self.LOAD_UMAP}/{self.UMAP_MODEL}' # GM 학습 & 프로젝트에 사용되는 최적 UMAP MODEL
        self.GM_LOAD_PATH = f'{self.LOAD_GM}/{self.GM_MODEL}'       # 프로젝트에 사용되는 최적 GLOSS MODEL

        # 저장 경로
        self.LOCAL_PATHS = {
            "gm_ckpt": f"{L_CKPT}/{files['gm_keras']}",
            "optuna_gm_ckpt": f"{L_CKPT}/{files['optuna_gm_keras']}",
            "gm_final": f"{L_GM}/{files['gm_keras']}",
            "gm_tflite": f"{L_GM}/{files['gm_tflite']}",
            "umap_ckpt": f"{L_CKPT}/{files['umap_ckpt']}",
            "umap_final": f"{L_UMAP}/{files['umap_keras']}"
        }

        # ============== 학습을 위한 모델 설정 ==============
        self.CHANNELS = self.KEYPOINT_DIM * NUM_NODES  # x, y for each point
        self.OUTPUT_DIM = self.CHANNELS

        # ============== WandB 설정 ==============
        self.WANDB_GM_PROJECT = f"grad-gloss-{self.SELECTED_GM_TYPE}-training"
        self.WANDB_GM_NAME = f"gloss-{self.SELECTED_GM_TYPE}-{date_idx}"
        self.WANDB_UMAP_PROJECT = f"grad-umap-training"
        self.WANDB_UMAP_NAME = f"umap-{date_idx}"

        # ========= WandB run organization (grouping & tagging) 설정 =========
        self.WANDB_GM_GROUP = self.SELECTED_GM_TYPE
        self.WANDB_UMAP_GROUP = "umap"
        self.WANDB_GM_TAGS = [self.SELECTED_GM_TYPE, f"dim{self.OUTPUT_DIM}", f"classes{NUM_CLASSES}"]
        self.WANDB_UMAP_TAGS = ["umap", f"dim{self.OUTPUT_DIM}"]

# ============= KOREAN SIGN LANGUAGE SENTENCES =============
# 라벨링 데이터
label_map_name = "label_map.json"
LABEL_MAP_PATH = PROJECT_ROOT / "src" / label_map_name
if LABEL_MAP_PATH.exists():
    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        KSL_SENTENCES = json.load(f)
else:
    raise FileNotFoundError(f"라벨 맵 파일을 찾을 수 없습니다: {LABEL_MAP_PATH}")
DIRECTIONS = ['D', 'F', 'L', 'R', 'U']

# Model parameters
THRESHOLD = 0.5
SEQ_LEN = 60  # 현재 test에서 사용 중. 통일 필요한지 고민할 것.
ROWS_PER_FRAME = 137 # 제거 필요.
CROP_LEN = 281
MAX_LEN = 380 # CROP_LEN # 최대 프레임 길이 (매번 계산할 수 없으므로 수동 계산)
NUM_CLASSES = len(KSL_SENTENCES)
PAD = -100.

# Training parameters
LEARNING_RATE = 0.0001
LEARNING_RATE_FOR_UMAP = 0.001
BATCH_SIZE = 32
BATCH_SIZE_FOR_UMAP = 1024
WEIGHT_DECAY = 0.01
EPOCHS = 200
EPOCHS_FOR_UMAP = 100
VALIDATION_SPLIT = 0.2
TEST_RATE = 1
UMAP_DATA_NUM = 3000
TEST_UMAP_DATA_NUM = 1000
DROPOUT_RATE_FOR_UMAP = 0.2

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

# ==========================================
# 전역 패스(Path) 변수 설정
# ==========================================
def get_config_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = get_base_parser()
    args, rest = parser.parse_known_args()

    # 선택된 trainer에 따른 세부 인자 설정
    if args.trainer == Trainer.UMAP.value:
        return get_umap_args(parser, rest)
    else:
        return get_gm_args(parser, rest)

def get_base_parser():
    """
    상위 인자 설정. 반드시 설정해야 하는 인자들 및 공통 인자 설정을 담당한다.
    1. 모델 선택
        -t 혹은 --trainer로 훈련 객체가 umap인지 gm인지 선택
    2. 다운로드/업로드 저장소 선택
        - 다운로드: -s 혹은 --storage 뒤에 L,S,G 중 하나를 받도록 설정
            명령어 예시: `python -m train.gloss_transformer_train -s L` 혹은 `-storage L`
        - 업로드: -u 혹은 --upload 뒤에 L,S,G 중 하나를 받도록 설정
            명령어 예시: `python -m train.gloss_transformer_train -u G` 혹은 `-upload G`
    3. keypoint 차원수 선택
        -k 혹은 --kpt_dim으로 keypoint 차원수 선택. 2와 3 중 하나
        명령어 예시: `python -m train.gloss_transformer_train -k 2` 혹은 `-k 3`

    :return: GM인지 UMAP인지에 대한 구분
    """
    parser = argparse.ArgumentParser(add_help=False) # 도움말 중복 방지
    # 1. 모델 선택
    parser.add_argument(
        "-t", "--trainer",
        choices=[t.value for t in Trainer],
        default="gm",
        help="Training model type: gm or umap"
    )
    # 2-1. 저장소 선택 옵션 - args.storage에 저장
    parser.add_argument(
        "-s", "--storage",
        choices=["L", "S", "G"],
        default="L",
        help="Storage type: L(Local), S(S3), G(GCS)"
    )
    # 2-2. 업로드 선택 옵션 - args.upload에 저장
    parser.add_argument(
        "-u", "--upload",
        choices=["L", "S", "G"],
        default="G",
        help="Storage type: L(Local), S(S3), G(GCS)"
    )
    # 3. keypoint 차원수 선택 옵션 - args.kpt_dim에 저장
    parser.add_argument(
        "-k", "--kpt_dim",
        type=int,
        choices=[2,3],
        default=2,
        help="Keypoint Dimension type: 2 or 3 (meaning 2D or 3D)"
    )
    return parser

def get_umap_args(parent: argparse.ArgumentParser, rest: List[str]) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    1. umap 축소 차원수(결과) 선택
        --u_dim 뒤에 umap을 통해 축소하고자 하는 차원수 선택
        명령어 예시: `python -m train.reduction_train --u_dim 64`
    :return:
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=False) # 공통 인자 상속
    # 1. umap 축소 차원수 선택 옵션 - args.u_dim에 저장
    parser.add_argument(
        "--u_dim",
        type=int,
        default=32,
        help="Reducing dimension using Umap: default 32"
    )
    args, _ = parser.parse_known_args(rest)
    return parser, args

def get_gm_args(parent: argparse.ArgumentParser, rest: List[str])  -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    1. umap 모델 선택
        --umap 뒤에 모델명
        명령어 예시: `python -m train.gloss_transformer_train --umap "umap.keras"`
    2. 사용할 gloss 모델 선택
        -g 혹은 --gm 혹은 --gloss_model 뒤에 모델명
        명령어 예시: `python -m train.gloss_transformer_train -g "gloss_transformer.keras"` 혹은 `--gt "gloss_transformer.keras"` 혹은 `--gloss_model "gloss_transformer.keras"`
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=False)
    # 1. 차원 축소를 위한 umap 모델 선택 옵션 - args.umap에 저장: 차원 축소를 하지 않을 경우 None일 수 있음; TO-DO
    parser.add_argument(
        "--umap",
        default="encoder.keras"
    )
    parser.add_argument(
        "--use-umap",
        dest="use_umap",
        action="store_true",
        default=False,
        help="UMAP 인코더를 전처리에 사용 (기본값: 미사용)"
    )
    # 2. main 사용 gloss 모델 선택 옵션 - args.gm에 저장
    parser.add_argument(
        "-g", "--gm", "--gloss_model",
        default="gloss_transformer_2026_02_26_06-07.keras"
    )
    # 3. gloss 모델 종류 선택 옵션 - args.gmt 저장
    parser.add_argument(
        "--gmt", "--gloss_model_type",
        default="transformer"
    )
    args, _ = parser.parse_known_args(rest)
    return parser, args

# 최종 경로
UMAP_MODEL = 'encoder.keras'
KEYPOINT_DIM = 2 # default 2
GM_MODEL = "gloss_transformer_2026_03_25_16-22.keras"
USE_UMAP = False  # True로 변경하면 UMAP 인코더 전처리 활성화 (UMAP을 사용하는 모델 전용)
UMAP_LOAD_PATH = f'{L_UMAP}/{UMAP_MODEL}' # GM 학습 & 프로젝트에 사용되는 최적 UMAP MODEL
GM_LOAD_PATH = f'{L_GM}/{GM_MODEL}'       # 프로젝트에 사용되는 최적 GLOSS MODEL

# ============== 학습을 위한 모델 설정 ==============
DATA_TYPE = DataType.GLOSS # default
NUM_NODES = len(POINT_LANDMARKS) # 49
CHANNELS = KEYPOINT_DIM * NUM_NODES  # x, y OR x, y, z for each point
OUTPUT_DIM = CHANNELS
UMAP_OUTPUT_DIM = 32 # default

# ============== optuna 설정 ==============
OPTUNA_TRIALS_PATH = "sqlite:///optuna_trials.db" # 로컬 수정 필요
SUBSET_RATIO = 0.4
OPTUNA_STUDY_NAME = "transformers_optuna_study"
OPTUNA_MODEL = "transformer"
N_TRIALS = 20

if __name__ == "__main__":
    print(f"Number of selected keypoints: {NUM_NODES}")
    print(f"[DEFAULT] Feature dimension: {CHANNELS}")
    print(f"Number of classes: {len(KSL_SENTENCES)}")
    print(f"[DEFAULT] Expected model input shape: ({CROP_LEN}, {CHANNELS})")