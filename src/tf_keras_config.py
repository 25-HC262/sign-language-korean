import os
import warnings

# 1. 경고 차단
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_LOG_LEVEL"] = "3"

# 2. 캐라스-텐서플로우 백엔드 설정
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf

# 3. GPU 동적 할당
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("WARNING: 지금 GPU가 아니라 CPU를 쓰고 있습니다!")
else:
    # 현재 인식된 GPU 개수와 상세 명칭 출력
    print(f"GPU 사용 중. 현재 사용 가능한 GPU 개수: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f" - GPU [{i}]: {gpu}")
# 1. 경고 설정
tf.get_logger().setLevel('ERROR')