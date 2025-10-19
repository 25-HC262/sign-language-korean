# Windows 환경에서 실행
import tensorflow as tf

# 기존 모델 로드
model = tf.keras.models.load_model('../models/umap_models/encoder.keras')

# 모델 구조만 가져와서 재저장
model.save('../models/umap_models/encoder_v2.keras', save_format='keras')

# 또는 H5 포맷으로 저장
model.save('../models/umap_models/encoder.h5')