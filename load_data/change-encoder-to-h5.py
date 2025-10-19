# Windows에서 실행
import tensorflow as tf
import json

# 모델 로드 (Windows에서는 되니까)
model = tf.keras.models.load_model('../models/umap_models/encoder.keras')

# 가중치만 저장
model.save_weights('../models/umap_models/encoder.weights.h5')

# 모델 구조 확인 및 저장
print("Model summary:")
model.summary()

# Config 저장
config = model.get_config()
with open('../models/umap_models/encoder_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\nSaved files:")
print("- encoder_weights.h5")
print("- encoder_config.json")