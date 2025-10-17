import os
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
from typing import Dict

# 커스텀
from src.backbone import get_model, TFLiteModel
from src.config import MAX_LEN, LEARNING_RATE, EPOCHS, KSL_SENTENCES, DIRECTIONS, VALIDATION_SPLIT, S3_UMAP_MODEL_PATH
from preprocessing.dataset_loader import KSLDataLoader

'''
영상 1개(SEQUENCE) = 키포인트 파일 100-384개(현재 최대 MAX_LEN이 384라)로 구성됨.
    SAMPLE은 시퀀스에 라벨 이미지를 더 가진 데이터.
SAMPLES = 영상(시퀀스)들의 모임!
'''

DIRECTION = DIRECTIONS
LABEL_MAP = KSL_SENTENCES
NUM_CLASSES = len(LABEL_MAP)

umap_encoder_path = os.path.join(os.path.expanduser(S3_UMAP_MODEL_PATH),'/encoder.keras')
encoder = tf.keras.models.load_model(umap_encoder_path)
encoder.summary()

date_idx = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
base_name = f"ksl_model_{date_idx}"
checkpoint_path = f"checkpoints/{base_name}.h5"
save_model = f"models/{base_name}.h5"
tflite_path = f"models/gloss_transformer_models/{base_name}.tflite"

def train_model():
    # Load training data
    print("\nLoading training data...")
    dataset_loader = KSLDataLoader('data/openpose_keypoints', label_map=LABEL_MAP, is_training=True)
    all_samples = dataset_loader.samples
    if not all_samples:
        raise ValueError("No training data found. Please check the path.")
    print(f"Total training samples found: {len(all_samples)}")

    # VALIDATION_SPLIT 비율로 train/test split
    print(f"Using {(1-VALIDATION_SPLIT)*10}/{VALIDATION_SPLIT*10} train/test split.")
    all_samples = dataset_loader.samples
    train_samples = []
    val_samples = []

    grouped_by_folder_name = defaultdict(list)
    # 폴더명 별로 샘플 분리
    for sample in all_samples:
        grouped_by_folder_name[sample['folder_name']].append(sample)

    for folder_name, samples in grouped_by_folder_name.items():
        np.random.shuffle(samples)
        split_train_idx = int((1-VALIDATION_SPLIT)*len(samples))
        train_samples.extend(samples[:split_train_idx])
        val_samples.extend(samples[split_train_idx:])
    print(f"Final dataset split - Train: {len(train_samples)}, Validation: {len(val_samples)}")

    # sample 덮어 써서 get_dataset()으로 데이터셋화
    dataset_loader.samples = train_samples
    train_dataset = dataset_loader.get_dataset()
    dataset_loader.samples = val_samples
    val_dataset = dataset_loader.get_dataset()

    # Create model
    print("\nCreating model...")
    model = get_model(max_len=MAX_LEN, dropout_step=0, dim=192, num_classes=NUM_CLASSES)

    if tf.config.list_physical_devices('GPU'):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(f"Model compiled successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    print("\nSaving model...")
    model.save(save_model)

    # Convert to TFLite
    print("Converting to TFLite...")
    tflite_model = TFLiteModel(model)  # Pass single model, not list

    concrete_input_signature = tf.TensorSpec(
        shape=[1, 137, 3],  # (배치=1, 키포인트=137, 채널=3(x,y,c))
        dtype=tf.float32
    )
    concrete_function = tflite_model.__call__.get_concrete_function(concrete_input_signature)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_quant_model = converter.convert()
        # Save TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_quant_model)
        print("TFLite model saved successfully!")
    except Exception as e:
        print(f"Warning: TFLite conversion failed: {e}")

    print("Training completed!")

    return history


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs('../models/gloss_transformer_models', exist_ok=True)

    # Train model
    history = train_model()