import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from pathlib import Path
import wandb
from wandb.integration.keras import WandbMetricsLogger
import keras
keras.mixed_precision.set_global_policy("mixed_float16") # fp16 가속 keras3 버전

# 커스텀
from src.backbone import get_model, TFLiteModel
from src.config import MAX_LEN, LEARNING_RATE, EPOCHS, BATCH_SIZE, OUTPUT_DIM, NUM_CLASSES, \
    WANDB_GM_PROJECT, WANDB_GM_NAME, L_CKPT, LOCAL_PATHS, LOAD_GM, LOAD_DATA, WEIGHT_DECAY
from load_data.create_dataset import TrainDataLoader
from load_data.create_dataset import upload_file

def train_model(data_path: str):
    wandb.init(
        project=WANDB_GM_PROJECT,
        name=WANDB_GM_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }
    )
    print("\nLoading training data...")
    train_dataset, val_dataset, test_dataset = TrainDataLoader(data_path=data_path, is_training_transformer=True).create_transformer_dataset()

    # 1. 모델 생성
    print("\nCreating model...")
    model = get_model(max_len=MAX_LEN, dropout_step=0, dim=OUTPUT_DIM, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    print(f"Model compiled successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # 2. 모델 학습
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                L_CKPT,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            WandbMetricsLogger()
        ]
    )

    # 3. 모델 저장
    print("\nSaving model...")
    model.save(LOCAL_PATHS["gm_final"])
    upload_file(local_root_path=str(Path(LOCAL_PATHS["gm_final"]).parent), upload_path=LOAD_GM, file_name=str(Path(Path(LOCAL_PATHS["gm_final"]).name)))

    # 4. 경량화 모델 변환
    print("Converting to TFLite...")
    tflite_model = TFLiteModel(model)  # Pass single model, not list

    concrete_input_signature = tf.TensorSpec(
        shape=[1, MAX_LEN, OUTPUT_DIM],  # (배치=1, 최대프레임=137, 채널=유맵차원)
        dtype=tf.float32
    )
    concrete_function = tflite_model.__call__.get_concrete_function(concrete_input_signature)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_quant_model = converter.convert()
        # 7. 경량화 모델 저장
        with open(LOCAL_PATHS["gm_tflite"], 'wb') as f:
            f.write(tflite_quant_model)
        print("TFLite model saved successfully!")
        upload_file(local_root_path=str(Path(LOCAL_PATHS["gm_tflite"]).parent), upload_path=LOAD_GM, file_name=str(Path(LOCAL_PATHS["gm_tflite"]).name))
    except Exception as e:
        print(f"Warning: TFLite conversion failed: {e}")

    print("Training completed!")

    # 5. 모델 평가
    eval_results = model.evaluate(test_dataset, return_dict=True)
    print("Model Test Results: ")
    print(*(f"  > {k}: {v:.3f}", for k, v in eval_results.items()), sep='\n')

    wandb.log({f"test_{k}": v for k, v in eval_results.items()})
    wandb.finish()

    return history

if __name__ == "__main__":
    # 1. 사용 가능한 GPU 리스트 출력
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("WARNING: 지금 GPU가 아니라 CPU를 쓰고 있습니다!")
    else:
        # 현재 인식된 GPU 개수와 상세 명칭 출력
        print(f"GPU 사용 중. 현재 사용 가능한 GPU 개수: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f" - GPU [{i}]: {gpu}")

    # Train model
    history = train_model(data_path=LOAD_DATA) # parameter로 설정값 받도록 수정