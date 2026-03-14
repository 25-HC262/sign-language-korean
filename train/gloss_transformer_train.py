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
from src.config import CROP_LEN, LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES, \
    NUM_NODES, CHANNELS, VALIDATION_SPLIT, STORAGE_MODE, SELECTED_GM_TYPE, \
    WANDB_GM_PROJECT, WANDB_GM_NAME, WANDB_GM_GROUP, WANDB_GM_TAGS, \
    LOCAL_PATHS, LOAD_GM, UMAP_OUTPUT_DIM, WEIGHT_DECAY
from load_data.create_dataset import DataSetter
from load_data.create_dataset import upload_file

def train_model(learning_rate: float=LEARNING_RATE, epochs: int=EPOCHS, batch_size: int=BATCH_SIZE, weight_decay: float=WEIGHT_DECAY,
                max_sequence_len: int=CROP_LEN
                ):
    # print(" ============== 받은 하이퍼파라미터 ============== ")
    # print(f"    > learning_rate: {learning_rate}")
    # print(f"    > epochs: {epochs}")
    # print(f"    > batch_size: {batch_size}")
    # print(f"    > weight_decay: {weight_decay}")
    # print(f"    > max_sequence_len: {max_sequence_len}")

    wandb.init(
        project=WANDB_GM_PROJECT,
        name=WANDB_GM_NAME,
        group=WANDB_GM_GROUP,
        tags=WANDB_GM_TAGS,
        job_type="train",
        config={
            # Training
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": VALIDATION_SPLIT,
            "optimizer": "adam",
            # Model architecture
            "model_type": SELECTED_GM_TYPE,
            "max_len": max_sequence_len,
            "input_channels": CHANNELS,
            "conv_dim": UMAP_OUTPUT_DIM,
            "kernel_size": 17,
            "num_heads": 4, # TO-DO: backbone.py에서 실제로 가져오도록 수정
            "transformer_expand": 2,
            "conv_blocks_per_stage": 3,
            "transformer_stages": 2,
            "conv_dropout": 0.2,
            "attn_dropout": 0.2,
            "late_dropout": 0.8,
            "num_classes": NUM_CLASSES,
            # Data
            "num_nodes": NUM_NODES,
            "output_dim": UMAP_OUTPUT_DIM,
            # Callbacks
            "early_stopping_patience": 10,
            "lr_reduce_patience": 5,
            "lr_reduce_factor": 0.5,
            "min_lr": 1e-6,
            # Environment
            "storage_mode": STORAGE_MODE,
        }
    )
    print("\nLoading training data...")
    train_dataset, val_dataset, test_dataset = DataSetter(
        max_seq_len=max_sequence_len,
        batch_size=batch_size,
        dim_reduction=True
    ).get_datasets()

    # 1. 모델 생성
    print("\nCreating model...")
    model = get_model(max_len=max_sequence_len, dropout_step=0, dim=UMAP_OUTPUT_DIM, num_classes=NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
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
        epochs=epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                LOCAL_PATHS["gm_ckpt"],
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

    artifact = wandb.Artifact(
        name=f"gloss-{SELECTED_GM_TYPE}-model",
        type="model",
        description=f"Trained gloss {SELECTED_GM_TYPE} model",
        metadata=dict(wandb.config),
    )
    artifact.add_file(LOCAL_PATHS["gm_final"])
    wandb.log_artifact(artifact)

    # 4. 경량화 모델 변환
    print("Converting to TFLite...")
    tflite_model = TFLiteModel(model)  # Pass single model, not list

    concrete_input_signature = tf.TensorSpec(
        shape=[1, max_sequence_len, UMAP_OUTPUT_DIM],  # (배치=1, 최대프레임=max_sequence_len, 채널=umap_dimension)
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

        tflite_artifact = wandb.Artifact(
            name=f"gloss-{SELECTED_GM_TYPE}-tflite",
            type="model",
            description=f"TFLite-converted gloss {SELECTED_GM_TYPE} model",
            metadata=dict(wandb.config),
        )
        tflite_artifact.add_file(LOCAL_PATHS["gm_tflite"])
        wandb.log_artifact(tflite_artifact)
    except Exception as e:
        print(f"Warning: TFLite conversion failed: {e}")

    print("Training completed!")

    # 5. 모델 평가
    eval_results = model.evaluate(test_dataset, return_dict=True)
    print("Model Test Results: ")
    print(*(f"  > {k}: {v:.3f}" for k, v in eval_results.items()), sep='\n')

    wandb.log({f"test_{k}": v for k, v in eval_results.items()})
    wandb.finish()

    return history

def get_model_args():
    """
    명령어 예시: python -m train.gloss_transformer_train --storage L --lr 0.4 --bs 64 --epochs 120 --wd 0.03 --msl 170
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr", "--learning_rate",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--bs", "--batch_size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=EPOCHS
    )
    parser.add_argument(
        "--wd", "--weight_decay",
        type=float,
        default=WEIGHT_DECAY
    )
    parser.add_argument(
        "--msl", "--max_sequence_len",
        type=int,
        default=CROP_LEN
    )

    args, _ = parser.parse_known_args()
    print(*(f"   > {'[default]' if v==parser.get_default(k) else ''} {k}: {v} selected." for k,v in vars(args).items()), sep='\n')
    return args

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
    # 2. 사용자 옵션 받기
    args = get_model_args()

    # 3. 모델 학습
    history = train_model(learning_rate=args.lr, batch_size=args.bs, epochs=args.epochs, weight_decay=args.wd, max_sequence_len=args.msl)
