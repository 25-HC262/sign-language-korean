import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from pathlib import Path
import wandb
from wandb.integration.keras import WandbMetricsLogger
import keras
keras.mixed_precision.set_global_policy("mixed_float16")

# 커스텀
from src.cnn_lstm import get_model
from src.config import CROP_LEN, LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES, \
    NUM_NODES, CHANNELS, VALIDATION_SPLIT, STORAGE_MODE, SELECTED_GM_TYPE, \
    WANDB_GM_PROJECT, WANDB_GM_NAME, WANDB_GM_GROUP, WANDB_GM_TAGS, \
    LOCAL_PATHS, LOAD_GM, UMAP_OUTPUT_DIM, WEIGHT_DECAY, UMAP_LOAD_PATH
from load_data.create_dataset import DataSetter
from load_data.create_dataset import upload_file


def train_model(learning_rate: float = LEARNING_RATE,
                epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE,
                weight_decay: float = WEIGHT_DECAY,
                max_sequence_len: int = CROP_LEN,
                cnn_channels: int = 64,
                lstm_units: int = 128,
                dropout: float = 0.3):

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
            "optimizer": "adamw",
            # Model architecture
            "model_type": SELECTED_GM_TYPE,
            "max_len": max_sequence_len,
            "input_channels": CHANNELS,
            "umap_output_dim": UMAP_OUTPUT_DIM,
            "cnn_channels": cnn_channels,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "num_classes": NUM_CLASSES,
            # Data
            "num_nodes": NUM_NODES,
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
        umap_path=UMAP_LOAD_PATH,
        max_seq_len=max_sequence_len,
        batch_size=batch_size,
        dim_reduction=True
    ).get_datasets()

    # 1. 모델 생성
    print("\nCreating model...")
    model = get_model(
        max_len=max_sequence_len,
        dim=UMAP_OUTPUT_DIM,
        num_classes=NUM_CLASSES,
        cnn_channels=cnn_channels,
        lstm_units=lstm_units,
        dropout=dropout
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print("Model compiled successfully!")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # 2. 모델 학습
    # val_dataset이 비어있는 경우(데이터 부족 시 val_size=0) 대비
    # cardinality: 0=empty, -1=infinite, -2=unknown(데이터 있을 수 있음)
    val_cardinality = val_dataset.cardinality().numpy()
    val_available = val_cardinality != 0  # 0만 진짜 빈 데이터셋
    if not val_available:
        print("Warning: val_dataset이 비어있습니다. validation 없이 학습을 진행합니다.")
    monitor_metric = 'val_loss' if val_available else 'loss'

    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset if val_available else None,
        epochs=epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                LOCAL_PATHS["gm_ckpt"],
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
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
    upload_file(
        local_root_path=str(Path(LOCAL_PATHS["gm_final"]).parent),
        upload_path=LOAD_GM,
        file_name=str(Path(LOCAL_PATHS["gm_final"]).name)
    )

    artifact = wandb.Artifact(
        name=f"gloss-{SELECTED_GM_TYPE}-model",
        type="model",
        description=f"Trained gloss {SELECTED_GM_TYPE} model",
        metadata=dict(wandb.config),
    )
    artifact.add_file(LOCAL_PATHS["gm_final"])
    wandb.log_artifact(artifact)

    # 4. TFLite 변환
    print("Converting to TFLite...")
    concrete_input_signature = tf.TensorSpec(
        shape=[1, max_sequence_len, UMAP_OUTPUT_DIM],
        dtype=tf.float32
    )

    @tf.function(input_signature=[concrete_input_signature])
    def serve(inputs):
        return {'outputs': model(inputs, training=False)}

    concrete_function = serve.get_concrete_function(concrete_input_signature)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS  # LSTM ops에 필요
    ]

    try:
        tflite_model = converter.convert()
        with open(LOCAL_PATHS["gm_tflite"], 'wb') as f:
            f.write(tflite_model)
        print("TFLite model saved successfully!")
        upload_file(
            local_root_path=str(Path(LOCAL_PATHS["gm_tflite"]).parent),
            upload_path=LOAD_GM,
            file_name=str(Path(LOCAL_PATHS["gm_tflite"]).name)
        )

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
    test_cardinality = test_dataset.cardinality().numpy()
    if test_cardinality != 0:
        eval_results = model.evaluate(test_dataset, return_dict=True)
        print("Model Test Results:")
        print(*(f"  > {k}: {v:.3f}" for k, v in eval_results.items()), sep='\n')
        wandb.log({f"test_{k}": v for k, v in eval_results.items()})
    else:
        print("Warning: test_dataset이 비어있습니다. 평가를 건너뜁니다.")
    wandb.finish()

    return history


def get_model_args():
    """
    명령어 예시:
        python -m train.cnn_lstm_train --lr 0.001 --bs 32 --epochs 200 --cnn 64 --lstm 128
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", "--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--bs", "--batch_size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("-e",  "--epochs",          type=int,   default=EPOCHS)
    parser.add_argument("--wd", "--weight_decay",  type=float, default=WEIGHT_DECAY)
    parser.add_argument("--msl", "--max_sequence_len", type=int, default=CROP_LEN)
    parser.add_argument("--cnn", "--cnn_channels", type=int,   default=64)
    parser.add_argument("--lstm", "--lstm_units",  type=int,   default=128)
    parser.add_argument("--dropout",               type=float, default=0.3)

    args, _ = parser.parse_known_args()
    print(*(f"   > {'[default]' if v == parser.get_default(k) else ''} {k}: {v} selected."
            for k, v in vars(args).items()), sep='\n')
    return args


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("WARNING: 지금 GPU가 아니라 CPU를 쓰고 있습니다!")
    else:
        print(f"GPU 사용 중. 현재 사용 가능한 GPU 개수: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f" - GPU [{i}]: {gpu}")

    args = get_model_args()

    history = train_model(
        learning_rate=args.lr,
        batch_size=args.bs,
        epochs=args.epochs,
        weight_decay=args.wd,
        max_sequence_len=args.msl,
        cnn_channels=args.cnn,
        lstm_units=args.lstm,
        dropout=args.dropout,
    )
