import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from pathlib import Path
import wandb
from wandb.integration.keras import WandbMetricsLogger
import keras

# 커스텀
from src.backbone import get_model, TFLiteModel
from src.config import MAX_LEN, LEARNING_RATE, EPOCHS, BATCH_SIZE, OUTPUT_DIM, NUM_CLASSES, \
    NUM_NODES, CHANNELS, VALIDATION_SPLIT, STORAGE_MODE, SELECTED_GM_TYPE, \
    WANDB_GM_PROJECT, WANDB_GM_NAME, WANDB_GM_GROUP, WANDB_GM_TAGS, \
    L_CKPT, LOCAL_PATHS, LOAD_GM, LOAD_DATA
from load_data.create_dataset import TrainDataLoader
from load_data.create_dataset import upload_file

def train_model(data_path: str):
    wandb.init(
        project=WANDB_GM_PROJECT,
        name=WANDB_GM_NAME,
        group=WANDB_GM_GROUP,
        tags=WANDB_GM_TAGS,
        job_type="train",
        config={
            # Training
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "validation_split": VALIDATION_SPLIT,
            "optimizer": "adam",
            # Model architecture
            "model_type": SELECTED_GM_TYPE,
            "max_len": MAX_LEN,
            "input_channels": CHANNELS,
            "conv_dim": OUTPUT_DIM,
            "kernel_size": 17,
            "num_heads": 4,
            "transformer_expand": 2,
            "conv_blocks_per_stage": 3,
            "transformer_stages": 2,
            "conv_dropout": 0.2,
            "attn_dropout": 0.2,
            "late_dropout": 0.8,
            "num_classes": NUM_CLASSES,
            # Data
            "num_nodes": NUM_NODES,
            "output_dim": OUTPUT_DIM,
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
    train_dataset, val_dataset = TrainDataLoader(data_path=data_path, is_training_transformer=True).create_transformer_dataset()

    # Create model
    print("\nCreating model...")
    model = get_model(max_len=MAX_LEN, dropout_step=0, dim=OUTPUT_DIM, num_classes=NUM_CLASSES)

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    wandb.watch(model, log="all", log_freq=10)

    print(f"Model compiled successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Callbacks
    callbacks = [
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

    # Convert to TFLite
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
        # Save TFLite model
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
    wandb.finish()

    return history

if __name__ == "__main__":
    # Train model
    history = train_model(data_path=LOAD_DATA)