import os
import tensorflow as tf
import datetime
from pathlib import Path
import wandb
from wandb.integration.keras import WandbMetricsLogger

# 커스텀
from src.backbone import get_model, TFLiteModel
from src.config import MAX_LEN, LEARNING_RATE, EPOCHS, BATCH_SIZE, OUTPUT_DIM, NUM_CLASSES, WANDB_PROJ_NAME, S3_DATA_PATH, S3_GLOSS_TRANSFORMER_PATH
from load_data.create_dataset import TrainDataLoader, upload_file_to_s3

date_idx = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
base_name = f"sign_language_v2_{date_idx}"
checkpoint_path = f"checkpoints/{base_name}.h5"
save_model = f"models/{base_name}.h5"
tflite_path = f"models/gloss_transformer_models/{base_name}.tflite"

def train_model(data_path: str, mini_project_name: str):
    wandb.init(
        project=WANDB_PROJ_NAME,
        name=mini_project_name,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }
    )
    print("\nLoading training data...")
    train_dataset, val_dataset = TrainDataLoader(data_path=data_path, is_training_transformer=True).create_transformer_dataset()

    # Create model
    print("\nCreating model...")
    model = get_model(max_len=MAX_LEN, dropout_step=0, dim=OUTPUT_DIM, num_classes=NUM_CLASSES)

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
    model.save(save_model)
    upload_file_to_s3(local_root_path=str(Path(save_model).parent), s3_path=S3_GLOSS_TRANSFORMER_PATH, file_name=str(Path(save_model).name))

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
        with open(tflite_path, 'wb') as f:
            f.write(tflite_quant_model)
        print("TFLite model saved successfully!")
        upload_file_to_s3(local_root_path=str(Path(tflite_path).parent), s3_path=S3_GLOSS_TRANSFORMER_PATH, file_name=str(Path(tflite_model).name))
    except Exception as e:
        print(f"Warning: TFLite conversion failed: {e}")

    print("Training completed!")
    wandb.finish()

    return history

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs('../models/gloss_transformer_models', exist_ok=True)

    # Train model
    history = train_model(data_path=S3_DATA_PATH, mini_project_name="output-dim:32")