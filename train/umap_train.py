import os
import pickle

import boto3
import tensorflow as tf
from google.cloud import storage
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, \
    ReduceLROnPlateau
from umap.parametric_umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
from wandb.integration.keras import WandbMetricsLogger

import wandb
from load_data.create_dataset import upload_file, DataSetter, Trainer
from src.config import NUM_NODES, EPOCHS_FOR_UMAP, \
    OUTPUT_DIM, LEARNING_RATE_FOR_UMAP, BATCH_SIZE_FOR_UMAP, \
    WANDB_UMAP_NAME, WANDB_UMAP_PROJECT, WANDB_UMAP_GROUP, WANDB_UMAP_TAGS, \
    LOAD_UMAP, UMAP_DATA_NUM, TEST_UMAP_DATA_NUM, \
    UMAP_OUTPUT_DIM, DROPOUT_RATE_FOR_UMAP, LOCAL_PATHS, L_UMAP, names


class DataDimensionReducer:
    def __init__(self, save_path: str=L_UMAP, storage_save_path: str=LOAD_UMAP, checkpoint_path: str=LOCAL_PATHS["umap_ckpt"],
                 umap_output_dim=UMAP_OUTPUT_DIM, data_num=UMAP_DATA_NUM,
                 epochs=EPOCHS_FOR_UMAP, learning_rate=LEARNING_RATE_FOR_UMAP, dropout_rate=DROPOUT_RATE_FOR_UMAP, batch_size=BATCH_SIZE_FOR_UMAP) -> None:
        # 경로
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path
        self.storage_path = storage_save_path

        print("Creating dataset for umap training...")
        self.train_dataset, self.val_dataset, _ = DataSetter(
            trainer=Trainer.UMAP,
            dim_reduction=False,
            umap_data_num=data_num,
            test_umap_data_num=None
        ).get_datasets()

        self.dims = (OUTPUT_DIM, )  # 49*2
        self.n_components = umap_output_dim
        print(f"{self.dims}에서 {self.n_components}로 차원축소 진행")

        early_stopping_patience = 10
        lr_reduce_patience = 5
        lr_reduce_factor = 0.5
        min_lr = 1e-6

        print("Constructing encoder...")
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.dims),
            tf.keras.layers.Flatten(),

            # tf.keras.layers.Dense(units=512, activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.3),

            # tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.5),
            #
            # tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(learning_rate)),
            tf.keras.layers.Dropout(dropout_rate),

            tf.keras.layers.Dense(units=self.n_components, kernel_regularizer=tf.keras.regularizers.l2(learning_rate))
        ])
        print("Constructing decoder....")
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_components,)),

            # tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(learning_rate)),
            tf.keras.layers.Dropout(dropout_rate),

            # tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.3),
            #
            # tf.keras.layers.Dense(units=256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.5),

            # tf.keras.layers.Dense(units=512, activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(units=OUTPUT_DIM, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(learning_rate)),
            tf.keras.layers.Reshape(target_shape=self.dims)
        ])

        wandb.init(
            project=WANDB_UMAP_PROJECT,
            name=WANDB_UMAP_NAME,
            group=WANDB_UMAP_GROUP,
            tags=WANDB_UMAP_TAGS,
            job_type="umap-train",
            config={
                # Training
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                # UMAP Architecture
                "input_dim": self.dims,
                "output_dim": self.n_components,
                "encoder_hidden_units": 128,
                "decoder_hidden_units": 128,
                "dropout_rate": dropout_rate,
                "l2_reg": learning_rate,
                # UMAP-specific
                "parametric_reconstruction": True,
                "reconstruction_loss_fn": "MSE",
                "autoencoder_loss": True,
                # Data
                "num_nodes": NUM_NODES,
                "data_num": data_num,
                # Callbacks
                "early_stopping_patience": early_stopping_patience,
                "lr_reduce_patience": lr_reduce_patience,
                "lr_reduce_factor": lr_reduce_factor,
                "min_lr": min_lr,
            }
        )

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_reduce_factor,
                patience=lr_reduce_patience,
                min_lr=min_lr,
                verbose=1,
                mode='min'
            ),
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1), # {patience}번 동안 개선 안되면 학습 조기 중단
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                save_weights_only=True, # 가중치만 저장
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            ),
            WandbMetricsLogger()
        ]

        print("Constructing embedder...")
        self.embedder = ParametricUMAP(
            encoder=self.encoder,
            decoder=self.decoder,
            dims=self.dims,
            n_components=self.n_components,
            parametric_reconstruction=True, # umap loss & reconstruction loss 모두 반영

            parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(), # default: Cross Entropy (범위 0~1) 에서 MSE (범위 -1~1)로 변경

            reconstruction_validation=self.val_dataset,
            autoencoder_loss=True,

            batch_size=batch_size,
            verbose=True,
            keras_fit_kwargs={'callbacks': callbacks},
            # device='cuda:0' # specific CUDA gpu
        )
        epochs //= 10    # 모델은 총 에폭수 = n_training_epochs*loss_report_frequency(10) 로 계산하므로
        self.embedder.n_training_epochs = epochs

    def train_reducer(self):
        print("Converting tensorflow dataset to numpy array for UMAP fitting...")
        # 1. 훈련 데이터셋에 맞게
        self.embedder.fit(self.train_dataset)

        # 2. 저장
        print(f"Saving embedder model in {self.save_path}...")
        try:
            self.embedder.encoder.save(LOCAL_PATHS["umap_final"])
        except Exception as e:
            print(f"embedder 정의한 이름으로 저장하는 도중에 에러 발생: {e}")

        # 2-2. Wandb에도 저장
        artifact = wandb.Artifact(
            name=names["umap"],
            type="model",
            description="Trained ParametricUMAP encoder",
            metadata=dict(wandb.config),
        )
        artifact.add_dir(self.save_path)
        wandb.log_artifact(artifact)

        # 임베딩 결과 확인
        print("Transforming data for visualization...")
        embedding = self.embedder.transform(self.train_dataset)
        print(f"Embedding shape: {embedding.shape}")

        # 2-3. embedding.pkl 객체 저장하기
        embedding_file_path = os.path.join(self.save_path, "embedding.pkl")
        print(f"Saving embedding to {embedding_file_path}...")
        with open(embedding_file_path, 'wb') as ef:
            pickle.dump(embedding, ef)

        upload_file(local_root_path=self.save_path, upload_path=self.storage_path, file_name=os.path.basename(self.save_path))
        wandb.finish()

def get_model_args():
    """
    명령어 예시: python -m train.reduction_train --storage L --lr 0.4 --bs 64 --epochs 120 --wd 0.03 -n 10000 --tn 3000
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr", "--learning_rate",
        type=float,
        default=LEARNING_RATE_FOR_UMAP,
    )
    parser.add_argument(
        "--bs", "--batch_size",
        type=int,
        default=BATCH_SIZE_FOR_UMAP
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=EPOCHS_FOR_UMAP
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=UMAP_DATA_NUM
    )
    parser.add_argument(
        "--tn", "--test_num",
        type=int,
        default=TEST_UMAP_DATA_NUM
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
    DataDimensionReducer().train_reducer()