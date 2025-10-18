import os
import tensorflow as tf
import boto3
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from umap.parametric_umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.config import NUM_NODES, EPOCHS_FOR_UMAP, UMAP_PATH, S3_DATA_PATH, S3_UMAP_PATH, \
    WANDB_PROJ_NAME, OUTPUT_DIM, LEARNING_RATE_FOR_UMAP, BATCH_SIZE_FOR_UMAP
from load_data.create_dataset import TrainDataLoader, upload_file_to_s3

class DataDimensionReducer:
    def __init__(self, data_path, save_path, s3_save_path, output_dim=32, samples_per_class=1000, epochs=10, learning_rate=0.001, batch_size=1024, mini_project_name="landmark-reducer-v3") -> None:
        self.data_path = data_path
        # s3 경로 확인
        self.is_s3 = data_path.startswith('s3://')
        if self.is_s3:
            parsed_s3_path = urlparse(data_path)
            self.s3_bucket = parsed_s3_path.netloc
            self.s3_prefix = parsed_s3_path.path.lstrip('/')
            self.s3_client = boto3.client('s3')
        self.save_path = os.path.expanduser(save_path)  # f"{base_path}/졸업프로젝트"
        self.s3_save_path = s3_save_path

        self.checkpoint_filepath = os.path.join(self.save_path, 'best_umap_model.weights.h5')

        self.samples_per_class = samples_per_class
        print("Creating dataset...")
        self.trainer = TrainDataLoader(data_path=data_path, save_path=save_path, s3_save_path=s3_save_path, samples_per_class=samples_per_class)
        self.train_dataset, self.test_dataset = self.trainer.create_umap_dataset()

        self.dims = (NUM_NODES*2, )  # 49*2
        self.n_components = output_dim

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

            tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(units=self.n_components, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        ])
        print("Constructing decoder....")
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_components,)),

            # tf.keras.layers.Dense(units=32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),

            #
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

            tf.keras.layers.Dense(units=NUM_NODES*2, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Reshape(target_shape=self.dims)
        ])

        wandb.init(
            project=WANDB_PROJ_NAME,
            name=mini_project_name,
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size # 변수화 하는 게 좋을 듯
            }
        )

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            EarlyStopping(monitor='val_loss', patience=15, verbose=1), # {patience}번 동안 개선 안되면 학습 조기 중단
            ModelCheckpoint(
                filepath=self.checkpoint_filepath,
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

            reconstruction_validation=self.test_dataset,
            autoencoder_loss=True,

            batch_size=1024,
            verbose=True,
            keras_fit_kwargs={'callbacks': callbacks}
        )
        epochs //= 10    # 모델은 총 에폭수 = n_training_epochs*loss_report_frequency(10) 로 계산하므로
        self.embedder.n_training_epochs = epochs

    def train_reducer(self):
        print("Converting tensorflow dataset to numpy array for UMAP fitting...")
        self.embedder.fit(self.train_dataset)
        print("Saving embedder model...")
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        self.embedder.save(self.save_path)

        # 임베딩 결과 확인
        print("Transforming data for visualization...")
        embedding = self.embedder.transform(self.train_dataset)
        print(f"Embedding shape: {embedding.shape}")

        # pickle을 통해서 embedding 객체 저장하기
        embedding_file_path = os.path.join(self.save_path, "embedding.pkl")
        with open(embedding_file_path, 'wb') as ef:
            pickle.dump(embedding, ef)
        print(f"Saved embedding to {embedding_file_path}")

        # 히스토리 출력
        if hasattr(self.embedder, '_history'):
            print("\n=== Training History ===")
            for key, values in self.embedder._history.items():
                print(f"{key}: last value = {values[-1]:.4f}")

            if not self.is_s3:
                try:
                    fig, ax = plt.subplots()
                    ax.plot(self.embedder._history['loss'])
                    ax.set_ylabel('Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_title('Training Loss')
                    plt.savefig(os.path.join(self.save_path, 'training_loss.png'))
                    plt.show()
                except Exception as e:
                    print(f"Error plotting history: {e}")

        upload_file_to_s3(local_root_path=self.save_path, s3_path=self.s3_save_path)
        wandb.finish()


if __name__ == "__main__":
    dr = DataDimensionReducer(
        data_path=S3_DATA_PATH,
        save_path=UMAP_PATH,
        s3_save_path=S3_UMAP_PATH,
        output_dim=OUTPUT_DIM,
        samples_per_class=2000,
        epochs=EPOCHS_FOR_UMAP,
        learning_rate=LEARNING_RATE_FOR_UMAP,
        batch_size=BATCH_SIZE_FOR_UMAP,
        mini_project_name="98->128->32/dropout:0.1/L2:0.001/sizex2/scheduler")
    dr.train_reducer()