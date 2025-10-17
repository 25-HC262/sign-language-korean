import os
import json
import tensorflow as tf
import glob
import boto3
from urllib.parse import urlparse
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple
import pickle
from pathlib import Path
from umap.parametric_umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
from tqdm import tqdm
import wandb
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import defaultdict

class DataDimensionReducer:
    def __init__(self, data_path, save_path, s3_save_path, samples_per_class=1000, epochs=10, mini_project_name="landmark-reducer-v2") -> None:
        self.setting_in_colab()
        self.data_path = data_path
        self.label_map = label_map

        # s3 경로 확인
        self.is_s3 = data_path.startswith('s3://')
        if self.is_s3:
            parsed_s3_path = urlparse(data_path)
            self.s3_bucket = parsed_s3_path.netloc
            self.s3_prefix = parsed_s3_path.path.lstrip('/')
            self.s3_client = boto3.client('s3')
        self.save_path = os.path.expanduser(save_path)  # f"{base_path}/졸업프로젝트"
        self.s3_save_path = s3_save_path
        self.checkpoint_filepath = os.path.join(self.save_path, 'best_model.weights.h5')

        self.samples_per_class = samples_per_class
        print("Creating dataset...")
        self.train_dataset, self.test_dataset = self.create_dataset()

        self.dims = (LEN_LANDMARKS*2, )  # 49*2
        self.n_components = 32

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

            tf.keras.layers.Dense(units=LEN_LANDMARKS * 2, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Reshape(target_shape=self.dims)
        ])

        wandb.init(
            project="grad-umap-project",
            name=mini_project_name,
            config={
                "learning_rate": 0.001,
                "epochs": epochs,
                "batch_size": 1024 # 변수화 하는 게 좋을 듯
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

    def create_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f"\nStarting data loading from: {self.data_path}")
        files_by_class = defaultdict(list)

        for folder_name, folder_meaning in self.label_map.items():
            class_label = folder_meaning
            for direction in DIRECTIONS:
                if self.is_s3:
                    direction_prefix = os.path.join(self.s3_prefix, folder_name, f"{folder_name}_{direction}/")
                    person_prefixes = self._list_s3_subdirs(direction_prefix)
                    print(f"Searching in : {direction_prefix}")
                else:
                    direction_dir = os.path.join(self.data_path, folder_name, f"{folder_name}_{direction}")
                    if not os.path.exists(direction_dir):
                        continue
                    person_dirs = glob.glob(os.path.join(direction_dir, f"*REAL*_{direction}"))
                person_paths = person_prefixes if self.is_s3 else person_dirs
                for person_path in tqdm(person_paths, desc=f"Loading {folder_name}_{direction}"):
                    if self.is_s3:
                        keypoint_files = self._get_s3_keypoint_files(person_path)
                    else:
                        if not os.path.isdir(person_path):
                            continue
                        keypoint_files = sorted(glob.glob(os.path.join(person_path,"*_keypoints.json")))
                    files_by_class[class_label].extend(keypoint_files)

        sampled_files = []
        print("\n--- Stratified Sampling ---")
        for class_label, file_list in files_by_class.items():
            num_files = len(file_list)
            num_to_sample = min(self.samples_per_class, num_files)
            print(f"Class: '{class_label}': Found {num_files}")
            sampled_files.extend(random.sample(file_list, num_to_sample))

        print("---------------------------\n")

        random.shuffle(sampled_files)
        num_samples = len(sampled_files)
        print(f"Total files sampled across all classes: {num_samples}")

        dataset = tf.data.Dataset.from_tensor_slices(sampled_files)
        print(f"Loading and processing data from {self.data_path}...")
        dataset = dataset.map(
            lambda path: tf.py_function(
                self.load_and_process_s3_path, inp=[path], Tout=tf.float32
            ), num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(lambda x: tf.reshape(x, [-1]))

        numpy_dataset = np.array([item.numpy() for item in tqdm(dataset, total=num_samples, desc="Processing Dataset")])

        validation_size = max(1, int(len(numpy_dataset) * VALIDATION_SPLIT)) # 최소 1개

        if validation_size >= len(numpy_dataset):
            raise ValueError(f"Not enough samples: {len(numpy_dataset)}. Need at least 2.")

        train_dataset = numpy_dataset[:-validation_size]
        validation_dataset = numpy_dataset[-validation_size:]

        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(validation_dataset)}")

        return train_dataset, validation_dataset

    def json_to_numpy(self, data: Dict[str, Any]) -> np.ndarray:
        person = data
        # Extract pose keypoints (25 points, 3 values each)
        pose_kp = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
        assert pose_kp.shape[0] == 25, f"pose keypoint shape differs: {pose_kp.shape[0]}"

        # Extract face keypoints (70 points)
        face_kp = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
        assert face_kp.shape[0] == 70, f"face keypoint shape differs: {face_kp.shape[0]}"

        # Extract hand keypoints (21 points each)
        left_hand_kp = np.array(person.get(
            'hand_left_keypoints_2d', [])).reshape(-1, 3)
        assert left_hand_kp.shape[0] == 21, f"left hand keypoint shape differs: {left_hand_kp.shape[0]}"

        right_hand_kp = np.array(person.get(
            'hand_right_keypoints_2d', [])).reshape(-1, 3)
        assert right_hand_kp.shape[0] == 21, f"right hand keypoint shape differs: {right_hand_kp.shape[0]}"

        # Take x, y coordinates and confidence scores
        pose_xy = pose_kp[:, :2]
        pose_conf = pose_kp[:, 2:3]

        face_xy = face_kp[:, :2]
        face_conf = face_kp[:, 2:3]

        left_hand_xy = left_hand_kp[:, :2]
        left_hand_conf = left_hand_kp[:, 2:3]

        right_hand_xy = right_hand_kp[:, :2]
        right_hand_conf = right_hand_kp[:, 2:3]

        # [-1, 1] 정규화
        # First, find the bounding box of all valid points
        all_x = np.concatenate([
            pose_xy[pose_conf[:, 0] > 0.1, 0],
            face_xy[face_conf[:, 0] > 0.1, 0],
            left_hand_xy[left_hand_conf[:, 0] > 0.1, 0],
            right_hand_xy[right_hand_conf[:, 0] > 0.1, 0]
        ])
        all_y = np.concatenate([
            pose_xy[pose_conf[:, 0] > 0.1, 1],
            face_xy[face_conf[:, 0] > 0.1, 1],
            left_hand_xy[left_hand_conf[:, 0] > 0.1, 1],
            right_hand_xy[right_hand_conf[:, 0] > 0.1, 1]
        ])

        if len(all_x) > 0 and len(all_y) > 0:
            # Calculate center and scale
            center_x = (np.max(all_x) + np.min(all_x)) / 2
            center_y = (np.max(all_y) + np.min(all_y)) / 2
            scale = max(
                np.max(all_x) - np.min(all_x),
                np.max(all_y) - np.min(all_y)) / 2

            # Avoid division by zero
            if scale < 1:
                scale = 1

            # [-1, 1] 정규화
            pose_xy = (pose_xy - [center_x, center_y]) / scale
            face_xy = (face_xy - [center_x, center_y]) / scale
            left_hand_xy = (left_hand_xy - [center_x, center_y]) / scale
            right_hand_xy = (right_hand_xy - [center_x, center_y]) / scale

            # Clip
            pose_xy = np.clip(pose_xy, -2, 2)
            face_xy = np.clip(face_xy, -2, 2)
            left_hand_xy = np.clip(left_hand_xy, -2, 2)
            right_hand_xy = np.clip(right_hand_xy, -2, 2)

        # Concatenate all keypoints: body(25) + face(70) + left_hand(21) + right_hand(21) = 137
        all_keypoints = np.concatenate([
            pose_xy,
            face_xy,
            left_hand_xy,
            right_hand_xy
        ], axis=0)

        # Extract selected landmarks
        selected_keypoints = all_keypoints[POINT_LANDMARKS, :]

        # 목 기준 정규화
        neck_pos = all_keypoints[NECK:NECK + 1, :]
        neck_mean = np.nanmean(neck_pos, axis=0, keepdims=True)
        if np.isnan(neck_mean).any():
            neck_mean = np.array([[0.5, 0.5]])

        std = np.nanstd(selected_keypoints)
        if std == 0:
            std = 1.0
        selected_keypoints = (selected_keypoints - neck_mean) / std
        selected_keypoints = np.nan_to_num(selected_keypoints, 0)

        return selected_keypoints.astype(np.float32) # (98, 2)

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

        self._upload_model_to_s3()
        wandb.finish()

    def load_encoder(self):
        return load_ParametricUMAP(self.save_path)  # embedder.encoder로 사용할 것

    # s3 관련 함수
    def load_and_process_s3_path(self, file_path_tensor: tf.Tensor) -> tf.Tensor:
        file_path = file_path_tensor.numpy().decode('utf-8')
        try:
            if file_path.startswith('s3://'):
                parsed_s3_path = urlparse(file_path)
                s3_object = self.s3_client.get_object(
                    Bucket=parsed_s3_path.netloc,
                    Key=parsed_s3_path.path.lstrip('/')
                )
                file_content = s3_object['Body'].read().decode('utf-8')
                data = json.loads(file_content)
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            assert 'people' in data, "데이터에 키포인트 없음"
            people_data = data['people']

            assert isinstance(people_data, dict), "이상한 데이터 형식"
            person = people_data
            assert len(people_data) != 0, "데이터 길이 0"

            return self.json_to_numpy(person)
        except Exception as e:
            raise ValueError(f"Error processing {file_path_tensor.numpy().decode('utf-8')}: {e}")

    def _list_s3_subdirs(self, prefix):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(
            Bucket=self.s3_bucket,
            Prefix=prefix,
            Delimiter='/')
        return [p.get('Prefix')
                for page in result for p in page.get('CommonPrefixes', [])]

    def _get_s3_keypoint_files(self, prefix):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)
        return sorted([f"s3://{self.s3_bucket}/{o.get('Key')}" for page in result for o in page.get(
            'Contents', []) if o.get('Key').endswith('_keypoints.json')])

    def _upload_model_to_s3(self):
        local_dir = os.path.expanduser(self.save_path)
        print(f"Uploading files of {local_dir} to {self.s3_save_path}")
        parsed_s3_path = urlparse(self.s3_save_path)
        save_bucket = parsed_s3_path.netloc
        save_prefix = parsed_s3_path.path.lstrip('/')

        # 기존 s3_client 재사용 (이미 __init__에서 생성됨)
        if not hasattr(self, 's3_client'):
            self.s3_client = boto3.client('s3')

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.join(save_prefix, os.path.relpath(local_path, local_dir))

                try:
                    self.s3_client.upload_file(local_path, save_bucket, s3_path)
                    print(f"Uploaded {local_path} to s3://{save_bucket}/{s3_path}")

                except Exception as e:
                    print(f"Error uploading {local_path} to s3://{save_bucket}/{s3_path}: {e}")

    @staticmethod
    def setting_in_colab():
        global label_map, MAX_LEN, BATCH_SIZE, POINT_LANDMARKS, DIRECTIONS, NECK, LEN_LANDMARKS, VALIDATION_SPLIT
        label_map = {
            "NIA_SL_SEN0181": "도움받다",
            "NIA_SL_SEN0354": "안녕하세요",
            "NIA_SL_SEN0355": "감사합니다",
            "NIA_SL_SEN0356": "미안합니다",
            "NIA_SL_SEN2000": "수고"
        }
        MAX_LEN = 125
        BATCH_SIZE = 16
        POSE = [
            # 0,   # Nose
            1,   # Neck (normalization reference)
            2,   # RShoulder
            3,   # RElbow
            4,   # RWrist
            5,   # LShoulder
            6,   # LElbow
            7,   # LWrist
            # 15,  # REye
            # 16,  # LEye
            # 17,  # REar
            # 18   # LEar
        ]
        NECK = 1
        # 왼손 (95-115): 21개 포인트
        LHAND = list(range(95, 116))

        # 오른손 (116-136): 21개 포인트
        RHAND = list(range(116, 137))

        POINT_LANDMARKS = (
                POSE +
                LHAND +
                RHAND
        )
        DIRECTIONS = ['D', 'F', 'L', 'R', 'U']

        LEN_LANDMARKS = len(POINT_LANDMARKS)

        VALIDATION_SPLIT = 0.2


if __name__ == "__main__":
    S3_DATA_PATH = 's3://openpose-keypoints'
    LOCAL_SAVE_PATH = '~/store_umap'
    S3_SAVE_PATH = 's3://trout-model/umap_models'
    dr = DataDimensionReducer(
        data_path=S3_DATA_PATH,
        save_path=LOCAL_SAVE_PATH,
        s3_save_path=S3_SAVE_PATH,
        samples_per_class=2000,
        epochs=100,
        mini_project_name="98->128->32/dropout:0.1/L2:0.001/sizex2/scheduler")
    dr.train_reducer()