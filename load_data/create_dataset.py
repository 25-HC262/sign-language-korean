import importlib
importlib.import_module("src.config")
import json
import os
import random
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple
from typing import Union, Type
from urllib.parse import urlparse

import boto3
import keras
import numpy as np
import tensorflow as tf
from google.cloud import storage
from tqdm import tqdm

from src.config import KSL_SENTENCES, POINT_LANDMARKS, DIRECTIONS, VALIDATION_SPLIT, CROP_LEN, \
    BATCH_SIZE, UMAP_LOAD_PATH, TEST_RATE, UMAP_OUTPUT_DIM, UPLOAD_MODE, NUM_CLASSES, \
    L_PREPROCESSED_DATA, MAX_LEN, L_DATA, LOAD_TEST, LOAD_DATA, UMAP_DATA_NUM, TEST_UMAP_DATA_NUM, \
    KEYPOINT_DIM, DATA_TYPE, DataDim, DataType, Trainer

class DataSetter:
    def __init__(self, umap_path: str=None, # num_classes: int=NUM_CLASSES, # 전역변수로 이미 관리 중.
                 trainer: Trainer=Trainer.GM,
                 max_seq_len: int=MAX_LEN,
                 batch_size: int=BATCH_SIZE,
                 dim_reduction: bool=False,
                 umap_data_num=UMAP_DATA_NUM,
                 test_umap_data_num=TEST_UMAP_DATA_NUM):
        """
        :param umap_path: Umap Version. 사용자 옵션`--umap`에서 입력한 Umap 차원축소기 파일명에 따라 경로가 자동 주입됨. None인 경우 umap 훈련 혹은 umap을 사용하지 않는다.
        :param trainer:
        :param max_seq_len:
        """
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        # 입력이 문자열이면 Enum으로 변환, 아니면 그대로 유지
        def to_enum(enum_class: Type[Enum], value: Union[str, Enum]) -> Enum:
            if isinstance(value, (str, int)):
                try:
                    lookup_value = value.lower() if isinstance(value, str) else value
                    return enum_class(lookup_value) # enum_class.value 리턴
                except ValueError:
                    print(f"[!] Warning: {value}는 {enum_class.__name__}에 없습니다.")
                    return list(enum_class)[0] # 첫번째 항목을 기본값으로 사용
            return value

        # x: (original_len, dim) -> (target_len, dim)
        def truncate_sequence(max_len):
            # x shape: (original_max_len, UMAP_OUTPUT_DIM)
            return lambda x, y: (x[:max_len, :], y) # 직렬화 문제로 인해 return x[:self.max_seq_len, :], y 대신

        def seq_and_batch_finalize(path_or_ds: Union[Path, tf.data.Dataset]) -> tf.data.Dataset:
            # 입력이 경로면 로드하고, 데이터셋 객체면 그대로 사용
            if isinstance(path_or_ds, (Path, str)):
                ds = tf.data.Dataset.load(str(path_or_ds))
            else:
                ds = path_or_ds

            # 최종 배치 처리
            def batch_finalize(ds: tf.data.Dataset, target_batch_size: int=BATCH_SIZE) -> tf.data.Dataset:
                return ds.cache().batch(target_batch_size).prefetch(tf.data.AUTOTUNE) # cache 설정으로 메모리에 올려 속도 극대화

            # 기본 길이(MAX_LEN)가 아닌 경우: 시퀀스 자르기
            if self.max_seq_len < MAX_LEN:
                ds = ds.map(truncate_sequence(self.max_seq_len), num_parallel_calls=tf.data.AUTOTUNE)
            return batch_finalize(ds=ds, target_batch_size=self.batch_size)

        data_type = to_enum(DataType, DATA_TYPE)
        data_dim = to_enum(DataDim, KEYPOINT_DIM)
        self.trainer = to_enum(Trainer, trainer)

        # 경로 설정
        umap_name = Path(umap_path).stem if umap_path else "None" # None 설정이 default가 되도록 바꿀 것. & TO-DO. umap 선택 가능하도록
        # gm인 경우 모델 구분 없이 사용하도록. umap dataset인 경우 맨 앞에 umap 붙일 것.
        data_dir = f"un={umap_name}-nc={NUM_CLASSES}-dt={data_type.value}-dd={data_dim.value}" if self.trainer is Trainer.GM else f"{self.trainer.value}-nc={NUM_CLASSES}-dt={data_type.value}-dd={data_dim.value}"
        dir_path = Path(L_PREPROCESSED_DATA) / data_dir

        print(f"[*] 초기화 완료: {data_type.name} / {data_dim.value}") # TO-DO: dim에 맞게 create_dataset에서 가져오도록.

        train_path = dir_path / "train_ds"
        val_path = dir_path / "val_ds"
        test_path = dir_path / "test_ds"

        loader_args = {
            "data_path": LOAD_DATA,
            "trainer": trainer,
            "dim_reduction": dim_reduction,
            "umap_path": umap_path
        }
        if trainer is Trainer.UMAP:
            loader_args.update({
                "umap_data_num": umap_data_num,
                "test_umap_data_num": test_umap_data_num
            })
        self.loader = TrainDataLoader(**loader_args) # 딕셔너리 언패킹

        def get_or_create_datasets_from_path() -> Union[Tuple[tf.data.Dataset,tf.data.Dataset,tf.data.Dataset], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            if (train_path / train_path.with_suffix('.npy')).exists() or (train_path / "dataset_spec.pb").exists():
                try:
                    print(f"[*] 기존 데이터셋을 {data_dir}에서 로드 중;")
                    if self.trainer is Trainer.GM:
                        return (
                            seq_and_batch_finalize(train_path),
                            seq_and_batch_finalize(val_path),
                            seq_and_batch_finalize(test_path)
                        )
                    else:
                        train_file = train_path.with_suffix('.npy')
                        val_file = val_path.with_suffix('.npy')
                        test_file = test_path.with_suffix('.npy')

                        return (
                            np.load(train_file, allow_pickle=True),
                            np.load(val_file, allow_pickle=True),
                            np.load(test_file, allow_pickle=True),
                        )
                except Exception as e:
                    print(f"[!] 로드 실패: {e}. 불완전한 데이터셋을 삭제합니다.")
                    import shutil
                    shutil.rmtree(dir_path, ignore_errors=True)
            try:
                print("[!] 데이터셋 생성을 시작합니다.")
                os.makedirs(dir_path, exist_ok=True) # 하위 폴더가 생성되지 않은 경우일 수 있음.

                if self.trainer is Trainer.GM:
                    # [생성] 원본 MAX_LEN과 지정된 batch_size로 생성됨
                    train_ds, val_ds = self.loader.create_gm_train_dataset()
                    test_ds = self.loader.create_test_dataset()

                    # [저장] MAX_LEN, unbatch 상태인 raw dataset 그대로 저장하여 범용성 확보
                    tf.data.Dataset.save(train_ds, str(train_path))
                    tf.data.Dataset.save(val_ds, str(val_path))
                    tf.data.Dataset.save(test_ds, str(test_path))

                    # [현재 사용 가공] 생성된 ds는 배치된 상태이므로 unbatch() 후 자르기
                    return (
                        seq_and_batch_finalize(train_ds),
                        seq_and_batch_finalize(val_ds),
                        seq_and_batch_finalize(test_ds)
                    )
                else:
                    # [생성] 키포인트 단위이므로 MAX_LEN, batch_size 영향을 받지 않는다
                    train_ds, val_ds = self.loader.create_umap_train_dataset()
                    test_ds = self.loader.create_umap_test_dataset()

                    # [저장] np 파일로 저장
                    train_file = train_path.with_suffix('.npy')
                    val_file = val_path.with_suffix('.npy')
                    test_file = test_path.with_suffix('.npy')

                    np.save(train_file, train_ds)
                    np.save(val_file, val_ds)
                    np.save(test_file, test_ds)

                    # [리턴]
                    return train_ds, val_ds, test_ds

            except Exception as e:
                print(f"[!] 생성 도중 에러 발생: {e}. 생성 중인 폴더를 삭제합니다.")
                import shutil
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                raise e # 최상위 에러는 알려줌

        self.final_datasets = get_or_create_datasets_from_path() # train, val, test

    def get_datasets(self) -> Union[Tuple[tf.data.Dataset,tf.data.Dataset,tf.data.Dataset], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.final_datasets

class TrainDataLoader:
    def __init__(self, trainer: Trainer=Trainer.GM, data_path=L_DATA, test_data_path=LOAD_TEST, umap_path=None, umap_data_num=None, test_umap_data_num=None, dim_reduction=False):
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.umap_path = umap_path
        self.umap_data_num = umap_data_num
        self.test_umap_data_num = test_umap_data_num
        self.trainer = trainer
        self.dim_reduction = dim_reduction
        self.label_map = {name: i for i, name in enumerate(KSL_SENTENCES.keys())}
        if self.dim_reduction:
            print(f"Loading UMAP encoder model from: {self.umap_path}")
            try:
                self.umap_encoder = keras.saving.load_model(self.umap_path) # keras 3 upgrade

            except Exception as e:
                print(f"Error loading encoder model: {e}")
                # 커스텀 객체 사용
                self.umap_encoder = keras.saving.load_model(
                    self.umap_path,
                    custom_objects={'InputLayer': keras.layers.InputLayer},
                    compile=False
                )
            print("UMAP encoder model loaded successfully.")

        self.videos = []
        self.umap_keypoints_list = []

    def _base_generator(self, path=None) -> tf.data.Dataset:
        self.max_len = MAX_LEN
        # self.videos 구성
        self._get_all_filepaths(path=path)

        videos_snapshot = list(self.videos)
        random.seed(42)
        random.shuffle(videos_snapshot)

        def _data_generator():
            for video in videos_snapshot:
                seq = video['sequence']
                if len(seq) > self.max_len:
                    seq = seq[:self.max_len]
                else:
                    padding = np.zeros((self.max_len - len(seq), UMAP_OUTPUT_DIM))
                    seq = np.concatenate([seq, padding], axis=0)
                yield seq.astype(np.float32), np.int32(video['class_label'])
        full_dataset = tf.data.Dataset.from_generator(
            _data_generator, # 함수 자체 전달
            output_signature=(
                tf.TensorSpec(shape=(self.max_len, UMAP_OUTPUT_DIM), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        return full_dataset

    def create_test_dataset(self) -> tf.data.Dataset:
        test_dataset = self._base_generator(path=self.test_data_path)
        test_size = int(len(self.videos) * TEST_RATE)
        print(f"테스트 데이터셋 크기 (예상): {test_size}")

        return test_dataset

    def create_umap_test_dataset(self) -> np.ndarray:
        if self.test_umap_data_num is None: return np.array([], dtype=np.float32)
        self._get_all_filepaths(path=self.test_data_path, umap_data_num=self.test_umap_data_num)
        test_ds = np.array(self.umap_keypoints_list, dtype=np.float32)

        print(f"유맵 테스트 데이터셋 크기 (예상): {len(test_ds)}")
        # 데이터 비우기
        self.umap_keypoints_list = None
        return test_ds

    def create_umap_train_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._get_all_filepaths(umap_data_num=self.umap_data_num)
        print(f"Loading and processing umap data from {self.data_path}...")

        umap_list_len = len(self.umap_keypoints_list)
        val_size = max(1, int(umap_list_len*VALIDATION_SPLIT))
        if val_size >= umap_list_len:
            raise ValueError(f"Not enough samples: {len(self.umap_keypoints_list)}. Need at least 2.")

        train_ds = np.array(self.umap_keypoints_list[:-val_size], dtype=np.float32)
        val_ds = np.array(self.umap_keypoints_list[-val_size:], dtype=np.float32)

        print(f"유맵 훈련 데이터셋 크기 (예상): {len(train_ds)}\n 유맵 검증 데이터셋 크기 (예상): {len(val_ds)}")
        # 데이터 비우기
        self.umap_keypoints_list = None
        return train_ds, val_ds

    def create_gm_train_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        full_dataset = self._base_generator() # path는 기본 self.data_path
        dataset_size = len(self.videos)
        print(f"훈련에 사용되는 전체 데이터셋 크기: {dataset_size}")
        val_size = int(dataset_size * VALIDATION_SPLIT) # 남은 데이터 1:1-VAL로 분리
        train_size = dataset_size-val_size
        print(f"훈련 데이터셋 크기 (예상): {train_size}")
        print(f"검증 데이터셋 크기 (예상): {val_size}")

        val_dataset = full_dataset.take(val_size)
        train_dataset = full_dataset.skip(val_size)

        print("\nTrain Dataset Spec:\n", train_dataset)
        print("\nValidation Dataset Spec:\n", val_dataset)
        # 데이터 비우기
        self.videos = None

        return train_dataset, val_dataset

    # 훈련 data 혹은 test data를 가져옴.
    # dim_reduction에 따라 데이터 차원축소를 하거나 하지 않음.
    def _get_all_filepaths(self, umap_data_num: int=None, path=None):
        path = self.data_path if path is None else path
        print(f"\nStarting data loading from: {path}")

        is_s3 = path.startswith('s3://')
        is_gcs = path.startswith('gs://')

        if is_s3:
            parsed_s3_path = urlparse(path)
            self.s3_bucket = parsed_s3_path.netloc
            self.s3_prefix = parsed_s3_path.path.lstrip('/')
            self.s3_client = boto3.client('s3')
        if is_gcs:
            parsed_path = urlparse(path)
            self.gcs_bucket_name = parsed_path.netloc
            self.gcs_prefix = parsed_path.path.lstrip('/')
            self.gcs_client = storage.Client()
            self.gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)

        self.videos = []
        keypoints_list = []

        umap_data_num = umap_data_num if umap_data_num is not None else self.umap_data_num
        with tqdm(KSL_SENTENCES.items(), desc="[Total Progress]", position=0) as pbar_sentences:
            for folder_name, folder_meaning in pbar_sentences:
                pbar_sentences.set_postfix(current_class=folder_name)
                for direction in DIRECTIONS:
                    if is_s3:
                        direction_dir = f"{self.s3_prefix}/{folder_name}/{folder_name}_{direction}"
                        person_paths = self._list_s3_subdirs(direction_dir)
                    elif is_gcs:
                        direction_dir = f"{self.gcs_prefix}/{folder_name}/{folder_name}_{direction}/"
                        person_paths = self._list_gcs_subdirs(direction_dir)
                    else:
                        base_path = Path(path)
                        direction_dir = base_path / folder_name / f"{folder_name}_{direction}"
                        if not direction_dir.exists(): continue
                        person_paths = list(direction_dir.glob(f"*REAL*_{direction}"))
                    tqdm.write(f"Searching in : {direction_dir}")

                    for person_path in tqdm(person_paths, desc=f"Loading {folder_name}_{direction}",
                                            position=1, leave=False):
                        person_path_str = str(person_path)
                        if is_s3:
                            keypoint_files = self._get_s3_keypoint_files(person_path_str)
                        elif is_gcs:
                            keypoint_files = self._get_gcs_keypoint_files(person_path_str)
                        else:
                            person_p = Path(person_path)
                            if not person_p.is_dir(): continue
                            keypoint_files = sorted([str(f) for f in person_p.glob("*_keypoints.json")])

                        keypoints_batch = []
                        for kp_file in tqdm(keypoint_files, desc="      Processing Frames", position=2, leave=False,
                                            disable=len(keypoint_files)<10):
                            try:
                                file_path_tensor = tf.constant(kp_file)
                                keypoints = self._load_json_from_path(file_path_tensor)
                                keypoints = keypoints.reshape(-1) # (98, )

                                # UMAP인 경우 모든 프레임 개별 저장
                                if self.trainer is Trainer.UMAP:
                                    keypoints_list.append(keypoints)
                                # GM인 경우 배치에 저장
                                keypoints_batch.append(keypoints)

                            except Exception as e:
                                tqdm.write(f"file load error: {kp_file} - {e}")
                                continue

                        if len(keypoints_batch) == 0: continue

                        if self.trainer is Trainer.GM:
                            keypoints_batch = np.array(keypoints_batch)                 # (T, 98)
                            if self.dim_reduction:
                                keypoints_batch = self.umap_encoder.predict(keypoints_batch) # (T, 32)
                            self.videos.append({
                                'sequence': keypoints_batch,
                                'class_label': self.label_map[folder_name]
                            })
        if self.trainer is Trainer.UMAP:
            print(f"[*] UMAP용 키포인트 변환 중... {len(keypoints_list)} 프레임 중 {umap_data_num}개 뽑기")
            random_keypoints_list = random.sample(keypoints_list, umap_data_num)
            self.umap_keypoints_list = np.array(random_keypoints_list, dtype=np.float32)
            del random_keypoints_list, keypoints_list

    # 과거 umap 데이터 구성에 사용
    # 개별 키포인트 파일 경로의 묶음을 리턴함
    def _stratified_sample_files(self, files_by_class: dict, samples_per_class: int=1000) -> list:
        sampled_files = []
        print("\n--- Stratified Sampling ---")
        for class_label, file_list in files_by_class.items():
            num_files = len(file_list)
            num_to_sample = min(samples_per_class, num_files)
            print(f"Class: '{class_label}': Found {num_files}")
            sampled_files.extend(random.sample(file_list, num_to_sample)) # keypoint 파일들 중 일부 선택
        print("---------------------------\n")

        random.shuffle(sampled_files)
        print(f"Total files sampled across all classes: {len(sampled_files)}")
        return sampled_files

    def _json_to_numpy(self, data: Dict[str, Any]) -> np.ndarray:
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
            if scale < 1e-6: # 아주 작은 스케일 값에 대한 예외 처리
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

        # 선택된 키포인트 추출
        selected_keypoints = all_keypoints[POINT_LANDMARKS, :]
        selected_keypoints = np.nan_to_num(selected_keypoints, 0)

        return selected_keypoints.astype(np.float32) # (49, 2)

    def _load_json_from_path(self, file_path_tensor: tf.Tensor) -> np.ndarray:
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
            elif file_path.startswith('gs://'):
                parsed_gcs_path = urlparse(file_path)

                gcs_bucket_name = parsed_gcs_path.netloc
                blob_name = parsed_gcs_path.path.lstrip('/')

                gcs_bucket = self.gcs_client.bucket(gcs_bucket_name)

                blob = gcs_bucket.blob(blob_name)
                file_content = blob.download_as_text()
                data = json.loads(file_content) # 'utf-8' 설정 디폴트
            else:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            assert 'people' in data, "데이터에 키포인트 없음"
            people_data = data['people']

            assert isinstance(people_data, dict), "이상한 데이터 형식"
            person = people_data
            assert len(people_data) != 0, "데이터 길이 0"

            return self._json_to_numpy(person)
        except Exception as e:
            raise ValueError(f"Error processing {file_path_tensor.numpy().decode('utf-8')}: {e}")

    #  --- S3 exclusive helper method ---
    def _list_s3_subdirs(self, prefix):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(
            Bucket=self.s3_bucket,
            Prefix=prefix,
            Delimiter='/')
        return [p.get('Prefix') for page in result for p in page.get('CommonPrefixes', [])]
    def _get_s3_keypoint_files(self, prefix):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)
        return sorted([f"s3://{self.s3_bucket}/{o.get('Key')}" for page in result for o in page.get(
            'Contents', []) if o.get('Key').endswith('_keypoints.json')])

    #  --- GCP exclusive helper method ---
    def _list_gcs_subdirs(self, prefix):
        blobs = self.gcs_client.list_blobs(self.gcs_bucket_name, prefix=prefix, delimiter='/')
        list(blobs) # 호출을 통해 prefixes를 채움.
        return list(blobs.prefixes)
    def _get_gcs_keypoint_files(self, prefix):
        blobs = self.gcs_client.list_blobs(self.gcs_bucket_name, prefix=prefix, delimiter='/')
        return sorted([f"gs://{self.gcs_bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith('_keypoints.json')])

# --- 업로드 인터페이스 함수 ---
def upload_file(local_root_path: str, upload_path: str, file_name: str = None):
    if UPLOAD_MODE == 'S':
        upload_file_to_s3(local_root_path=local_root_path, s3_path=upload_path, file_name=file_name)
    elif UPLOAD_MODE == 'G':
        upload_file_to_gcs(local_root_path=local_root_path, gcs_path=upload_path, file_name=file_name)
    # UPLOAD_MODE == 'L'인 경우 아무것도 하지 않음.

#  --- S3 upload method ---
def upload_file_to_s3(local_root_path: str, s3_path: str, file_name: str = None):
    """
    - file_name = None인 경우 local_path의 전체 파일 s3 업로드
    - file_name != None인 경우 개별 파일 s3 업로드
    """
    local_root = Path(local_root_path).expanduser()
    parsed_s3_path = urlparse(s3_path)
    save_bucket = parsed_s3_path.netloc
    save_prefix = parsed_s3_path.path.lstrip('/')

    s3_client = boto3.client('s3')

    files_to_upload = []
    if file_name is None:
        # 디렉터리 전체 업로드
        print(f"Preparing to upload directory {local_root} to s3://{save_bucket}/{save_prefix}")
        for local_path in local_root.rglob('*'):
            if local_path.is_file():
                s3_key = str(save_prefix / local_path.relative_to(local_root))
                files_to_upload.append((str(local_path), s3_key))
    else:
        # 단일 파일 업로드
        local_file = local_root/file_name
        print(f"Preparing to upload file {local_file} to s3://{save_bucket}/{save_prefix}")
        if local_file.is_file():
            s3_key = str(save_prefix / local_file.relative_to(local_root))
            files_to_upload.append((str(local_file), s3_key))

    for local_file_path, s3_key in files_to_upload:
        try:
            s3_client.upload_file(local_file_path, save_bucket, s3_key)
            print(f"  Uploaded {local_file_path} to s3://{save_bucket}/{s3_key}")
        except Exception as e:
            print(f"  Error uploading {local_file_path}: {e}")

#  --- GCP upload method ---
def upload_file_to_gcs(local_root_path: str, gcs_path: str, file_name: str = None):
    local_root = Path(local_root_path).expanduser()
    parsed_gcs_path = urlparse(gcs_path)
    bucket_name = parsed_gcs_path.netloc
    save_prefix = parsed_gcs_path.path.lstrip('/')

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files_to_upload = []
    if file_name is None:
        print(f"Preparing to upload directory {local_root} to gs://{bucket_name}/{save_prefix}")
        for local_path in local_root.rglob('*'):
            if local_path.is_file():
                rel_path = local_path.relative_to(local_root).as_posix()
                gcs_key = f"{save_prefix}/{rel_path}"
                files_to_upload.append((local_path, gcs_key))
    else:
        local_file = local_root / file_name
        print(f"Preparing to upload file {local_file} to gs://{bucket_name}/{save_prefix}")
        if local_file.is_file():
            gcs_key = f"{save_prefix}/{file_name}"
            files_to_upload.append((local_file, gcs_key))

    for local_file_path, gcs_key in files_to_upload:
        try:
            blob = bucket.blob(gcs_key)
            blob.upload_from_filename(str(local_file_path))
            print(f"  Uploaded {local_file_path} to gs://{bucket_name}/{gcs_key}")
        except Exception as e:
            print(f"  Error uploading {local_file_path}: {e}")

# 전처리 및 키포인트 변환 함수
def mediapipe_hands_to_openpose_format(mp_hand_landmarks, image_width, image_height):
    hand_keypoints = np.zeros((21, 3))
    if mp_hand_landmarks:
        for i, landmark in enumerate(mp_hand_landmarks.landmark):
            hand_keypoints[i] = [landmark.x * image_width, landmark.y * image_height, 1.0]
    return hand_keypoints

def mediapipe_to_openpose_keypoints(results, image_width, image_height):
    pose = np.zeros((25, 3)); face = np.zeros((70, 3))
    left_hand = np.zeros((21, 3)); right_hand = np.zeros((21, 3))
    def to_pixel_coords(landmark):
        return [landmark.x * image_width, landmark.y * image_height, landmark.visibility if hasattr(landmark, 'visibility') else 1.0]
    if results.pose_landmarks:
        mp_pose = results.pose_landmarks.landmark
        pose[0] = to_pixel_coords(mp_pose[0]) # Nose
        pose[1] = [(to_pixel_coords(mp_pose[11])[0] + to_pixel_coords(mp_pose[12])[0]) / 2, (to_pixel_coords(mp_pose[11])[1] + to_pixel_coords(mp_pose[12])[1]) / 2, 1.0] # Neck
        pose[2] = to_pixel_coords(mp_pose[12]); pose[3] = to_pixel_coords(mp_pose[14]); pose[4] = to_pixel_coords(mp_pose[16]) # Right Arm
        pose[5] = to_pixel_coords(mp_pose[11]); pose[6] = to_pixel_coords(mp_pose[13]); pose[7] = to_pixel_coords(mp_pose[15]) # Left Arm
    left_hand = mediapipe_hands_to_openpose_format(results.left_hand_landmarks, image_width, image_height)
    right_hand = mediapipe_hands_to_openpose_format(results.right_hand_landmarks, image_width, image_height)
    # 현재 face 전체 0임.
    return np.concatenate([pose, face, left_hand, right_hand], axis=0)

# 인코더 통과한 데이터를 리턴함.
def main_preprocess_sequence(sequence: np.ndarray, max_len: int=CROP_LEN, umap_path:str=UMAP_LOAD_PATH) -> np.ndarray:
    sequence = np.array(sequence)
    original_len = len(sequence)

    if original_len > max_len: sequence = sequence[:max_len]
    else:
        padding = np.zeros((max_len - original_len, sequence.shape[1], sequence.shape[2]))
        sequence = np.concatenate([sequence, padding], axis=0)

    # Padding 프레임 제외 (원본 길이까지만)
    valid_frames = sequence[:original_len]

    # 모든 valid 프레임의 모든 점 수집
    all_points = valid_frames.reshape(-1, 2)  # (T*N, 2)

    # Valid points만 (confidence > 0 또는 (0,0)이 아닌 점)
    # 여기서는 단순히 모든 점 사용
    all_x = all_points[:, 0]
    all_y = all_points[:, 1]

    # Bounding box 계산
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    scale = max(x_max - x_min, y_max - y_min) / 2

    if scale < 1e-6:  # 거의 0
        scale = 1.0

    # 전체 시퀀스를 같은 center와 scale로 정규화
    normalized_sequence = (sequence - [center_x, center_y]) / scale

    # Clip to [-1, 1]
    normalized_sequence = np.clip(normalized_sequence, -2, 2)

    # 선택된 키포인트만 추출
    selected_seq = normalized_sequence[:, POINT_LANDMARKS, :]  # (MAX_LEN, 49, 2)

    # Flatten
    selected_seq = selected_seq.reshape(max_len, -1)  # (MAX_LEN, 98)

    # NaN 처리
    selected_seq = np.nan_to_num(selected_seq, 0)

    # Umap embedding
    umap_encoder = keras.models.load_model(umap_path)
    embedding = umap_encoder.predict(selected_seq)

    return embedding # (T, 32)
