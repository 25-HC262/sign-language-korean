from src.config import KSL_SENTENCES, POINT_LANDMARKS, DIRECTIONS, VALIDATION_SPLIT, MAX_LEN, S3_UMAP_PATH, \
    BATCH_SIZE, OUTPUT_DIM

from urllib.parse import urlparse
import os, boto3, json, glob, random
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pathlib import Path

class TrainDataLoader:
    def __init__(self, data_path, save_path=None, s3_save_path=None, samples_per_class=1000, is_training_transformer=False):
        self.data_path = data_path
        # s3 경로 확인
        self.is_s3 = data_path.startswith('s3://')
        if self.is_s3:
            parsed_s3_path = urlparse(data_path)
            self.s3_bucket = parsed_s3_path.netloc
            self.s3_prefix = parsed_s3_path.path.lstrip('/')
            self.s3_client = boto3.client('s3')
        self.save_path = os.path.expanduser(save_path) if save_path is not None else None  # f"{base_path}/졸업프로젝트"
        self.s3_save_path = s3_save_path

        self.samples_per_class = samples_per_class
        self.is_training_transformer = is_training_transformer
        if self.is_training_transformer:
            import h5py
            umap_encoder_path = "models/umap_models/encoder.h5"
            try:
                self.umap_encoder = tf.keras.models.load_model(umap_encoder_path)

            except Exception as e:
                print(f"Error loading encoder model: {e}")
                # 또는 커스텀 객체 사용
                self.umap_encoder = tf.keras.models.load_model(
                    umap_encoder_path,
                    custom_objects={'InputLayer': tf.keras.layers.InputLayer},
                    compile=False
                )

        self.videos = []

    def create_umap_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        files_by_class = self._get_all_filepaths()
        sampled_files = self._stratified_sample_files(files_by_class=files_by_class)

        dataset = tf.data.Dataset.from_tensor_slices(sampled_files)
        print(f"Loading and processing data from {self.data_path}...")
        dataset = dataset.map(
            lambda path: tf.py_function(
                self._load_json_from_path, inp=[path], Tout=tf.float32
            ), num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(lambda x: tf.reshape(x, [-1]))

        numpy_dataset = np.array([item.numpy() for item in tqdm(dataset, total=len(sampled_files), desc="Processing Dataset")])

        validation_size = max(1, int(len(numpy_dataset) * VALIDATION_SPLIT)) # 최소 1개

        if validation_size >= len(numpy_dataset):
            raise ValueError(f"Not enough samples: {len(numpy_dataset)}. Need at least 2.")

        train_dataset = numpy_dataset[:-validation_size]
        validation_dataset = numpy_dataset[-validation_size:]

        print(f"Train samples: {len(train_dataset)}, Validation samples: {len(validation_dataset)}")
        return train_dataset, validation_dataset

    def create_transformer_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        # self.videos 구성
        self._get_all_filepaths()
        def _data_generator():
            for video in self.videos:
                seq = video['sequence']
                if len(seq) > MAX_LEN:
                    seq = seq[:MAX_LEN]
                else:
                    padding = np.zeros((MAX_LEN - len(seq), OUTPUT_DIM))
                    seq = np.concatenate([seq, padding], axis=0)
                yield seq.astype(np.float32), np.int32(video['class_label'])
        full_dataset = tf.data.Dataset.from_generator(
            _data_generator, # 함수 자체 전달
            output_signature=(
                tf.TensorSpec(shape=(MAX_LEN, OUTPUT_DIM), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        full_dataset = full_dataset.shuffle(buffer_size=1024)
        dataset_size = len(self.videos)
        val_size = int(dataset_size*VALIDATION_SPLIT)
        train_size = dataset_size-val_size

        val_dataset = full_dataset.take(val_size)
        train_dataset = full_dataset.skip(val_size)

        train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        print(f"전체 데이터셋 크기: {dataset_size}")
        print(f"훈련 데이터셋 크기 (예상): {train_size}")
        print(f"검증 데이터셋 크기 (예상): {val_size}")
        print("\nTrain Dataset Spec:\n", train_dataset)
        print("\nValidation Dataset Spec:\n", val_dataset)

        return train_dataset, val_dataset

    def _get_all_filepaths(self) -> dict:
        print(f"\nStarting data loading from: {self.data_path}")
        files_by_class = defaultdict(list)

        for folder_name, folder_meaning in KSL_SENTENCES.items():
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

                    # When training Transformer
                    if self.is_training_transformer:
                        keypoints_batch = []
                        for kp_file in keypoint_files:
                            try:
                                keypoints = self._load_json_from_path(kp_file)
                                keypoints = keypoints.reshape(-1) # (98, )
                                keypoints_batch.append(keypoints)
                            except Exception as e:
                                print(f"file load error: {kp_file} - {e}")
                                continue

                        if len(keypoints_batch) > 0:
                            keypoints_batch = np.array(keypoints_batch)             # (T, 98)
                            embeddings = self.umap_encoder.predict(keypoints_batch) # (T, 32)

                            self.videos.append({
                                'sequence': embeddings, # 이미 (T, 32) 형태이므로 바로 사용
                                'class_label': list(KSL_SENTENCES.keys()).index(folder_name)
                            })

        return files_by_class

    def _stratified_sample_files(self, files_by_class: dict) -> list:
        sampled_files = []
        print("\n--- Stratified Sampling ---")
        for class_label, file_list in files_by_class.items():
            num_files = len(file_list)
            num_to_sample = min(self.samples_per_class, num_files)
            print(f"Class: '{class_label}': Found {num_files}")
            sampled_files.extend(random.sample(file_list, num_to_sample))
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

def upload_file_to_s3(local_root_path: str, s3_path: str, file_name: str = None):
    '''
    file_name=None인 경우 local_path의 전체 파일 s3 업로드
    file_name!=None인 경우 개별 파일 s3 업로드
    '''
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
def main_preprocess_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = np.array(sequence)
    original_len = len(sequence)

    if original_len > MAX_LEN: sequence = sequence[:MAX_LEN]
    else:
        padding = np.zeros((MAX_LEN - original_len, sequence.shape[1], sequence.shape[2]))
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
    selected_seq = selected_seq.reshape(MAX_LEN, -1)  # (MAX_LEN, 98)

    # NaN 처리
    selected_seq = np.nan_to_num(selected_seq, 0)

    # Umap embedding
    umap_encoder_path = os.path.join(os.path.expanduser(S3_UMAP_PATH), 'encoder.keras')
    umap_encoder = tf.keras.models.load_model(umap_encoder_path)
    embedding = umap_encoder.predict(selected_seq)

    return embedding # (T, 32)
