import os
import json
import numpy as np
from typing import Optional, List, Dict, Any
import tensorflow as tf
import glob

# 커스텀
from src.config import MAX_LEN, BATCH_SIZE, POINT_LANDMARKS, DIRECTIONS

# 폴더의 데이터를 가져오는 역할.
class KSLDataLoader:
    # samples로 가공된 폴더 데이터 확인
    def __init__(self, data_path, label_map, is_training=True):
        self.data_path = data_path
        self.label_map = label_map
        self.is_training = is_training
        self.samples = []
        self.generate_samples()

        print("Samples verification =====================================")
        for sample in self.samples: # samples는 sequence(영상), folder_name, folder_meaning 를 가진 데이터임.
            seq = sample['sequence']
            folder_meaning = sample['folder_meaning']
            print(f"  - Sample: {sample['folder_name']} (meaning: {folder_meaning}), shape: {seq.shape}")

    # 폴더 접근 -> 영상 시퀀스를 samples에 추가
    def generate_samples(self):
        """영상 키포인트들 -> 시퀀스+라벨 데이터 -> 샘플 데이터화"""
        print(f"\nStarting data loading from: {self.data_path}")

        # 영상 시퀀스를 samples에 추가하는 작업.
        for folder_name, folder_meaning in self.label_map.items():  # folder_name : 폴더명 ; folder_meaning : 수어의미
            label_dir = os.path.join(self.data_path, folder_name)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory {label_dir} not found")
                continue
            #(f"\nProcessing folder_name: {folder_name} ({folder_meaning})")
            if self.is_training:
                for direction in DIRECTIONS:
                    direction_dir = os.path.join(label_dir, f"{folder_name}_{direction}")
                    if not os.path.exists(direction_dir):
                        print(f"  Direction {direction} not found")
                        continue

                    print(f"  Processing direction: {direction}")

                    # 개개인 영상 폴더 내  (REAL12, REAL17, etc.)
                    person_dirs = glob.glob(os.path.join(direction_dir, f"*REAL*_{direction}"))
                    print(f"    Found {len(person_dirs)} person directories")

                    for person_dir in person_dirs:
                        if not os.path.isdir(person_dir):
                            continue

                        person_name = os.path.basename(person_dir)

                        # 사람 영상
                        keypoint_files = sorted(glob.glob(os.path.join(person_dir, "*_keypoints.json")))

                        if len(keypoint_files) > 0:
                            sequences = self.generate_sequence(keypoint_files)  # 단일 또는 리스트 반환
                            if sequences is not None:
                                # sequences가 list인지 확인
                                if isinstance(sequences, list):
                                    for seq in sequences:
                                        self.samples.append({
                                            'sequence': seq,
                                            'folder_name': folder_meaning,
                                            'folder_meaning': list(self.label_map.keys()).index(folder_name)
                                        })
                                else:  # 단일 ndarray
                                    self.samples.append({
                                        'sequence': sequences,
                                        'folder_name': folder_meaning,
                                        'folder_meaning': list(self.label_map.keys()).index(folder_name)
                                    })

        print(f"\nTotal samples loaded: {len(self.samples)}")

    def generate_sequence(self, keypoint_files: List[str]) -> Optional[np.ndarray]:
        """
        Args:
          keypoint_files (List[str]) : keypoint json file path list
        Returns:
          Optional[np.ndarray]:
                if sequence generate success, numpy array (N x 98 x 3) of keypoints in a single video
                not None
        """
        sequence = []

        for file_path in keypoint_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                if 'people' not in data:
                    keypoints = None
                else:
                    people_data = data['people']

                    if isinstance(people_data, dict):
                        person = people_data
                    else:
                        if len(people_data) == 0:
                            return None
                        person = people_data[0]

                    keypoints = self.process_folder_data(person)

                if keypoints is not None:
                    sequence.append(keypoints)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(sequence) > 0:
            return np.array(sequence)
        return None

    def process_folder_data(self, data):
        """
        data는 people: {pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d}
        방식으로 들어가 있음.

        OpenPose JSON 포맷에서 가공 후 {x,y,confidence} 형태 넘파이 배열 리턴.

        - OpenPose pose_keypoints_2d 총 25개인지
        - OpenPose face_keypoints_2d 총 70개인지
        - OpenPose hand_left_keypoints_2d 총 21개인지
        - OpenPose hand_right_keypoints_2d 총 21개인지

        - z축 고려하는 문제 & face 사용 여부도 따로 값으로 취급하기

        가공 방식
        1. 정규화
        2. 클리핑
        3. 결측값 처리
        """
        person = data
        # Extract pose keypoints (25 points, 3 values each)
        pose_kp = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
        if pose_kp.shape[0] != 25:
            return None

        # Extract face keypoints (70 points)
        face_kp = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
        if face_kp.shape[0] != 70:
            return None

        # Extract hand keypoints (21 points each)
        left_hand_kp = np.array(person.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
        if left_hand_kp.shape[0] != 21:
            return None

        right_hand_kp = np.array(person.get('hand_right_keypoints_2d', [])).reshape(-1, 3)
        if right_hand_kp.shape[0] != 21:
            return None

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
            scale = max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y)) / 2

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

        # Concatenate confidence scores
        all_conf = np.concatenate([
            pose_conf,
            face_conf,
            left_hand_conf,
            right_hand_conf
        ], axis=0)

        # Use confidence as z coordinate
        all_keypoints = np.concatenate([all_keypoints, all_conf], axis=1)

        # Replace invalid points (confidence < 0.1) with zeros
        mask = all_conf[:, 0] < 0.1
        all_keypoints[mask] = 0

        return all_keypoints

    def _preprocess_sample(self, sample):
        # 샘플 전처리 함수
        seq = sample['sequence']
        # Pad or truncate
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        elif len(seq) < MAX_LEN:
            pad_length = MAX_LEN - len(seq)
            padding = np.zeros((pad_length, seq.shape[1], seq.shape[2]))
            seq = np.concatenate([seq, padding], axis=0)

        # 1. Extract selected landmarks
        selected_seq = seq[:, POINT_LANDMARKS, :]

        # 2. Normalize using neck
        neck_pos = seq[:, 1:2, :2]
        neck_mean = np.nanmean(neck_pos, axis=(0, 1), keepdims=True)
        if np.isnan(neck_mean).any():
            neck_mean = np.array([[[0.5, 0.5]]])

        # 3. Extract x, y and normalize
        selected_xy = selected_seq[:, :, :2]
        std = np.nanstd(selected_xy)
        if std == 0:
            std = 1.0
        selected_xy = (selected_xy - neck_mean) / std

        # 4. Calculate derivatives
        #dx = np.diff(selected_xy, n=1, axis=0, prepend=selected_xy[0:1])
        #dx2 = np.diff(selected_xy, n=2, axis=0, prepend=selected_xy[0:2])

        # 5. Flatten and concatenate
        processed = np.concatenate([
            selected_xy.reshape(MAX_LEN, -1),
            #dx.reshape(MAX_LEN, -1),
            #dx2.reshape(MAX_LEN, -1)
        ], axis=-1)
        processed = selected_xy.reshape(MAX_LEN, -1)

        # 6. Final cleanup
        processed = np.nan_to_num(processed, 0)
        label = sample['folder_meaning']

        return processed.astype(np.float32), np.int32(label)

    def _data_generator(self):
        for sample in self.samples:
            yield self._preprocess_sample(sample)

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self._data_generator,
            output_signature=(
                tf.TensorSpec(shape=(MAX_LEN, 98), dtype=tf.float32), # (시퀀스 길이, 특징 개수)
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return dataset