# ========================== src/dataset.py ==========================
"""
OpenPose-based Korean SL dataset loader
--------------------------------------
* 폴더 구조
    data/
      train/NIA_SL_SEN0181/NIA_SL_SEN0181_D/NIA_SL_SEN0181_REAL12_D/*.json
      val  / ...

* 반환 형태
    X : (batch, MAX_LEN, 67, 3) float32
    y : (batch, NUM_CLASSES)     one-hot
"""

import os, glob, json, matplotlib.pyplot as plt, numpy as np, tensorflow as tf
from src.config import MAX_LEN, NUM_CLASSES, BATCH_SIZE, POINT_LANDMARKS

class KSLDataLoader:
    """Korean Sign Language Data Loader"""
    def __init__(self, data_path, label_map, is_training=True):
        self.data_path = data_path
        self.label_map = label_map
        self.is_training = is_training
        self.samples = []
        self._load_data()

        for sample in self.samples: # samples는 sequence(영상 키포인트 묶음), label, label_idx 를 가진 데이터임.
            seq = sample['sequence']
            label_idx = sample['label_idx']
            print(f"  - Sample: {sample['label']} (idx: {label_idx}), shape: {seq.shape}")

        if self.samples != None:
            plot_sequence_length_distribution(samples=self.samples)

    def _load_data(self):
        """Load all keypoint sequences and their labels"""
        print(f"\nStarting data loading from: {self.data_path}")

        for label, label_text in self.label_map.items(): # label : 폴더명 ; label_text : 의미
            label_dir = os.path.join(self.data_path, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory {label_dir} not found")
                continue

            print(f"\nProcessing label: {label} ({label_text})")

            # Check if this is train or val data by directory structure
            if self.is_training:
                # Train data: has direction folders
                for direction in ['D', 'F', 'L', 'R', 'U']:
                    direction_dir = os.path.join(label_dir, f"{label}_{direction}")
                    if not os.path.exists(direction_dir):
                        print(f"  Direction {direction} not found")
                        continue

                    print(f"  Processing direction: {direction}")

                    # # Load morpheme data if training
                    # morpheme_data = None
                    # morpheme_path = os.path.join(direction_dir, f"{label}_{direction}_morpheme.json")
                    # if os.path.exists(morpheme_path):
                    #     with open(morpheme_path, 'r', encoding='utf-8') as f:
                    #         morpheme_data = json.load(f)

                    # Process each person's data (REAL12, REAL17, etc.)
                    person_dirs = glob.glob(os.path.join(direction_dir, f"*REAL*_{direction}"))
                    print(f"    Found {len(person_dirs)} person directories")

                    for person_dir in person_dirs:
                        if not os.path.isdir(person_dir):
                            continue

                        person_name = os.path.basename(person_dir)

                        # Load all keypoint files for this person
                        keypoint_files = sorted(glob.glob(os.path.join(person_dir, "*_keypoints.json")))
                        print(f"      {person_name}: {len(keypoint_files)} keypoint files")

                        if len(keypoint_files) > 0:
                            sequence = self._load_sequence(keypoint_files)  # Limit to first 100 frames -> 키포인트 프레임 보통 140개까지인데 100개로 자르면 안됨!!
                            if sequence is not None:
                                self.samples.append({
                                    'sequence': sequence,
                                    'label': label_text,
                                    'label_idx': list(self.label_map.keys()).index(label)
                                    # 'morpheme_data': morpheme_data
                                })
                                print(f"        ✓ Successfully loaded sequence with shape: {sequence.shape}")
            else:
                # Val data: person folders directly under label folder
                person_dirs = glob.glob(os.path.join(label_dir, f"{label}_REAL*"))
                print(f"  Found {len(person_dirs)} person directories")

                for person_dir in person_dirs:
                    if not os.path.isdir(person_dir):
                        continue

                    person_name = os.path.basename(person_dir)

                    # Load all keypoint files for this person
                    keypoint_files = sorted(glob.glob(os.path.join(person_dir, "*_keypoints.json")))
                    print(f"    {person_name}: {len(keypoint_files)} keypoint files")

                    if len(keypoint_files) > 0:
                        sequence = self._load_sequence(keypoint_files)  # Limit to first 100 frames
                        if sequence is not None:
                            self.samples.append({
                                'sequence': sequence,
                                'label': label_text,
                                'label_idx': list(self.label_map.keys()).index(label),
                                'morpheme_data': None
                            })
                            print(f"      ✓ Successfully loaded sequence with shape: {sequence.shape}")

        print(f"\nTotal samples loaded: {len(self.samples)}")

    def _load_sequence(self, keypoint_files):
        """사람 폴더 내의 키포인트 파일들을 묶어 전체 영상 길이"""
        sequence = []

        for file_path in keypoint_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract keypoints from OpenPose format
                keypoints = self._extract_keypoints_from_openpose(data)
                if keypoints is not None:
                    sequence.append(keypoints)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        if len(sequence) > 0:
            return np.array(sequence)
        return None

    def _extract_keypoints_from_openpose(self, data):
        """
        OpenPose JSON 포맷에서 키포인트 뽑아내기

        """
        try:
            if 'people' not in data:
                return None

            people_data = data['people']

            # Handle the single person dictionary format
            if isinstance(people_data, dict):
                person = people_data
            else:
                # If it's a list (shouldn't happen based on your data)
                if len(people_data) == 0:
                    return None
                person = people_data[0]

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

            # Normalize coordinates to [-1, 1] range
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

                # Normalize coordinates to [-1, 1]
                pose_xy = (pose_xy - [center_x, center_y]) / scale
                face_xy = (face_xy - [center_x, center_y]) / scale
                left_hand_xy = (left_hand_xy - [center_x, center_y]) / scale
                right_hand_xy = (right_hand_xy - [center_x, center_y]) / scale

                # Clip to reasonable range
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

        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_dataset(self):
        """폴더 데이터 -> tf.data.Dataset 형식으로 변환"""
        sequences = []
        labels = []

        #
        print(f"Processing {len(self.samples)} samples...")

        for i, sample in enumerate(self.samples):
            seq = sample['sequence']

            # Pad or truncate sequence to MAX_LEN
            if len(seq) > MAX_LEN:
                seq = seq[:MAX_LEN]
            elif len(seq) < MAX_LEN:
                pad_length = MAX_LEN - len(seq)
                padding = np.zeros((pad_length, seq.shape[1], seq.shape[2]))
                seq = np.concatenate([seq, padding], axis=0)

            # Apply preprocessing manually
            # 1. Extract selected landmarks
            selected_seq = seq[:, POINT_LANDMARKS, :]  # Shape: (384, 95, 3)

            # 2. Normalize using neck as reference
            neck_pos = seq[:, 1:2, :2]  # Neck position (x, y)
            neck_mean = np.nanmean(neck_pos, axis=(0, 1), keepdims=True)
            if np.isnan(neck_mean).any():
                neck_mean = np.array([[[0.5, 0.5]]])

            # 3. Extract x, y coordinates only
            selected_xy = selected_seq[:, :, :2]  # Shape: (384, 95, 2)

            # 4. Normalize
            std = np.nanstd(selected_xy)
            if std == 0:
                std = 1.0
            selected_xy = (selected_xy - neck_mean) / std

            # 5. Calculate derivatives
            dx = np.zeros_like(selected_xy)
            dx[1:] = selected_xy[1:] - selected_xy[:-1]

            dx2 = np.zeros_like(selected_xy)
            dx2[2:] = selected_xy[2:] - selected_xy[:-2]

            # 6. Flatten and concatenate
            x_flat = selected_xy.reshape(MAX_LEN, -1)  # (384, 190)
            dx_flat = dx.reshape(MAX_LEN, -1)  # (384, 190)
            dx2_flat = dx2.reshape(MAX_LEN, -1)  # (384, 190)

            processed = np.concatenate([x_flat, dx_flat, dx2_flat], axis=-1)  # (384, 570)

            # 7. Replace NaN with 0
            processed = np.nan_to_num(processed, 0)

            sequences.append(processed)
            labels.append(sample['label_idx'])

            if i % 10 == 0:
                print(f"  Processed {i+1}/{len(self.samples)} samples...")

        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        print(f"Final dataset shape: sequences={sequences.shape}, labels={labels.shape}")   # sequence.shape = (450, 384, 570)

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return dataset, len(sequences)


def create_label_map():
    """Create label mapping for Korean Sign Language"""
    label_map = {
        'NIA_SL_SEN0181': '도와주세요',
        'NIA_SL_SEN0354': '안녕하세요',
        'NIA_SL_SEN0355': '감사합니다',
        'NIA_SL_SEN0356': '죄송합니다',
        'NIA_SL_SEN2000': '수고하셨습니다'
    }
    return label_map

# sequence MAX_LEN 확인 겸
def plot_sequence_length_distribution(samples):
    lengths = [len(sample['sequence']) for sample in samples]

    print(f"총 {len(lengths)} 개의 샘플")
    print(f"최소 길이: {min(lengths)}")
    print(f"최대 길이: {max(lengths)}")
    print(f"평균 길이: {np.mean(lengths):.2f}")
    print(f"중앙값 길이: {np.median(lengths)}")

    plt.figure(figsize=(10,5))
    plt.hist(lengths, bins=30, color='skyblue',edgecolor='black')
    plt.axvline(x=384, color='red', linestyle='--', label='MAX_LEN=384')
    plt.title('시퀀스 길이 분포')
    plt.xlabel('시퀀스 길이 (프레임 수)')
    plt.ylabel('샘플 수')
    plt.legend()
    plt.grid(True)
    plt.show()

# 테스트용 실행 -------------------------------------------------------
if __name__ == "__main__":
    # Create label map
    label_map = create_label_map()

    # Update NUM_CLASSES
    global NUM_CLASSES
    NUM_CLASSES = len(label_map)

    print("Label mapping:")
    for key, value in label_map.items():
        print(f"  {key}: {value}")

    print(f"Number of classes: {NUM_CLASSES}")
    train_loader = KSLDataLoader('data/train', label_map, is_training=True)
    train_dataset, train_size = train_loader.get_dataset()
    for x, y in train_dataset.take(1):
        print("batch X:", x.shape)       # (B, 384, 67, 3)
        print("batch y:", y.shape)
