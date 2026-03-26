"""
ASL 파인튜닝 호환 데이터로더.
create_dataset.py의 TrainDataLoader를 상속받아 다음을 변경합니다:
  1. POINT_LANDMARKS_ASL (90개, 얼굴 포함) 사용
  2. per-frame bounding box 정규화 대신 sequence-level ASL 정규화 사용
  3. 로컬 파일만 지원 (파인튜닝 전용)
"""
import importlib
importlib.import_module("src.tf_keras_config")

import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.asl_finetune_config import (
    POINT_LANDMARKS_ASL, NUM_NODES_ASL, CHANNELS_ASL,
    KSL_SENTENCES, DIRECTIONS, VALIDATION_SPLIT, MAX_LEN,
    BATCH_SIZE, PAD, NECK_IDX_IN_SELECTED,
)
from src.config import L_DATA, L_TEST, Trainer, UMAP_OUTPUT_DIM


class ASLCompatDataLoader:
    """
    ASL 스타일 정규화와 확장 랜드마크(얼굴 포함)를 사용하는 데이터로더.
    로컬 OpenPose JSON 파일만 처리합니다.
    """

    def __init__(self, data_path=L_DATA, test_data_path=L_TEST, use_asl_norm: bool = True):
        """
        Args:
            data_path: 학습 데이터 경로 (로컬 디렉토리)
            test_data_path: 테스트 데이터 경로
            use_asl_norm: True이면 시퀀스 레벨 ASL 정규화 사용,
                          False이면 기존 per-frame bounding box 정규화 사용
        """
        self.data_path = data_path
        self.test_data_path = test_data_path
        self.use_asl_norm = use_asl_norm
        self.output_dim = CHANNELS_ASL  # 540
        self.label_map = {name: i for i, name in enumerate(KSL_SENTENCES.keys())}
        self.videos = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_gm_train_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        self._load_videos(path=self.data_path)
        dataset_size = len(self.videos)
        print(f"전체 데이터셋 크기: {dataset_size}")

        val_size = int(dataset_size * VALIDATION_SPLIT)
        train_size = dataset_size - val_size
        print(f"Train: {train_size}, Val: {val_size}")

        random.seed(42)
        random.shuffle(self.videos)

        full_ds = self._make_tf_dataset(self.videos)
        val_ds = full_ds.take(val_size)
        train_ds = full_ds.skip(val_size)
        self.videos = []
        return train_ds, val_ds

    def create_test_dataset(self) -> tf.data.Dataset:
        self._load_videos(path=self.test_data_path)
        ds = self._make_tf_dataset(self.videos)
        self.videos = []
        return ds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_tf_dataset(self, videos) -> tf.data.Dataset:
        max_len = MAX_LEN
        output_dim = self.output_dim

        def _gen():
            for video in videos:
                seq = video['sequence']
                T = len(seq)
                if T > max_len:
                    seq = seq[:max_len]
                else:
                    pad = np.zeros((max_len - T, output_dim), dtype=np.float32)
                    seq = np.concatenate([seq, pad], axis=0)
                yield seq.astype(np.float32), np.int32(video['class_label'])

        return tf.data.Dataset.from_generator(
            _gen,
            output_signature=(
                tf.TensorSpec(shape=(max_len, output_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )
        )

    def _load_videos(self, path=None):
        path = str(self.data_path) if path is None else str(path)
        base_path = Path(path)
        self.videos = []

        with tqdm(KSL_SENTENCES.items(), desc="[클래스 로딩]", position=0) as pbar:
            for folder_name, _ in pbar:
                pbar.set_postfix(cls=folder_name)
                for direction in DIRECTIONS:
                    direction_dir = base_path / folder_name / f"{folder_name}_{direction}"
                    if not direction_dir.exists():
                        continue
                    person_paths = sorted(direction_dir.glob(f"*REAL*_{direction}"))

                    for person_path in tqdm(person_paths, desc=f"  {folder_name}_{direction}", position=1, leave=False):
                        if not person_path.is_dir():
                            continue
                        kp_files = sorted(person_path.glob("*_keypoints.json"))

                        raw_frames = []
                        for kp_file in kp_files:
                            try:
                                frame = self._load_json_frame(str(kp_file))
                                raw_frames.append(frame)  # (90, 2)
                            except Exception as e:
                                tqdm.write(f"  skip {kp_file.name}: {e}")
                                continue

                        if len(raw_frames) == 0:
                            continue

                        # (T, 90, 2) → (T, 540)
                        seq = self._add_dynamic_features(np.array(raw_frames))
                        self.videos.append({
                            'sequence': seq,
                            'class_label': self.label_map[folder_name],
                        })

    def _load_json_frame(self, file_path: str) -> np.ndarray:
        """JSON 파일 한 프레임 → (90, 2) 키포인트 배열 반환."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert 'people' in data, "keypoints 없음"
        person = data['people']
        assert isinstance(person, dict) and len(person) > 0, "빈 people 데이터"
        return self._json_to_numpy(person)

    def _json_to_numpy(self, person: Dict[str, Any]) -> np.ndarray:
        """
        OpenPose JSON 한 프레임 → (90, 2) 선택된 랜드마크 (x, y).
        use_asl_norm=True이면 정규화 없이 raw 픽셀 좌표 반환.
        use_asl_norm=False이면 per-frame bounding box 정규화 적용.
        """
        pose_kp      = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
        face_kp      = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
        left_hand_kp = np.array(person.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
        right_hand_kp= np.array(person.get('hand_right_keypoints_2d', [])).reshape(-1, 3)

        assert pose_kp.shape[0]       == 25, f"pose: {pose_kp.shape[0]}"
        assert face_kp.shape[0]       == 70, f"face: {face_kp.shape[0]}"
        assert left_hand_kp.shape[0]  == 21, f"lhand: {left_hand_kp.shape[0]}"
        assert right_hand_kp.shape[0] == 21, f"rhand: {right_hand_kp.shape[0]}"

        pose_xy  = pose_kp[:, :2]
        face_xy  = face_kp[:, :2]
        lhand_xy = left_hand_kp[:, :2]
        rhand_xy = right_hand_kp[:, :2]

        if not self.use_asl_norm:
            # per-frame bounding box 정규화 (기존 방식)
            pose_conf  = pose_kp[:, 2:3]
            face_conf  = face_kp[:, 2:3]
            lhand_conf = left_hand_kp[:, 2:3]
            rhand_conf = right_hand_kp[:, 2:3]

            all_x = np.concatenate([
                pose_xy[pose_conf[:, 0] > 0.1, 0],
                face_xy[face_conf[:, 0] > 0.1, 0],
                lhand_xy[lhand_conf[:, 0] > 0.1, 0],
                rhand_xy[rhand_conf[:, 0] > 0.1, 0],
            ])
            all_y = np.concatenate([
                pose_xy[pose_conf[:, 0] > 0.1, 1],
                face_xy[face_conf[:, 0] > 0.1, 1],
                lhand_xy[lhand_conf[:, 0] > 0.1, 1],
                rhand_xy[rhand_conf[:, 0] > 0.1, 1],
            ])
            if len(all_x) > 0:
                cx = (np.max(all_x) + np.min(all_x)) / 2
                cy = (np.max(all_y) + np.min(all_y)) / 2
                scale = max(np.max(all_x) - np.min(all_x), np.max(all_y) - np.min(all_y)) / 2
                if scale < 1e-6:
                    scale = 1.0
                pose_xy  = np.clip((pose_xy  - [cx, cy]) / scale, -2, 2)
                face_xy  = np.clip((face_xy  - [cx, cy]) / scale, -2, 2)
                lhand_xy = np.clip((lhand_xy - [cx, cy]) / scale, -2, 2)
                rhand_xy = np.clip((rhand_xy - [cx, cy]) / scale, -2, 2)

        # all_keypoints 배열: pose(25) + face(70) + lhand(21) + rhand(21) = 137개
        all_kp = np.concatenate([pose_xy, face_xy, lhand_xy, rhand_xy], axis=0)

        # POINT_LANDMARKS_ASL 인덱스로 90개 선택
        selected = all_kp[POINT_LANDMARKS_ASL, :]
        return np.nan_to_num(selected, nan=0.0).astype(np.float32)  # (90, 2)

    def _apply_asl_norm(self, sequence: np.ndarray) -> np.ndarray:
        """
        ASL 스타일 시퀀스 레벨 정규화.
        sequence: (T, 90, 2) raw 픽셀 좌표
        - 목 포인트(NECK_IDX_IN_SELECTED)를 기준으로 중심화
        - 시퀀스 전체 std로 정규화
        """
        # 목 위치로 중심화 (T, 1, 2)
        neck = sequence[:, NECK_IDX_IN_SELECTED:NECK_IDX_IN_SELECTED + 1, :]
        centered = sequence - neck

        # 시퀀스 전체 std로 정규화
        valid = centered[~np.isnan(centered)]
        if len(valid) > 0:
            std = np.std(valid)
            if std > 1e-6:
                centered = centered / std

        return np.nan_to_num(centered, nan=0.0)

    def _add_dynamic_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        입력: (T, 90, 2) — raw 또는 per-frame 정규화된 키포인트
        출력: (T, 540) — position + velocity + acceleration

        use_asl_norm=True이면 먼저 시퀀스 레벨 정규화 적용 후 dynamics 계산.
        """
        if self.use_asl_norm:
            sequence = self._apply_asl_norm(sequence)  # (T, 90, 2)

        T = sequence.shape[0]

        # velocity: x[t+1] - x[t]
        dx = np.zeros_like(sequence)
        dx[:-1] = sequence[1:] - sequence[:-1]

        # acceleration: x[t+2] - x[t]
        dx2 = np.zeros_like(sequence)
        dx2[:-2] = sequence[2:] - sequence[:-2]

        # concatenate along last axis: (T, 90, 6) → flatten → (T, 540)
        combined = np.concatenate([sequence, dx, dx2], axis=-1)
        return combined.reshape(T, -1).astype(np.float32)
