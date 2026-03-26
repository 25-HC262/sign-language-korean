"""
ASL 사전학습 모델 기반 KSL 3단계 파인튜닝 학습 스크립트.

Stage 1 (10 epochs): stem_conv + classifier만 학습 (backbone 동결)
  - ASL backbone의 feature representation 보존
  - LR: 1e-3

Stage 2 (20 epochs): 상위 절반 backbone 해동
  - mbblock_4~6, TransformerBlock 2, top_conv 추가 학습
  - LR: 1e-4

Stage 3 (30 epochs): 전체 파인튜닝
  - 모든 레이어 학습, LR: 5e-6
  - label_smoothing=0.1 (데이터 부족 과적합 방지)

실행:
  cd sign-language-korean
  python -m train.asl_finetune_train [--use-asl-norm] [--no-wandb]
"""
import importlib
importlib.import_module("src.tf_keras_config")

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf
import keras

# ── 프로젝트 임포트 ──────────────────────────────────────────────────────────
from load_data.asl_create_dataset import ASLCompatDataLoader
from src.asl_finetune_config import (
    CHANNELS_ASL, KSL_FINETUNE_DIM, FINETUNE_CKPT_DIR, FINETUNE_MODEL_DIR,
    FINETUNE_STAGE1_LR, FINETUNE_STAGE2_LR, FINETUNE_STAGE3_LR,
    FINETUNE_STAGE1_EPOCHS, FINETUNE_STAGE2_EPOCHS, FINETUNE_STAGE3_EPOCHS,
    FINETUNE_LABEL_SMOOTHING,
)
from src.config import L_DATA, BATCH_SIZE
from src.config import NUM_CLASSES


# ── 이전 버전 Keras 호환 ──────────────────────────────────────────────────────
_orig_dense_from_config = keras.layers.Dense.from_config.__func__
@classmethod
def _compat_dense_from_config(cls, config):
    config = {k: v for k, v in config.items() if k != 'quantization_config'}
    return _orig_dense_from_config(cls, config)
keras.layers.Dense.from_config = _compat_dense_from_config


# ── Augmentation ──────────────────────────────────────────────────────────────
def random_temporal_crop(x, y, keep_ratio: float = 0.8):
    """시퀀스에서 keep_ratio 비율 프레임만 무작위 추출 후 최대 길이로 패딩."""
    seq_len = tf.shape(x)[0]
    channels = tf.shape(x)[1]
    keep = tf.cast(tf.cast(seq_len, tf.float32) * keep_ratio, tf.int32)
    keep = tf.maximum(keep, 1)

    start = tf.random.uniform((), 0, seq_len - keep + 1, dtype=tf.int32)
    cropped = x[start: start + keep, :]

    pad = tf.zeros([seq_len - keep, channels], dtype=x.dtype)
    padded = tf.concat([cropped, pad], axis=0)
    return padded, y


def random_temporal_shift(x, y, max_shift: int = 10):
    """시퀀스를 랜덤으로 시간축으로 이동."""
    shift = tf.random.uniform((), -max_shift, max_shift + 1, dtype=tf.int32)
    seq_len = tf.shape(x)[0]
    shift = tf.clip_by_value(shift, -seq_len + 1, seq_len - 1)

    if shift > 0:
        pad = tf.zeros([shift, tf.shape(x)[1]], dtype=x.dtype)
        x = tf.concat([pad, x[:-shift]], axis=0)
    elif shift < 0:
        pad = tf.zeros([-shift, tf.shape(x)[1]], dtype=x.dtype)
        x = tf.concat([x[-shift:], pad], axis=0)
    return x, y


def apply_augmentation(ds: tf.data.Dataset) -> tf.data.Dataset:
    ds = ds.map(random_temporal_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(random_temporal_shift, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


# ── 학습 유틸리티 ──────────────────────────────────────────────────────────────
def finalize_dataset(ds: tf.data.Dataset, batch_size: int, augment: bool = False) -> tf.data.Dataset:
    if augment:
        ds = apply_augmentation(ds)
    return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


def freeze_layers(model: keras.Model, frozen_names: list):
    """레이어 이름에 해당 문자열이 포함된 경우 동결."""
    for layer in model.layers:
        layer.trainable = not any(name in layer.name for name in frozen_names)


def unfreeze_all(model: keras.Model):
    for layer in model.layers:
        layer.trainable = True


def get_callbacks(ckpt_path: str, monitor: str = 'val_loss', min_lr: float = 1e-8):
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1,
        ),
    ]


# ── Stage 학습 함수 ─────────────────────────────────────────────────────────────
def run_stage(
    stage: int,
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    lr: float,
    epochs: int,
    batch_size: int,
    label_smoothing: float = 0.0,
    augment: bool = False,
    use_wandb: bool = True,
) -> keras.callbacks.History:
    print(f"\n{'=' * 60}")
    print(f"  Stage {stage} 시작 | LR={lr} | Epochs={epochs}")
    print(f"{'=' * 60}")

    # 학습 가능 레이어 출력
    trainable = [l.name for l in model.layers if l.trainable and l.weights]
    frozen   = [l.name for l in model.layers if not l.trainable and l.weights]
    print(f"  학습 레이어 ({len(trainable)}): {trainable}")
    print(f"  동결 레이어 ({len(frozen)}): {frozen[:5]}{'...' if len(frozen) > 5 else ''}")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    train_batched = finalize_dataset(train_ds, batch_size, augment=augment)
    val_batched   = finalize_dataset(val_ds,   batch_size, augment=False)

    ckpt_path = str(Path(FINETUNE_CKPT_DIR) / f"ksl_asl_stage{stage}.keras")
    callbacks = get_callbacks(ckpt_path, min_lr=lr * 0.01)

    if use_wandb:
        try:
            from wandb.integration.keras import WandbMetricsLogger
            callbacks.append(WandbMetricsLogger())
        except ImportError:
            pass

    history = model.fit(
        train_batched,
        validation_data=val_batched,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n  Stage {stage} 완료. 최고 val_loss: {min(history.history.get('val_loss', [float('inf')])):.4f}")
    return history


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-asl-norm', dest='use_asl_norm', action='store_true', default=True,
                        help='ASL 스타일 시퀀스 정규화 사용 (기본: True)')
    parser.add_argument('--no-asl-norm', dest='use_asl_norm', action='store_false')
    parser.add_argument('--no-wandb', action='store_true', default=False)
    parser.add_argument('--init-ckpt', default=None,
                        help='가중치 전이 완료 모델 경로 (없으면 transfer_weights.py 먼저 실행)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    keras.mixed_precision.set_global_policy("mixed_float16")

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="ksl-asl-finetune", config={
                "channels": CHANNELS_ASL,
                "latent_dim": KSL_FINETUNE_DIM,
                "num_classes": NUM_CLASSES,
                "use_asl_norm": args.use_asl_norm,
            })
        except Exception as e:
            print(f"[!] WandB 초기화 실패: {e}. WandB 없이 진행합니다.")
            use_wandb = False

    # ── 데이터 로드 ────────────────────────────────────────────────────────────
    print("\n[*] 데이터 로딩 중...")
    loader = ASLCompatDataLoader(
        data_path=L_DATA,
        use_asl_norm=args.use_asl_norm,
    )
    train_ds, val_ds = loader.create_gm_train_dataset()

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    init_ckpt = args.init_ckpt
    if init_ckpt is None:
        default_path = str(Path(FINETUNE_CKPT_DIR) / "ksl_asl_transfer_init.keras")
        if Path(default_path).exists():
            init_ckpt = default_path
            print(f"[*] 전이 초기화 모델 자동 탐지: {init_ckpt}")
        else:
            print("[!] 전이 초기화 모델이 없습니다. 먼저 실행하세요:")
            print("    python -m train.transfer_weights")
            sys.exit(1)

    print(f"\n[*] 모델 로딩: {init_ckpt}")
    from src.backbones.transformer_backbone import ECA, LateDropout, CausalDWConv1D, MultiHeadSelfAttention
    custom_objects = {
        'ECA': ECA, 'LateDropout': LateDropout,
        'CausalDWConv1D': CausalDWConv1D,
        'MultiHeadSelfAttention': MultiHeadSelfAttention,
    }
    model = keras.models.load_model(init_ckpt, custom_objects=custom_objects, compile=False)
    print(f"[*] 모델 로드 완료. 입력: {model.input_shape}")

    # ══════════════════════════════════════════════════════════════
    # Stage 1: stem_conv + classifier만 학습 (backbone 완전 동결)
    # ══════════════════════════════════════════════════════════════
    freeze_layers(model, frozen_names=[
        'stem_bn',
        'mbblock_1', 'mbblock_2', 'mbblock_3',
        'mbblock_4', 'mbblock_5', 'mbblock_6',
        'batch_normalization', 'multi_head_self_attention', 'dense',  # TransformerBlock 내 레이어
        'top_conv',
    ])
    # stem_conv와 classifier는 명시적으로 학습 가능하게 설정
    for layer in model.layers:
        if layer.name in ('stem_conv', 'classifier'):
            layer.trainable = True

    run_stage(
        stage=1, model=model,
        train_ds=train_ds, val_ds=val_ds,
        lr=FINETUNE_STAGE1_LR,
        epochs=FINETUNE_STAGE1_EPOCHS,
        batch_size=args.batch_size,
        label_smoothing=0.0,
        augment=False,
        use_wandb=use_wandb,
    )

    # ══════════════════════════════════════════════════════════════
    # Stage 2: 상위 절반 backbone 해동
    # ══════════════════════════════════════════════════════════════
    stage2_ckpt = str(Path(FINETUNE_CKPT_DIR) / "ksl_asl_stage1.keras")
    if Path(stage2_ckpt).exists():
        model = keras.models.load_model(stage2_ckpt, custom_objects=custom_objects, compile=False)

    # mbblock_4~6, top_conv 해동; mbblock_1~3, TransformerBlock 1 계속 동결
    freeze_layers(model, frozen_names=[
        'stem_bn',
        'mbblock_1', 'mbblock_2', 'mbblock_3',
    ])
    # 해동할 레이어 명시적 설정
    unfreeze_targets = ['stem_conv', 'mbblock_4', 'mbblock_5', 'mbblock_6', 'top_conv', 'classifier']
    for layer in model.layers:
        if any(t in layer.name for t in unfreeze_targets):
            layer.trainable = True

    run_stage(
        stage=2, model=model,
        train_ds=train_ds, val_ds=val_ds,
        lr=FINETUNE_STAGE2_LR,
        epochs=FINETUNE_STAGE2_EPOCHS,
        batch_size=args.batch_size,
        label_smoothing=0.0,
        augment=True,
        use_wandb=use_wandb,
    )

    # ══════════════════════════════════════════════════════════════
    # Stage 3: 전체 파인튜닝
    # ══════════════════════════════════════════════════════════════
    stage3_ckpt = str(Path(FINETUNE_CKPT_DIR) / "ksl_asl_stage2.keras")
    if Path(stage3_ckpt).exists():
        model = keras.models.load_model(stage3_ckpt, custom_objects=custom_objects, compile=False)

    unfreeze_all(model)

    run_stage(
        stage=3, model=model,
        train_ds=train_ds, val_ds=val_ds,
        lr=FINETUNE_STAGE3_LR,
        epochs=FINETUNE_STAGE3_EPOCHS,
        batch_size=args.batch_size,
        label_smoothing=FINETUNE_LABEL_SMOOTHING,
        augment=True,
        use_wandb=use_wandb,
    )

    # ── 최종 모델 저장 ──────────────────────────────────────────────────────────
    final_path = str(Path(FINETUNE_MODEL_DIR) / "ksl_asl_finetuned_final.keras")
    model.save(final_path)
    print(f"\n[*] 최종 모델 저장 완료: {final_path}")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
