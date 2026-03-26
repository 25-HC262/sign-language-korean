"""
ASL 파인튜닝용 모델 빌더.
transformer_backbone.py의 모든 레이어 클래스를 재사용하되,
latent_dim=192(ASL 백본과 동일)로 고정하여 가중치 전이가 가능하게 합니다.

아키텍처:
  Input (batch, MAX_LEN, 540)
  Dense(540→192, name='stem_conv')       ← 랜덤 초기화 (input shape 불일치)
  BatchNorm(name='stem_bn')              ← ASL 가중치 전이 ✅
  Conv1DBlock × 3 + TransformerBlock     ← ASL 가중치 전이 ✅ (block 1)
  Conv1DBlock × 3 + TransformerBlock     ← ASL 가중치 전이 ✅ (block 2)
  Dense(192→384, name='top_conv')        ← ASL 가중치 전이 ✅
  GlobalAveragePooling1D
  LateDropout(0.8)
  Dense(384→33, name='classifier')       ← 랜덤 초기화 (클래스 수 불일치)
"""
import importlib
importlib.import_module("src.tf_keras_config")

import itertools
import keras

# transformer_backbone.py의 모든 레이어 클래스 재사용
from src.backbones.transformer_backbone import (
    Conv1DBlock, TransformerBlock, LateDropout,
    ECA, CausalDWConv1D, MultiHeadSelfAttention,
)
from src.asl_finetune_config import CHANNELS_ASL, KSL_FINETUNE_DIM, MAX_LEN
from src.config import NUM_CLASSES

# Conv1DBlock 이름 카운터를 파인튜닝 모델 전용으로 별도 관리
_FINETUNE_MBBLOCK_COUNTER = itertools.count(1)


def get_asl_finetuned_model(
    max_len: int = MAX_LEN,
    num_classes: int = NUM_CLASSES,
    input_channels: int = CHANNELS_ASL,  # 540
    latent_dim: int = KSL_FINETUNE_DIM,  # 192
    dropout_step: int = 0,
    training: bool = True,
) -> keras.Model:
    """
    ASL 백본 아키텍처와 동일한 구조의 KSL 파인튜닝 모델을 생성합니다.

    Conv1DBlock 레이어 이름이 'mbblock_1' ~ 'mbblock_6'이 되도록
    명시적 name 인자를 전달합니다. (ASL 가중치 전이 시 이름 매칭용)

    Args:
        max_len: 입력 시퀀스 최대 길이
        num_classes: 출력 클래스 수 (KSL = 33)
        input_channels: 입력 채널 수 (540 = 90 landmarks × 2D × 3)
        latent_dim: 내부 hidden dim (192 = ASL 백본과 동일)
        dropout_step: LateDropout 활성화 시작 스텝
        training: True이면 Masking 레이어 활성화
    """
    ksize = 17
    num_heads = 4

    inp = keras.Input(shape=(max_len, input_channels), name='input')
    x = keras.layers.Masking(mask_value=-100.0)(inp) if training else inp

    # --- Stem (ASL: Dense 1116→192, KSL: Dense 540→192 — shape 불일치로 랜덤 초기화) ---
    x = keras.layers.Dense(latent_dim, use_bias=False, name='stem_conv')(x)
    x = keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    # --- Block 1: Conv1DBlock × 3 + TransformerBlock ---
    # 명시적 name으로 ASL 레이어 이름(mbblock_1~3)과 일치시킴
    x = Conv1DBlock(latent_dim, ksize, drop_rate=0.2, name='mbblock_1')(x)
    x = Conv1DBlock(latent_dim, ksize, drop_rate=0.2, name='mbblock_2')(x)
    x = Conv1DBlock(latent_dim, ksize, drop_rate=0.2, name='mbblock_3')(x)
    x = TransformerBlock(latent_dim, expand=2, num_heads=num_heads)(x)

    # --- Block 2: Conv1DBlock × 3 + TransformerBlock ---
    x = Conv1DBlock(latent_dim, ksize, drop_rate=0.2, name='mbblock_4')(x)
    x = Conv1DBlock(latent_dim, ksize, drop_rate=0.2, name='mbblock_5')(x)
    x = Conv1DBlock(latent_dim, ksize, drop_rate=0.2, name='mbblock_6')(x)
    x = TransformerBlock(latent_dim, expand=2, num_heads=num_heads)(x)

    # --- Top (ASL와 동일한 name, 동일한 shape → 전이 가능) ---
    x = keras.layers.Dense(latent_dim * 2, activation=None, name='top_conv')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    # classifier: shape (384, 33) ≠ ASL (384, 250) → 랜덤 초기화
    x = keras.layers.Dense(num_classes, activation='softmax', name='classifier', dtype='float32')(x)

    return keras.Model(inp, x, name='ksl_asl_finetune')
