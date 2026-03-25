# src/cnn_lstm.py
import importlib
importlib.import_module("src.tf_keras_config")
import keras
from .config import CROP_LEN, PAD, NUM_CLASSES


def get_model(dim: int, max_len=CROP_LEN, num_classes=NUM_CLASSES,
              cnn_channels=64, lstm_units=128, dropout=0.3):
    """
    수어 인식을 위한 CNN+LSTM 모델.

    구조:
      Masking
      Dense stem (dim → cnn_channels)
      Conv1D Block ×2  (local feature extraction)
      Bidirectional LSTM ×2 (temporal modeling)
      Dropout → Dense(num_classes, softmax)

    Args:
        max_len      (int)  : 입력 시퀀스 최대 길이       (기본: CROP_LEN)
        dim          (int)  : 입력 피처 차원 (UMAP 출력)  (기본: UMAP_OUTPUT_DIM)
        num_classes  (int)  : 분류 클래스 수              (기본: NUM_CLASSES)
        cnn_channels (int)  : Conv1D 출력 채널 수         (기본: 64)
        lstm_units   (int)  : LSTM 히든 유닛 수           (기본: 128)
        dropout      (float): Dropout 비율               (기본: 0.3)

    Returns:
        keras.Model  입력 shape: (batch, max_len, dim)
                     출력 shape: (batch, num_classes)
    """
    inp = keras.Input(shape=(max_len, dim))
    x = keras.layers.Masking(mask_value=PAD)(inp)

    # ----- Stem: 입력 차원 → cnn_channels (backbone.py 스타일) -----
    x = keras.layers.Dense(cnn_channels, use_bias=False, name='stem_conv')(x)
    x = keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)
    x = keras.layers.Activation('relu')(x)

    # ----- Conv Block 1 -----
    x = keras.layers.Conv1D(
        cnn_channels, kernel_size=3, padding='same', use_bias=False, name='conv1'
    )(x)
    x = keras.layers.BatchNormalization(momentum=0.95, name='bn1')(x)
    x = keras.layers.Activation('relu')(x)

    # ----- Conv Block 2 -----
    x = keras.layers.Conv1D(
        cnn_channels * 2, kernel_size=3, padding='same', use_bias=False, name='conv2'
    )(x)
    x = keras.layers.BatchNormalization(momentum=0.95, name='bn2')(x)
    x = keras.layers.Activation('relu')(x)

    # ----- LSTM (Bidirectional) -----
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=True), name='bilstm1'
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units // 2), name='bilstm2'
    )(x)

    # ----- Output -----
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32', name='classifier'
    )(x)

    return keras.Model(inp, x)
