"""
ISLR (Isolated Sign Language Recognition) 백본.
sign-language/src/backbone.py + src/utils.py 통합 포팅.

클래스명(ECA, LateDropout, CausalDWConv1D, MultiHeadSelfAttention)은
.h5 weights 로딩 시 layer 이름 매칭에 필요하므로 원본 그대로 유지.
"""
import tensorflow as tf
import numpy as np

# ── 설정 상수 ────────────────────────────────────────────────────────────────
ISLR_MAX_LEN = 384
ISLR_NUM_CLASSES = 250
ISLR_ROWS_PER_FRAME = 543  # face(468) + lhand(21) + pose(33) + rhand(21)

ISLR_NOSE = [1, 2, 98, 327]
ISLR_LIP = [
    0,
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
ISLR_REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
             246, 161, 160, 159, 158, 157, 173]
ISLR_LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362,
             466, 388, 387, 386, 385, 384, 398]
ISLR_LHAND = list(range(468, 489))
ISLR_RHAND = list(range(522, 543))

ISLR_POINT_LANDMARKS = ISLR_LIP + ISLR_LHAND + ISLR_RHAND + ISLR_NOSE + ISLR_REYE + ISLR_LEYE
ISLR_NUM_NODES = len(ISLR_POINT_LANDMARKS)
ISLR_CHANNELS = 6 * ISLR_NUM_NODES  # x, y + dx + dx2, per point


# ── Preprocessing ────────────────────────────────────────────────────────────

def _tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims
    ) / tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims
    )


def _tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = _tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(_tf_nan_mean(d * d, axis=axis, keepdims=keepdims))


class ISLRPreprocess(tf.keras.layers.Layer):
    """입력 (T, 543, 3) → 정규화된 (1, MAX_LEN, CHANNELS) 변환."""

    def __init__(self, max_len=ISLR_MAX_LEN, point_landmarks=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks if point_landmarks is not None else ISLR_POINT_LANDMARKS

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]   # (1, T, 543, 3)
        else:
            x = inputs              # (B, T, 543, 3)

        # 랜드마크 17(입술 중앙)을 기준으로 정규화
        mean = _tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)

        x = tf.gather(x, self.point_landmarks, axis=2)   # (B, T, NUM_NODES, 3)
        std = _tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]   # x, y only

        dx = tf.cond(
            tf.shape(x)[1] > 1,
            lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x)
        )
        dx2 = tf.cond(
            tf.shape(x)[1] > 2,
            lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x)
        )

        x = tf.concat([
            tf.reshape(x,   (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx,  (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)

        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        return x


# ── Backbone layers (클래스명 원본 유지 — weights 로딩 호환) ──────────────────

class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1,
                                           padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: inputs,
            lambda: self.dropout(inputs, training=training)
        )
        if training:
            self._train_counter.assign_add(1)
        return x


class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False,
                 depthwise_initializer='glorot_uniform', name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D(
            (dilation_rate * (kernel_size - 1), 0), name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size, strides=1, dilation_rate=dilation_rate,
            padding='valid', use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            name=name + '_dwconv')
        self.supports_masking = True

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :]
        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(
            tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def _Conv1DBlock(channel_size, kernel_size, dilation_rate=1, drop_rate=0.0,
                 expand_ratio=2, activation='swish', name=None):
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))

    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio
        skip = inputs
        x = tf.keras.layers.Dense(channels_expand, use_bias=True, activation=activation,
                                  name=name + '_expand_conv')(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate,
                           use_bias=False, name=name + '_dwconv')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)
        x = ECA()(x)
        x = tf.keras.layers.Dense(channel_size, use_bias=True,
                                  name=name + '_project_conv')(x)
        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1),
                                        name=name + '_drop')(x)
        if channels_in == channel_size:
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply


def _TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2,
                      drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x

    return apply


class ISLRTFLiteModel(tf.Module):
    """추론 래퍼: ISLRPreprocess → backbone → 클래스 확률."""

    def __init__(self, islr_models):
        super().__init__()
        self.prep_inputs = ISLRPreprocess()
        self.islr_models = islr_models

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}


def get_islr_model(max_len=ISLR_MAX_LEN, dropout_step=0, dim=192):
    """
    ISLR 백본 모델 구성. get_model()과 동일한 구조로 weights 로딩 호환.
    sign-language/src/backbone.py:get_model() 참조.
    """
    inp = tf.keras.Input((max_len, ISLR_CHANNELS))
    x = inp
    ksize = 17

    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    x = _Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = _Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = _Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = _TransformerBlock(dim, expand=2)(x)

    x = _Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = _Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = _Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = _TransformerBlock(dim, expand=2)(x)

    x = tf.keras.layers.Dense(dim * 2, activation=None, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(ISLR_NUM_CLASSES, name='classifier', activation='softmax')(x)

    return tf.keras.Model(inp, x)
