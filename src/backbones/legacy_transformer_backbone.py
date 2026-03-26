import importlib
importlib.import_module("src.tf_keras_config")

import itertools

import keras
keras.mixed_precision.set_global_policy("mixed_float16") # fp16 가속 keras3 버전, 모델 구성 & 학습 시에만 사용
import tensorflow as tf

from src.asl_finetune_config import MAX_LEN, CHANNELS_ASL, NUM_CLASSES_ASL
from src.utils import Preprocess

MBBLOCK_COUNTER = itertools.count(1)

# @keras.saving.register_keras_serializable()
class ECA(keras.layers.Layer):
    """
    Efficient Channel Attention layer.

    Args:
        kernel_size (int): Size of the kernel for the convolutional layer.

    Returns:
        Output tensor after applying the efficient channel attention mechanism.
    """

    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        """
        Applies the efficient channel attention mechanism to the input tensor.

        Args:
            inputs: Input tensor.
            mask: Mask tensor for masking specific values in the input.

        Returns:
            Output tensor after applying the efficient channel attention mechanism.
        """
        nn = keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)       # (Batch, MAX_SEQ, OUTPUT_DIM) -> (Batch, OUTPUT_DIM)
        nn = tf.expand_dims(nn, -1)        # (Batch, OUTPUT_DIM) -> (Batch, OUTPUT_DIM, 1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        nn = tf.cast(nn, inputs.dtype)
        return inputs * nn

class LateDropout(keras.layers.Layer):
    """
    Layer that applies dropout after a certain training step.

    Args:
        rate (float): Dropout rate.
        noise_shape: Shape of the binary dropout mask.
        start_step (int): The training step after which the dropout is applied.

    Returns:
        Output tensor after applying dropout.
    """
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.noise_shape = noise_shape
        self.dropout = keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        """
        Applies dropout to the input tensor.

        Args:
            inputs: Input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            Output tensor after applying dropout.
        """
        # dtype = inputs.dtype
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: inputs,
            lambda: tf.cast(self.dropout(inputs, training=training), inputs.dtype)
        )
        if training:
            self._train_counter.assign_add(1)
        return x # ops.cast(x, dtype)                                                        # inputs의 dtype으로 casting - fp16 가속 설정

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         "rate": self.rate,
    #         "start_step": self.start_step,
    #         "noise_shape": self.noise_shape
    #     })
    #     return config

class CausalDWConv1D(keras.layers.Layer):
    """
    Causal Dilated Depthwise Convolutional 1D layer.

    Args:
        kernel_size (int): Size of the kernel for the convolutional layer.
        dilation_rate (int): Dilation rate for the convolutional layer.
        use_bias (bool): Whether to use bias in the convolutional layer.
        depthwise_initializer: Initializer for the depthwise convolutional kernel.
        name (str): Name of the layer.

    Returns:
        Output tensor after applying the causal dilated depthwise convolution.
    """

    def __init__(self,
                 kernel_size=17,
                 dilation_rate=1,
                 use_bias=False,
                 depthwise_initializer='glorot_uniform',
                 name='', **kwargs):
        self.name = name
        super().__init__(name=self.name,**kwargs)
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer

        self.causal_pad = keras.layers.ZeroPadding1D((self.dilation_rate*(self.kernel_size-1),0),name=name + '_pad')
        self.dw_conv = keras.layers.DepthwiseConv1D(
            self.kernel_size,
            strides=1,
            dilation_rate=self.dilation_rate,
            padding='valid',
            use_bias=self.use_bias,
            depthwise_initializer=self.depthwise_initializer,
            name=self.name + '_dwconv')
        self.supports_masking = True

    def call(self, inputs):
        """
        Applies the causal dilated depthwise convolution to the input tensor.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor after applying the causal dilated depthwise convolution.
        """
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x # ops.cast(x, inputs.dtype) # inputs의 dtype으로 casting - fp16 가속 설정

def Conv1DBlock(channel_size,
                kernel_size,
                dilation_rate=1,
                drop_rate=0.0,
                expand_ratio=2,
                se_ratio=0.25,
                activation='swish',
                name=None):
    """
    Efficient Conv1D block, @hoyso48

    Args:
        channel_size (int): Number of output channels for the block.
        kernel_size (int): Size of the kernel for the convolutional layers.
        dilation_rate (int): Dilation rate for the convolutional layers.
        drop_rate (float): Dropout rate.
        expand_ratio (int): Expansion ratio for the Dense layer.
        se_ratio (float): Squeeze-and-Excitation ratio.
        activation (str): Activation function.
        name (str): Name of the block.

    Returns:
        Function to apply the Conv1D block to an input tensor.
    """

    if name is None:
        uid = next(MBBLOCK_COUNTER)
        name = f"mbblock_{uid}" # str(keras.backend.get_uid("mbblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = keras.ops.shape(inputs)[-1] # keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
                           dilation_rate=dilation_rate,
                           use_bias=False,
                           name=name + '_dwconv')(x)

        x = keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if channels_in == channel_size:
            x = keras.layers.Add(name=name + '_add')([x, skip])
        return x

    return apply

class MultiHeadSelfAttention(keras.layers.Layer):
    """
    Multi-Head Self-Attention layer.

    Args:
        dim (int): Dimension of the attention vectors.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.

    Returns:
        Output tensor after applying the multi-head self-attention mechanism.
    """
    def __init__(self, dim=256, num_heads=4, drop_rate=0.0, **kwargs):
        # 이전 버전 모델 호환: 'dropout' 키로 저장된 경우 처리
        dropout = kwargs.pop('dropout', None)
        if dropout is not None:
            drop_rate = dropout
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.drop_rate = drop_rate

        self.qkv = keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = keras.layers.Dropout(self.drop_rate)
        self.proj = keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        """
        Applies the multi-head self-attention mechanism to the input tensor.

        Args:
            inputs: Input tensor.
            mask: Mask tensor indicating valid positions.

        Returns:
            Output tensor after applying the multi-head self-attention mechanism.
        """
        # dtype = inputs.dtype # 입력 타입 고정

        qkv = self.qkv(inputs)
        qkv = keras.layers.Permute((2, 1, 3))(keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        # attn = tf.matmul(ops.cast(q, dtype), ops.cast(k, dtype), transpose_b=True) * ops.cast(self.scale, dtype)    # inputs의 dtype으로 casting - fp16 가속 설정
        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            # mask = ops.cast(mask[:, None, None, :], dtype) # inputs의 dtype으로 casting - fp16 가속 설정
            mask = mask[:, None, None, :]
            # attn = attn + (1.0 - mask) * -1e4

        attn = keras.layers.Softmax(axis=-1)(attn, mask=mask)
        # attn = ops.cast(self.drop1(attn), dtype)        # inputs의 dtype으로 casting - fp16 가속 설정
        attn = self.drop1(attn)

        # x = attn @ ops.cast(v, dtype)                   # inputs의 dtype으로 casting - fp16 가속 설정
        x = attn @ v
        x = keras.layers.Reshape((-1, self.dim))(keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x # ops.cast(x, dtype)                       # inputs의 dtype으로 casting - fp16 가속 설정

    # def build(self, input_shape):
    #     self.qkv.build(input_shape)
    #     self.drop1.build((None, self.num_heads, input_shape[1], input_shape[1]))
    #     self.proj.build((None, input_shape[1], self.dim))
    #     super().build(input_shape)
    #
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         "dim": self.dim,
    #         "num_heads": self.num_heads,
    #         "drop_rate": self.drop_rate
    #     })
    #     return config

def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    """
    Transformer Block.

    Args:
        dim (int): Dimension of the attention vectors.
        num_heads (int): Number of attention heads.
        expand (int): Expansion ratio for the Dense layer.
        attn_dropout (float): Dropout rate for attention mechanism.
        drop_rate (float): Dropout rate.
        activation (str): Activation function.

    Returns:
        Function to apply the Transformer Block to an input tensor.
    """
    def apply(inputs):
        x = inputs
        x = keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, drop_rate=attn_dropout)(x)
        x = keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = keras.layers.Add()([inputs, x])
        attn_out = x

        x = keras.layers.BatchNormalization(momentum=0.95)(x)
        x = keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = keras.layers.Dense(dim, use_bias=False)(x)
        x = keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = keras.layers.Add()([attn_out, x])
        return x
    return apply

class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – A Preprocessing Model
        – The ISLR model
    """

    def __init__(self, islr_models):
        """
        Initializes the TFLiteModel with the specified model.
        Args:
            islr_model: A single Keras model (not a list)
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main model
        self.prep_inputs = Preprocess()
        self.islr_models = islr_models  # Single model, not a list

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 137, 3] for OpenPose.

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}

def get_model(max_len=MAX_LEN, dropout_step=0, dim=192):
    """
    Creates a model for sequence classification using a combination of convolutional layers and transformer blocks.

    Args:
        max_len (int): Maximum length of the input sequence.
        dropout_step (int): Dropout step for the LateDropout layer.
        dim (int): Dimension of the hidden representations.

    Returns:
        A TensorFlow Keras Model object.
    """
    inp = keras.Input(shape=(max_len, CHANNELS_ASL))
    # x = keras.layers.Masking(mask_value=PAD)(inp) if training else inp # 학습 시 masking 활성화
    x = inp
    ksize = 17

    # Stem layers
    x = keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
    x = keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    # Convolutional and Transformer blocks
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    # Additional convolutional blocks and transformer blocks for larger models
    if dim == 384: #for the 4x sized model
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

    # Top layers
    x = keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = keras.layers.Dense(NUM_CLASSES_ASL, name='classifier', activation='softmax', dtype='float32')(x) # fp16 가속 설정을 위한 데이터 누락 방지
    return keras.Model(inp, x)