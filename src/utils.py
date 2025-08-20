# src/utils.py
from .config import MAX_LEN, POINT_LANDMARKS
import tensorflow as tf

def tf_nan_mean(x, axis=0, keepdims=False):
    """
    Computes the mean of the input tensor while ignoring NaN values.
    """
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    """
    Computes the standard deviation of the input tensor while ignoring NaN values.
    """
    if center is None:
        center = tf_nan_mean(x, axis=axis,  keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    """
    Preprocessing layer for OpenPose input data.
    """
    
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        """
        Preprocesses the OpenPose input data.
        
        Args:
            inputs: Input tensor [batch, time, 137, 3]
            
        Returns:
            Preprocessed tensor [batch, time, channels]
        """
        if tf.rank(inputs) == 3:
            x = inputs[None,...]
        else:
            x = inputs
        
        # Use neck (index 1) as reference point for OpenPose
        # Extract neck coordinates for normalization
        mean = tf_nan_mean(tf.gather(x, [1], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        
        # Extract selected keypoints from the 137 available
        # Ensure all indices are valid
        valid_landmarks = [idx for idx in self.point_landmarks if idx < 137]
        if len(valid_landmarks) != len(self.point_landmarks):
            tf.print(f"Warning: Some landmarks indices are >= 137. Using {len(valid_landmarks)} valid landmarks.")
        
        x = tf.gather(x, valid_landmarks, axis=2)  # N,T,P,C
        
        # Calculate standard deviation for normalization
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        std = tf.where(tf.equal(std, 0), tf.ones_like(std), std)  # Avoid division by zero
        
        # Normalize
        x = (x - mean) / std
        
        # Crop to max length if needed
        if self.max_len is not None:
            x = x[:, :self.max_len]
        
        # Get the actual sequence length
        length = tf.shape(x)[1]
        num_landmarks = len(valid_landmarks)
        
        # Use only x, y coordinates (OpenPose z is confidence, not depth)
        x = x[..., :2]
        
        # Calculate derivatives
        '''dx = tf.cond(
            tf.greater(tf.shape(x)[1], 1),
            lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0,0],[0,1],[0,0],[0,0]]),
            lambda: tf.zeros_like(x)
        )'''
        
        '''dx2 = tf.cond(
            tf.greater(tf.shape(x)[1], 2),
            lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0,0],[0,2],[0,0],[0,0]]),
            lambda: tf.zeros_like(x)
        )'''
        
        # Concatenate features: original coordinates + first derivative + second derivative
        # Shape: [batch, time, num_landmarks, 2] -> [batch, time, num_landmarks * 2 * 3]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        
        x_flat = tf.reshape(x, (batch_size, time_steps, num_landmarks * 2))
        #dx_flat = tf.reshape(dx, (batch_size, time_steps, num_landmarks * 2))
        #dx2_flat = tf.reshape(dx2, (batch_size, time_steps, num_landmarks * 2))
        
        # Concatenate all features
        #x = tf.concat([x_flat, dx_flat, dx2_flat], axis=-1)
        
        # Replace NaN with 0
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "point_landmarks": self.point_landmarks
        })
        return config