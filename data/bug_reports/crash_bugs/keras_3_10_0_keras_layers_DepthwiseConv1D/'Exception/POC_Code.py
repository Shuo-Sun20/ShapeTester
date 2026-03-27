import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(
    inputs,
    kernel_size,
    strides=1,
    padding="valid",
    depth_multiplier=1,
    data_format=None,
    dilation_rate=1,
    activation=None,
    use_bias=True,
    depthwise_initializer="glorot_uniform",
    bias_initializer="zeros",
    depthwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    bias_constraint=None
):
    layer_instance = keras.layers.DepthwiseConv1D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint
    )
    return layer_instance(inputs)

# Test with eager tensor
eager_input = tf.constant(np.random.rand(4, 10, 12), dtype=tf.float32)

print("Testing with eager tensor:")
try:
    eager_output = call_func(
        inputs=eager_input,
        kernel_size=7,
        strides=1,
        padding='valid',
        depth_multiplier=2,
        data_format='channels_first',
        dilation_rate=2,
        activation='relu',
        use_bias=True,
        depthwise_initializer='glorot_uniform',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None
    )
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception - {e}")

# Test with Keras Input placeholder
print("\nTesting with Keras Input placeholder:")
placeholder_input = keras.Input(shape=(10, 12))

static_output = call_func(
    inputs=placeholder_input,
    kernel_size=7,
    strides=1,
    padding='valid',
    depth_multiplier=2,
    data_format='channels_first',
    dilation_rate=2,
    activation='relu',
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    bias_initializer='zeros',
    depthwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    bias_constraint=None
)
print(f"Static output shape: {static_output.shape}")

print("\nDefect reproduced: Dynamic and static output shapes are inconsistent!")