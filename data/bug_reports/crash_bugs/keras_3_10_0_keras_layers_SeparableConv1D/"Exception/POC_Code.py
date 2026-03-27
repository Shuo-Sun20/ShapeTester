import keras
import numpy as np
import tensorflow as tf
import numpy as np
import keras
import tensorflow as tf

def call_func(
    filters,
    kernel_size,
    inputs,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer="glorot_uniform",
    pointwise_initializer="glorot_uniform",
    bias_initializer="zeros",
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None
):
    layer = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        pointwise_initializer=pointwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        pointwise_regularizer=pointwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        pointwise_constraint=pointwise_constraint,
        bias_constraint=bias_constraint
    )
    output = layer(inputs)
    return output

# Test with eager tensor
print("Testing with eager tensor:")
eager_input = tf.constant(np.random.rand(4, 10, 12), dtype=tf.float32)
try:
    eager_output = call_func(
        filters=1,
        kernel_size=5,
        inputs=eager_input,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=3,
        depth_multiplier=1,
        activation="relu",
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None
    )
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Exception with eager tensor: {e}")

# Test with Keras Input placeholder
print("\nTesting with Keras Input placeholder:")
placeholder_input = keras.Input(shape=(10, 12))
try:
    placeholder_output = call_func(
        filters=1,
        kernel_size=5,
        inputs=placeholder_input,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=3,
        depth_multiplier=1,
        activation="relu",
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None
    )
    print(f"Static output shape: {placeholder_output.shape}")
except Exception as e:
    print(f"Exception with placeholder: {e}")