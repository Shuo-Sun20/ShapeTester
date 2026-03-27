import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    return keras.ops.separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)

# Create eager tensors based on the test input
eager_inputs = tf.random.normal((1, 3, 10, 10))
eager_depthwise_kernel = tf.random.normal((3, 3, 3, 1))
eager_pointwise_kernel = tf.random.normal((1, 1, 3, 6))

# Call with eager tensors to get dynamic output shape
dynamic_result = call_func(
    inputs=eager_inputs,
    depthwise_kernel=eager_depthwise_kernel,
    pointwise_kernel=eager_pointwise_kernel,
    strides=2,
    padding="valid",
    data_format="channels_first",
    dilation_rate=3
)

# Create Keras.Input placeholders with the same shapes
placeholder_inputs = keras.Input(shape=(3, 10, 10))

# Call with placeholders to get static output shape
static_result = call_func(
    inputs=placeholder_inputs,
    depthwise_kernel=eager_depthwise_kernel,
    pointwise_kernel=eager_pointwise_kernel,
    strides=2,
    padding="valid",
    data_format="channels_first",
    dilation_rate=3
)

print(f"Dynamic output shape: {dynamic_result.shape}")
print(f"Static output shape: {static_result.shape}")
print(f"Shapes match: {dynamic_result.shape == static_result.shape}")