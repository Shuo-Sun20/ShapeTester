import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

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

# Test parameters
filters = 64
kernel_size = 11
strides = 5
padding = 'valid'
data_format = 'channels_last'
dilation_rate = 1
depth_multiplier = 1
activation = 'relu'
use_bias = True

# Create test input as eager tensor
input_shape = (4, 10, 12)
eager_input = tf.constant(np.random.rand(*input_shape), dtype=tf.float32)

# Test with eager tensor
eager_output = call_func(
    filters=filters,
    kernel_size=kernel_size,
    inputs=eager_input,
    strides=strides,
    padding=padding,
    data_format=data_format,
    dilation_rate=dilation_rate,
    depth_multiplier=depth_multiplier,
    activation=activation,
    use_bias=use_bias
)

# Test with Keras Input placeholder
placeholder_input = keras.Input(shape=(10, 12))
placeholder_output = call_func(
    filters=filters,
    kernel_size=kernel_size,
    inputs=placeholder_input,
    strides=strides,
    padding=padding,
    data_format=data_format,
    dilation_rate=dilation_rate,
    depth_multiplier=depth_multiplier,
    activation=activation,
    use_bias=use_bias
)

print(f"Dynamic output shape (eager tensor): {[None if i == 0 else eager_output.shape[i] for i in range(len(eager_output.shape))]}")
print(f"Static output shape (placeholder): {[None if i == 0 else placeholder_output.shape[i] for i in range(len(placeholder_output.shape))]}")
print(f"Shapes match: {eager_output.shape[1:] == placeholder_output.shape[1:]}")