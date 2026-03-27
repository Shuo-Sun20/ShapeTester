import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    inputs=None
):
    layer = keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint
    )
    output = layer(inputs)
    return output

# Test parameters from the defect report
filters = 256
kernel_size = 11
strides = 5
padding = 'valid'
data_format = None
dilation_rate = 1
groups = 1
activation = 'relu'
use_bias = True
kernel_initializer = 'glorot_uniform'
bias_initializer = 'zeros'
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None

# Create test input as eager tensor
eager_input = tf.constant(np.random.rand(4, 10, 128), dtype=tf.float32)

# Create Keras Input placeholder with same shape
placeholder_input = keras.Input(shape=(10, 128))

print("Testing with eager tensor:")
try:
    eager_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        inputs=eager_input
    )
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception - {e}")

print("\nTesting with Keras Input placeholder:")
try:
    placeholder_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        inputs=placeholder_input
    )
    print(f"Static output shape: {placeholder_output.shape}")
except Exception as e:
    print(f"Static output shape: Exception - {e}")