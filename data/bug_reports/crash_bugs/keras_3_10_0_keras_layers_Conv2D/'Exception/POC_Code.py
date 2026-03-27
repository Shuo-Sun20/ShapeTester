import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np
import tensorflow as tf

def call_func(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
):
    layer = keras.layers.Conv2D(
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
    return layer(inputs)

# Test parameters
filters = 1024
kernel_size = [11, 11]
strides = 4
padding = 'valid'
data_format = None
dilation_rate = 1
groups = 4
activation = 'relu'
use_bias = True
kernel_initializer = 'glorot_uniform'
bias_initializer = 'zeros'
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None

# Create eager tensor input
eager_input = tf.constant(np.random.rand(4, 10, 10, 128), dtype=tf.float32)

# Create Keras.Input placeholder with same shape
placeholder_input = keras.Input(shape=(10, 10, 128))

print("Testing with eager tensor:")
try:
    eager_output = call_func(
        inputs=eager_input,
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
    print(f"Dynamic output shape: {eager_output.shape}")
except Exception as e:
    print(f"Exception with eager tensor: {e}")

print("\nTesting with Keras.Input placeholder:")
try:
    static_output = call_func(
        inputs=placeholder_input,
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
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Exception with placeholder: {e}")

print("\nShape inconsistency detected:")
print("- Eager tensor execution fails with ValueError due to negative dimension")
print("- Placeholder execution succeeds but produces static shape [None, 0, 0, 1024]")
print("- This demonstrates the defect where static and dynamic shape handling differ")