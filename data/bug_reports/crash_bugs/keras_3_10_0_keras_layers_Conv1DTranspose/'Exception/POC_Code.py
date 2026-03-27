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
    output_padding=None,
    data_format=None,
    dilation_rate=1,
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
    layer = keras.layers.Conv1DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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

# Test parameters from the defect report
filters = 16
kernel_size = 3
strides = 2
padding = 'valid'
output_padding = 3
data_format = 'channels_last'
dilation_rate = 1
activation = 'relu'
use_bias = True
kernel_initializer = 'glorot_uniform'
bias_initializer = 'zeros'
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None

# Test with Keras.Input placeholders (static shape)
print("Testing with Keras.Input placeholders (static shape):")
input_placeholder = keras.Input(shape=(10, 128))
try:
    static_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        inputs=input_placeholder,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
    print(f"Static shape test failed with exception: {e}")

# Test with eager tensors (dynamic shape)
print("\nTesting with eager tensors (dynamic shape):")
eager_tensor = tf.constant(np.random.rand(4, 10, 128).astype(np.float32))
try:
    dynamic_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        inputs=eager_tensor,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic shape test failed with exception: {e}")

print("\nDefect reproduction: The static shape test should succeed while the dynamic shape test should fail with an exception.")