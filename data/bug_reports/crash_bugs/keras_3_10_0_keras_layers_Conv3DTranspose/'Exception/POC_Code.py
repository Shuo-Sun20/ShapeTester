import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding="valid",
    data_format=None,
    output_padding=None,
    dilation_rate=(1, 1, 1),
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
    layer = keras.layers.Conv3DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        output_padding=output_padding,
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
kernel_size = 2
strides = 1
padding = 'valid'
data_format = 'channels_first'
output_padding = [2, 2, 2]
dilation_rate = [2, 1, 2]
activation = 'relu'
use_bias = True
kernel_initializer = 'glorot_uniform'
bias_initializer = 'zeros'
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None

# Create input shapes
input_shape = (4, 10, 8, 12, 128)

print("Testing Conv3DTranspose with eager tensors vs Keras.Input placeholders")
print(f"Input shape: {input_shape}")
print(f"Parameters: filters={filters}, kernel_size={kernel_size}, strides={strides}")
print(f"padding={padding}, data_format={data_format}, output_padding={output_padding}")
print(f"dilation_rate={dilation_rate}, activation={activation}")
print()

# Test 1: With Keras.Input placeholder to get static output shape
print("=== Test 1: Using Keras.Input placeholder ===")
try:
    placeholder_input = keras.Input(shape=input_shape[1:], batch_size=None)
    static_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
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
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static test failed with exception: {e}")

print()

# Test 2: With eager tensor to get dynamic output shape
print("=== Test 2: Using eager tensor ===")
try:
    # Create eager tensor
    eager_input = tf.random.normal(input_shape, dtype=tf.float32)
    dynamic_output = call_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
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
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic test failed with exception: {e}")
    print(f"Exception type: {type(e).__name__}")

print()
print("=== Defect Analysis ===")
print("The defect manifests as inconsistent behavior between:")
print("1. Static shape inference using Keras.Input placeholders")
print("2. Dynamic execution using eager tensors")
print("Expected: Both should produce the same output shape")
print("Actual: Dynamic execution fails with MklNativeConv3DBackpropInputV2 error")