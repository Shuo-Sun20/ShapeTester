import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    return keras.ops.separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)

# Test input parameters
inputs_shape = (2, 5, 5, 3)
depthwise_kernel_shape = (3, 3, 3, 1)
pointwise_kernel_shape = (1, 1, 3, 4)
strides = 1
padding = "valid"
data_format = None
dilation_rate = 3

# Create eager tensors (numpy arrays)
inputs_eager = np.random.random(inputs_shape).astype(np.float32)
depthwise_kernel_eager = np.random.random(depthwise_kernel_shape).astype(np.float32)
pointwise_kernel_eager = np.random.random(pointwise_kernel_shape).astype(np.float32)

# Create Keras Input placeholders
inputs_placeholder = keras.Input(shape=inputs_shape[1:])
depthwise_kernel_placeholder = keras.Input(shape=depthwise_kernel_shape)
pointwise_kernel_placeholder = keras.Input(shape=pointwise_kernel_shape)

print("Testing with eager tensors:")
try:
    result_eager = call_func(inputs_eager, depthwise_kernel_eager, pointwise_kernel_eager, 
                           strides, padding, data_format, dilation_rate)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Error with eager tensors: {e}")

print("\nTesting with Keras Input placeholders:")
try:
    result_placeholder = call_func(inputs_placeholder, depthwise_kernel_placeholder, pointwise_kernel_placeholder,
                                 strides, padding, data_format, dilation_rate)
    print(f"Static output shape: {result_placeholder.shape}")
except Exception as e:
    print(f"Error with placeholders: {e}")