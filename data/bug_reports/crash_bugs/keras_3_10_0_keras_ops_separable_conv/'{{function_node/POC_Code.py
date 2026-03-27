import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    return keras.ops.separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)

# Test with eager tensors (dynamic)
print("Testing with eager tensors:")
try:
    inputs_eager = np.random.rand(2, 5, 5, 3).astype(np.float32)
    depthwise_kernel_eager = np.random.rand(3, 3, 3, 1).astype(np.float32)
    pointwise_kernel_eager = np.random.rand(1, 1, 3, 4).astype(np.float32)
    
    result_eager = call_func(inputs_eager, depthwise_kernel_eager, pointwise_kernel_eager, 
                           strides=1, padding='same', data_format='channels_first', dilation_rate=2)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test with Keras.Input placeholders (static)
print("\nTesting with Keras.Input placeholders:")
try:
    inputs_placeholder = keras.Input(shape=(5, 5, 3))
    depthwise_kernel_placeholder = keras.Input(shape=(3, 3, 3, 1))
    pointwise_kernel_placeholder = keras.Input(shape=(1, 1, 3, 4))
    
    result_static = call_func(inputs_placeholder, depthwise_kernel_placeholder, pointwise_kernel_placeholder,
                            strides=1, padding='same', data_format='channels_first', dilation_rate=2)
    print(f"Static output shape: {result_static.shape}")
except Exception as e:
    print(f"Static output shape: {e}")