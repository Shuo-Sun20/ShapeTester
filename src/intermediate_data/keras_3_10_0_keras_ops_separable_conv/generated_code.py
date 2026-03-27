import keras
import numpy as np

def call_func(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    return keras.ops.separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides, padding, data_format, dilation_rate)

# 2D separable convolution example
batch_size = 2
height, width = 5, 5
num_channels = 3
depth_multiplier = 1
num_output_channels = 4

# Generate random tensors
np.random.seed(42)
inputs = np.random.randn(batch_size, height, width, num_channels).astype(np.float32)
depthwise_kernel = np.random.randn(3, 3, num_channels, depth_multiplier).astype(np.float32)
pointwise_kernel = np.random.randn(1, 1, num_channels * depth_multiplier, num_output_channels).astype(np.float32)

example_output = call_func(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format="channels_last")