import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, window_shape, pooling_type, strides=None, padding="SAME", data_format=None, dilations=None, name=None):
    return tf.nn.pool(input=inputs, window_shape=window_shape, pooling_type=pooling_type, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=name)

# Create test input
inputs = tf.constant([[[[1.0, 2.0, 3.0] for _ in range(32)] for _ in range(32)] for _ in range(4)])
window_shape = [6, 6]
pooling_type = 'AVG'
strides = [1, 2]
padding = 'VALID'
data_format = 'NCHW'
dilations = [1, 1]
name = None

print("Input shape:", inputs.shape)

# Direct call - dynamic execution
try:
    dynamic_result = call_func(inputs, window_shape, pooling_type, strides, padding, data_format, dilations, name)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic execution error:", str(e))

# tf.function call - static execution
try:
    static_func = tf.function(call_func)
    static_result = static_func(inputs, window_shape, pooling_type, strides, padding, data_format, dilations, name)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static execution error:", str(e))