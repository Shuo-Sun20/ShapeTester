import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, ksize, strides, padding, data_format=None, name=None):
    return tf.nn.max_pool(
        input=inputs,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )

# Create test input
inputs = tf.constant([[[[1.0], [2.0], [3.0], [4.0]],
                       [[5.0], [6.0], [7.0], [8.0]],
                       [[9.0], [10.0], [11.0], [12.0]],
                       [[13.0], [14.0], [15.0], [16.0]]]], dtype=tf.float32)

# Test parameters that cause the defect
ksize = 2
strides = [1, 1]
padding = 'VALID'
data_format = 'NCHW'
name = None

print("Input shape:", inputs.shape)

# Dynamic execution (eager mode)
try:
    dynamic_result = call_func(inputs, ksize, strides, padding, data_format, name)
    print("Dynamic output shape:", list(dynamic_result.shape))
except Exception as e:
    print("Dynamic execution error:", str(e))

# Static execution (tf.function)
@tf.function
def static_call():
    return call_func(inputs, ksize, strides, padding, data_format, name)

try:
    static_result = static_call()
    print("Static output shape:", list(static_result.shape))
except Exception as e:
    print("Static execution error:", str(e))