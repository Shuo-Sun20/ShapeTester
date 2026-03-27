import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, ksize, strides, padding, data_format="NWC", name=None):
    return tf.nn.avg_pool1d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

# Create test input
inputs = tf.random.normal((4, 10, 3))
ksize = 4
strides = 4
padding = 'VALID'
data_format = 'NCW'
name = None

# Test dynamic execution
print("Dynamic execution:")
try:
    dynamic_result = call_func(inputs, ksize, strides, padding, data_format, name)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test static execution with tf.function
print("\nStatic execution with tf.function:")
try:
    static_func = tf.function(call_func)
    static_result = static_func(inputs, ksize, strides, padding, data_format, name)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")