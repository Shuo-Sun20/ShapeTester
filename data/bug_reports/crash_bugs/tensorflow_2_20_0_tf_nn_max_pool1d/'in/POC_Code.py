import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, ksize, strides, padding, data_format="NWC", name=None):
    return tf.nn.max_pool1d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

# Create test input
inputs = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], 
                      [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0], [28.0, 29.0, 30.0]],
                     [[31.0, 32.0, 33.0], [34.0, 35.0, 36.0], [37.0, 38.0, 39.0], [40.0, 41.0, 42.0], [43.0, 44.0, 45.0],
                      [46.0, 47.0, 48.0], [49.0, 50.0, 51.0], [52.0, 53.0, 54.0], [55.0, 56.0, 57.0], [58.0, 59.0, 60.0]]], 
                    dtype=tf.float32)  # shape=(2, 10, 3)

ksize = 4
strides = 1
padding = 'VALID'
data_format = 'NCW'

print("Input shape:", inputs.shape)

# Direct call - dynamic execution
try:
    dynamic_result = call_func(inputs, ksize, strides, padding, data_format)
    print("Dynamic output shape:", dynamic_result.shape.as_list())
except Exception as e:
    print("Dynamic execution error:", str(e))

# tf.function call - static execution
@tf.function
def static_call_func(inputs, ksize, strides, padding, data_format="NWC", name=None):
    return tf.nn.max_pool1d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

try:
    static_result = static_call_func(inputs, ksize, strides, padding, data_format)
    print("Static output shape:", static_result.shape.as_list())
except Exception as e:
    print("Static execution error:", str(e))