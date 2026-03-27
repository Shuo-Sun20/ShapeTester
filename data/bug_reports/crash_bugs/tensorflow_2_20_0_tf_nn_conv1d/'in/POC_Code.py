import tensorflow as tf
import numpy as np
import tensorflow as tf

def call_func(inputs, stride, padding, data_format="NWC", dilations=1, name=None):
    input_tensor, filters_tensor = inputs[0], inputs[1]
    return tf.nn.conv1d(input=input_tensor, filters=filters_tensor, stride=stride, padding=padding, data_format=data_format, dilations=dilations, name=name)

# Create test inputs that reproduce the defect
input_tensor = tf.constant([[[1.0, 2.0, 3.0, 4.0, 5.0],
                            [6.0, 7.0, 8.0, 9.0, 10.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0]],
                           [[16.0, 17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0, 25.0],
                            [26.0, 27.0, 28.0, 29.0, 30.0]]], dtype=tf.float32)  # shape: (2, 3, 5)

filters_tensor = tf.constant([[[1.0, 0.5, 0.2, 0.1],
                              [0.8, 0.6, 0.3, 0.15],
                              [0.7, 0.4, 0.25, 0.12]],
                             [[0.9, 0.55, 0.22, 0.11],
                              [0.75, 0.65, 0.35, 0.16],
                              [0.85, 0.45, 0.28, 0.13]],
                             [[0.95, 0.58, 0.24, 0.14],
                              [0.78, 0.68, 0.38, 0.18],
                              [0.88, 0.48, 0.31, 0.17]]], dtype=tf.float32)  # shape: (3, 3, 4)

inputs = [input_tensor, filters_tensor]
stride = 2
padding = 'VALID'
data_format = 'NCW'
dilations = 3

print("Input shape:", input_tensor.shape)
print("Filter shape:", filters_tensor.shape)

# Test dynamic execution (direct call)
try:
    dynamic_result = call_func(inputs, stride, padding, data_format, dilations)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic execution error:", str(e))

# Test static execution (tf.function)
try:
    static_func = tf.function(call_func)
    static_result = static_func(inputs, stride, padding, data_format, dilations)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static execution error:", str(e))