import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=None):
    return keras.ops.flip(x=inputs, axis=axis)

# Create eager tensor
eager_tensor = tf.constant([[[1, 2, 3, 4, 5],
                            [6, 7, 8, 9, 10],
                            [11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20]],
                           [[21, 22, 23, 24, 25],
                            [26, 27, 28, 29, 30],
                            [31, 32, 33, 34, 35],
                            [36, 37, 38, 39, 40]],
                           [[41, 42, 43, 44, 45],
                            [46, 47, 48, 49, 50],
                            [51, 52, 53, 54, 55],
                            [56, 57, 58, 59, 60]]], dtype=tf.float32)

# Create Keras Input placeholder
input_placeholder = keras.Input(shape=(4, 5))

print("Eager tensor shape:", eager_tensor.shape)
print("Input placeholder shape:", input_placeholder.shape)

# Test with eager tensor - this should cause the error
try:
    dynamic_result = call_func(eager_tensor, axis="invalid")
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic output shape error:", str(e))

# Test with Keras Input placeholder - this should work differently
try:
    static_result = call_func(input_placeholder, axis="invalid")
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output shape error:", str(e))