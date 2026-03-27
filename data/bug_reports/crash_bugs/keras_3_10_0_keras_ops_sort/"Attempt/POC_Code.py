import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=-1):
    return keras.ops.sort(inputs, axis=axis)

# Test with eager tensor
eager_tensor = tf.constant([[[1, 4, 3, 2, 5],
                            [9, 7, 6, 8, 10],
                            [11, 14, 13, 12, 15],
                            [19, 17, 16, 18, 20]],
                           [[21, 24, 23, 22, 25],
                            [29, 27, 26, 28, 30],
                            [31, 34, 33, 32, 35],
                            [39, 37, 36, 38, 40]],
                           [[41, 44, 43, 42, 45],
                            [49, 47, 46, 48, 50],
                            [51, 54, 53, 52, 55],
                            [59, 57, 56, 58, 60]]], dtype=tf.float32)

print(f"Eager tensor shape: {eager_tensor.shape}")

# Test with axis=None on eager tensor (this should cause the error)
try:
    dynamic_result = call_func(eager_tensor, axis=None)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test with Keras.Input placeholder
input_placeholder = keras.Input(shape=(4, 5))
static_result = call_func(input_placeholder, axis=None)
print(f"Static output shape: {static_result.shape}")