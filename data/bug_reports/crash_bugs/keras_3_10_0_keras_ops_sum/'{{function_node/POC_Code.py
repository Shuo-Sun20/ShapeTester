import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.sum(x=inputs, axis=axis, keepdims=keepdims)

# Test with eager tensor (dynamic)
print("Testing with eager tensor:")
eager_input = tf.constant([[[1, 2, 3, 4, 5],
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

print(f"Eager input shape: {eager_input.shape}")

try:
    dynamic_result = call_func(eager_input, axis=[1, -2], keepdims=True)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

# Test with Keras.Input placeholder (static)
print("\nTesting with Keras.Input placeholder:")
static_input = keras.Input(shape=(4, 5))
print(f"Static input shape: {static_input.shape}")

try:
    static_result = call_func(static_input, axis=[1, -2], keepdims=True)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output error: {e}")