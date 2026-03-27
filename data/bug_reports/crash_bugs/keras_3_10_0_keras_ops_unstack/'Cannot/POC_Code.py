import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=0, num=None):
    return keras.ops.unstack(inputs, axis=axis, num=num)

# Create eager tensor with shape (3, 4, 5)
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

print("Eager tensor shape:", eager_tensor.shape)

# Test with eager tensor - this should work and show dynamic shapes
print("\n=== Testing with eager tensor ===")
try:
    result_eager = call_func(eager_tensor, axis=-3, num=None)
    print("Dynamic output shapes:", [tensor.shape for tensor in result_eager])
except Exception as e:
    print("Error with eager tensor:", str(e))

# Create Keras Input placeholder with same shape but None batch dimension
print("\n=== Testing with Keras Input placeholder ===")
try:
    input_placeholder = keras.Input(shape=(4, 5))  # This creates shape (None, 4, 5)
    print("Input placeholder shape:", input_placeholder.shape)
    result_static = call_func(input_placeholder, axis=-3, num=None)
    print("Static output shapes:", [tensor.shape for tensor in result_static])
except Exception as e:
    print("Error with static placeholder:", str(e))