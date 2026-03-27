import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis):
    x, indices = inputs
    return keras.ops.take(x, indices, axis)

# Test with eager tensors (dynamic case)
x_eager = tf.constant([[[1, 2, 3, 4, 5],
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
                        [56, 57, 58, 59, 60]]], dtype=tf.float32)  # shape (3, 4, 5)
indices_eager = tf.constant([0, 2], dtype=tf.int32)  # shape (2,)

dynamic_result = call_func([x_eager, indices_eager], axis=1)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras Input placeholders (static case)
x_input = keras.Input(shape=(4, 5))  # shape (None, 4, 5)
indices_input = keras.Input(shape=(2,))  # shape (None, 2)

static_result = call_func([x_input, indices_input], axis=1)
print(f"Static output shape: {static_result.shape}")

# Verify the defect
print(f"Dynamic shape: [None, {dynamic_result.shape[1]}, {dynamic_result.shape[2]}]")
print(f"Static shape: [None, {static_result.shape[1]}, {static_result.shape[2]}]")
print(f"Shapes match: {dynamic_result.shape[1:] == static_result.shape[1:]}")