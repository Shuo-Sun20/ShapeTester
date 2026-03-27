import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axes=None):
    return keras.ops.transpose(x=inputs, axes=axes)

# Create eager tensor with shape (2, 3, 4)
eager_tensor = tf.constant([[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]],
                           [[13, 14, 15, 16],
                            [17, 18, 19, 20],
                            [21, 22, 23, 24]]], dtype=tf.float32)

print(f"Eager tensor shape: {eager_tensor.shape}")

# Test with eager tensor
eager_result = call_func(eager_tensor, axes=None)
print(f"Dynamic output shape (eager): {eager_result.shape}")

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(3, 4))
print(f"Input placeholder shape: {input_placeholder.shape}")

# Test with Keras Input placeholder
static_result = call_func(input_placeholder, axes=None)
print(f"Static output shape (placeholder): {static_result.shape}")

# Demonstrate the inconsistency
print(f"\nInconsistency detected:")
print(f"Dynamic output shape: {eager_result.shape} -> [None, 3, 2] format: [{eager_result.shape[1]}, {eager_result.shape[2]}, None]")
print(f"Static output shape: {static_result.shape} -> [4, 3, None] format: [{static_result.shape[0]}, {static_result.shape[1]}, None]")