import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axes=2):
    x1, x2 = inputs
    return keras.ops.tensordot(x1, x2, axes=axes)

# Test with eager tensors
x1_eager = tf.constant([[[1.0] * 5] * 4] * 3)  # shape (3, 4, 5)
x2_eager = tf.constant([[[1.0] * 3] * 4] * 5)  # shape (5, 4, 3)
eager_inputs = [x1_eager, x2_eager]
axes = [[0, 1], [1, 2]]

# Call with eager tensors
try:
    eager_result = call_func(eager_inputs, axes=axes)
    print(f"Dynamic output shape: {eager_result.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

# Test with Keras.Input placeholders
x1_placeholder = keras.Input(shape=(4, 5))  # shape (None, 4, 5)
x2_placeholder = keras.Input(shape=(4, 3))  # shape (None, 4, 3)
placeholder_inputs = [x1_placeholder, x2_placeholder]

# Call with placeholders
try:
    static_result = call_func(placeholder_inputs, axes=axes)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output error: {e}")