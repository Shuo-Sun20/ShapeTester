import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axes=2):
    x1, x2 = inputs
    return keras.ops.tensordot(x1, x2, axes=axes)

# Test with eager tensors
eager_x1 = tf.constant([[[1.0] * 5] * 4] * 3)  # shape (3, 4, 5)
eager_x2 = tf.constant([[[1.0] * 3] * 4] * 5)  # shape (5, 4, 3)
eager_inputs = [eager_x1, eager_x2]

# Get dynamic output shape
dynamic_result = call_func(eager_inputs, axes=0)
dynamic_shape = dynamic_result.shape
print(f"Dynamic output shape: {dynamic_shape}")

# Test with Keras.Input placeholders
static_x1 = keras.Input(shape=(4, 5), batch_size=3)  # shape (3, 4, 5)
static_x2 = keras.Input(shape=(4, 3), batch_size=5)  # shape (5, 4, 3)
static_inputs = [static_x1, static_x2]

# Get static output shape
static_result = call_func(static_inputs, axes=0)
static_shape = static_result.shape
print(f"Static output shape: {static_shape}")

# Compare shapes
print(f"Shapes match: {dynamic_shape == static_shape}")