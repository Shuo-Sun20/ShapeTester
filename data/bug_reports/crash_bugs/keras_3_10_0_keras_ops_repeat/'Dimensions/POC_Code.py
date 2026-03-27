import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, repeats, axis=None):
    return keras.ops.repeat(x=inputs, repeats=repeats, axis=axis)

# Test with eager tensor
eager_tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=tf.float32)
repeats = [0, 1, 2, 3]
axis = None

# Get dynamic output shape with eager tensor
dynamic_result = call_func(eager_tensor, repeats, axis)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(4,), dtype=tf.float32)
static_result = call_func(input_placeholder, repeats, axis)
print(f"Static output shape: {static_result.shape}")

# Verify the inconsistency
print(f"Shapes are consistent: {dynamic_result.shape == static_result.shape}")