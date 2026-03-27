import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis1, axis2):
    return keras.ops.swapaxes(inputs, axis1, axis2)

# Create eager tensor with shape (3, 4, 5)
eager_tensor = tf.constant([[[1, 2, 3, 4, 5] for _ in range(4)] for _ in range(3)])
print(f"Eager tensor shape: {eager_tensor.shape}")

# Test with eager tensor
dynamic_result = call_func(eager_tensor, -3, -2)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Create Keras Input placeholder with same shape (3, 4, 5)
input_placeholder = keras.Input(shape=(3, 4, 5))
print(f"Input placeholder shape: {input_placeholder.shape}")

# Test with Keras Input placeholder
static_result = call_func(input_placeholder, -3, -2)
print(f"Static output shape: {static_result.shape}")

# Show the inconsistency
print(f"\nInconsistency detected:")
print(f"Dynamic result shape: {dynamic_result.shape}")
print(f"Static result shape: {static_result.shape}")