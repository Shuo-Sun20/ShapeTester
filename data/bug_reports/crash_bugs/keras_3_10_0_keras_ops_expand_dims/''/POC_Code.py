import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis):
    return keras.ops.expand_dims(x=inputs, axis=axis)

# Create eager tensor with shape (3, 4)
eager_tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Eager tensor shape: {eager_tensor.shape}")

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(3, 4))
print(f"Input placeholder shape: {input_placeholder.shape}")

# Test with axis [-3, 0, 2]
axis = [-3, 0, 2]

# Call with eager tensor
eager_result = call_func(eager_tensor, axis)
print(f"Dynamic output shape (eager): {eager_result.shape}")

# Call with Keras Input placeholder
static_result = call_func(input_placeholder, axis)
print(f"Static output shape (placeholder): {static_result.shape}")

# Print comparison
print(f"\nComparison:")
print(f"Dynamic (eager): {list(eager_result.shape)}")
print(f"Static (placeholder): {list(static_result.shape)}")
print(f"Shapes match: {eager_result.shape == static_result.shape}")