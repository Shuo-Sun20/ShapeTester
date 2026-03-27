import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, num_classes, axis=-1, dtype=None, sparse=False):
    return keras.ops.one_hot(inputs, num_classes, axis, dtype, sparse)

# Create eager tensor with shape (4,)
eager_tensor = tf.constant([0, 0, 0, 0])

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(4,))

# Test with eager tensor
eager_result = call_func(eager_tensor, num_classes=1, axis=0, dtype=None, sparse=False)
print(f"Eager tensor result shape: {eager_result.shape}")
print(f"Dynamic output shape: [None, 4] (actual: {list(eager_result.shape)})")

# Test with Keras Input placeholder
placeholder_result = call_func(input_placeholder, num_classes=1, axis=0, dtype=None, sparse=False)
print(f"Placeholder result shape: {placeholder_result.shape}")
print(f"Static output shape: [1, None] (actual: {list(placeholder_result.shape)})")

# Demonstrate the inconsistency
print(f"\nInconsistency detected:")
print(f"Eager tensor shape: {list(eager_result.shape)}")
print(f"Placeholder shape: {list(placeholder_result.shape)}")
print(f"Shapes are {'consistent' if list(eager_result.shape) == list(placeholder_result.shape) else 'inconsistent'}")