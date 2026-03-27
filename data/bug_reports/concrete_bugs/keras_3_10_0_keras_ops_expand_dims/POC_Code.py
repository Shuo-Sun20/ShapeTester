import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis):
    return keras.ops.expand_dims(x=inputs, axis=axis)

# Create eager tensor input
eager_tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])  # shape (3, 4)

# Test with eager tensor
dynamic_result = call_func(eager_tensor, axis=-3)
print("Dynamic output shape:", dynamic_result.shape)

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(3, 4))

# Test with Keras Input placeholder
static_result = call_func(input_placeholder, axis=-3)
print("Static output shape:", static_result.shape)

# Verify the defect
print(f"Dynamic shape: {list(dynamic_result.shape)}")
print(f"Static shape: {list(static_result.shape)}")
print(f"Shapes are inconsistent: {list(dynamic_result.shape) != list(static_result.shape)}")