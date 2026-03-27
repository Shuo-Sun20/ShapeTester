import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, repeats):
    return keras.ops.tile(inputs, repeats)

# Test with eager tensor
eager_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
repeats = [1, 1, 1]

# Get dynamic output shape with eager tensor
dynamic_result = call_func(eager_tensor, repeats)
dynamic_shape = dynamic_result.shape.as_list()
print(f"Dynamic output shape: {dynamic_shape}")

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(2, 3))
static_result = call_func(input_placeholder, repeats)
static_shape = static_result.shape
print(f"Static output shape: {static_shape}")

# Verify the defect
print(f"Shapes are inconsistent: {dynamic_shape != static_shape}")