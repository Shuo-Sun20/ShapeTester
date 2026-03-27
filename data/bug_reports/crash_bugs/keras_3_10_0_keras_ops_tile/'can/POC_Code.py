import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, repeats):
    return keras.ops.tile(inputs, repeats)

# Test with eager tensor
eager_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
repeats = [0]

# Get dynamic output shape with eager tensor
dynamic_result = call_func(eager_tensor, repeats)
dynamic_shape = dynamic_result.shape
print(f"Dynamic output shape: {dynamic_shape}")

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(3,))  # shape (None, 3) - same as eager tensor
try:
    static_result = call_func(input_placeholder, repeats)
    static_shape = static_result.shape
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static output shape error: {e}")

# Verify the defect
print(f"Eager tensor input shape: {eager_tensor.shape}")
print(f"Keras Input shape: {input_placeholder.shape}")
print(f"Repeats: {repeats}")