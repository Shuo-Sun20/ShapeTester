import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=0):
    return keras.ops.stack(x=inputs, axis=axis)

# Create eager tensors
eager_tensor1 = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
eager_tensor2 = tf.constant([[7, 8, 9], [10, 11, 12]])  # shape (2, 3)
eager_tensor3 = tf.constant([[13, 14, 15], [16, 17, 18]])  # shape (2, 3)
eager_inputs = [eager_tensor1, eager_tensor2, eager_tensor3]

# Test with eager tensors
axis = -3
dynamic_result = call_func(eager_inputs, axis=axis)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras.Input placeholders
input1 = keras.Input(shape=(2, 3))
input2 = keras.Input(shape=(2, 3))
input3 = keras.Input(shape=(2, 3))
placeholder_inputs = [input1, input2, input3]

static_result = call_func(placeholder_inputs, axis=axis)
print(f"Static output shape: {static_result.shape}")

# Verify the defect
print(f"Dynamic shape: {list(dynamic_result.shape)}")
print(f"Static shape: {list(static_result.shape)}")
print(f"Shapes are consistent: {list(dynamic_result.shape) == list(static_result.shape)}")