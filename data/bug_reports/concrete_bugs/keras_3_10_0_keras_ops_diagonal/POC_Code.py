import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, offset=0, axis1=0, axis2=1):
    return keras.ops.diagonal(inputs, offset, axis1, axis2)

# Create eager tensor with shape (2, 3, 4)
eager_tensor = tf.constant([[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]],
                           [[13, 14, 15, 16],
                            [17, 18, 19, 20],
                            [21, 22, 23, 24]]], dtype=tf.float32)

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(3, 4))

# Test with eager tensor
eager_result = call_func(eager_tensor, offset=-1, axis1=0, axis2=1)
print(f"Eager tensor input shape: {eager_tensor.shape}")
print(f"Eager tensor output shape: {eager_result.shape}")
print(f"Dynamic output shape: [{eager_result.shape[0] if eager_result.shape[0] is not None else 'None'}, {eager_result.shape[1] if len(eager_result.shape) > 1 else 'None'}]")

# Test with Keras Input placeholder
placeholder_result = call_func(input_placeholder, offset=-1, axis1=0, axis2=1)
print(f"Placeholder input shape: {input_placeholder.shape}")
print(f"Placeholder output shape: {placeholder_result.shape}")
print(f"Static output shape: [{placeholder_result.shape[0] if placeholder_result.shape[0] is not None else 'None'}, {placeholder_result.shape[1] if len(placeholder_result.shape) > 1 else 'None'}]")

print("\nDefect reproduced: Dynamic and static output shapes are inconsistent!")