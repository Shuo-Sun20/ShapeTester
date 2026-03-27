import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.argmin(x=inputs, axis=axis, keepdims=keepdims)

# Test with eager tensor
eager_tensor = tf.ones((3, 4, 5))
dynamic_result = call_func(eager_tensor, axis=None, keepdims=True)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras Input placeholder
input_placeholder = keras.Input(shape=(4, 5))
static_result = call_func(input_placeholder, axis=None, keepdims=True)
print(f"Static output shape: {static_result.shape}")

print(f"Dynamic shape: {list(dynamic_result.shape)}")
print(f"Static shape: {list(static_result.shape)}")
print(f"Shapes match: {list(dynamic_result.shape) == list(static_result.shape)}")