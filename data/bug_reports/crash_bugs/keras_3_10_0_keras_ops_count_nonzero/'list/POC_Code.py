import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axis=None):
    if isinstance(inputs, list):
        x = inputs[0]
    else:
        x = inputs
    return keras.ops.count_nonzero(x, axis)

# Create eager tensor input
eager_input = tf.constant([[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 0]], dtype=tf.float32)
print(f"Eager input shape: {eager_input.shape}")

# Test with eager tensor
dynamic_result = call_func(eager_input, axis=[-2, -1])
print(f"Dynamic output shape: {dynamic_result.shape}")
print(f"Dynamic result: {dynamic_result}")

# Test with Keras.Input placeholder of same shape
static_input = keras.Input(shape=(3, 4))
print(f"Static input shape: {static_input.shape}")

try:
    static_result = call_func(static_input, axis=[-2, -1])
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {e}")