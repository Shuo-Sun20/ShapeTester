import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
import numpy as np

def call_func(inputs, axis=None, dtype=None):
    return keras.ops.cumprod(x=inputs, axis=axis, dtype=dtype)

# Test with eager tensor
eager_input = tf.constant(np.random.random((3, 4)), dtype=tf.float64)
print("Testing with eager tensor:")
print(f"Input shape: {eager_input.shape}")
try:
    eager_result = call_func(eager_input, axis=-3, dtype=None)
    print(f"Dynamic output shape: {eager_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: Exception encountered when calling Cumprod.call().")
    print(f"Error: {e}")

# Test with Keras.Input placeholder
print("\nTesting with Keras.Input placeholder:")
placeholder_input = keras.Input(shape=(4,), dtype=tf.float64)
print(f"Input shape: {placeholder_input.shape}")
try:
    static_result = call_func(placeholder_input, axis=-3, dtype=None)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: Exception - {e}")