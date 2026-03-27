import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.var(inputs, axis=axis, keepdims=keepdims)

# Create eager tensor with shape (3, 4, 5)
eager_tensor = tf.constant([[[1.0, 2.0, 3.0, 4.0, 5.0],
                            [6.0, 7.0, 8.0, 9.0, 10.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0],
                            [16.0, 17.0, 18.0, 19.0, 20.0]],
                           [[21.0, 22.0, 23.0, 24.0, 25.0],
                            [26.0, 27.0, 28.0, 29.0, 30.0],
                            [31.0, 32.0, 33.0, 34.0, 35.0],
                            [36.0, 37.0, 38.0, 39.0, 40.0]],
                           [[41.0, 42.0, 43.0, 44.0, 45.0],
                            [46.0, 47.0, 48.0, 49.0, 50.0],
                            [51.0, 52.0, 53.0, 54.0, 55.0],
                            [56.0, 57.0, 58.0, 59.0, 60.0]]])

# Test with eager tensor
print("Testing with eager tensor:")
print(f"Input shape: {eager_tensor.shape}")
try:
    dynamic_result = call_func(eager_tensor, axis=[-1, -3], keepdims=False)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test with Keras.Input placeholder
print("\nTesting with Keras.Input placeholder:")
placeholder_input = keras.Input(shape=(4, 5))
print(f"Placeholder shape: {placeholder_input.shape}")
try:
    static_result = call_func(placeholder_input, axis=[-1, -3], keepdims=False)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")