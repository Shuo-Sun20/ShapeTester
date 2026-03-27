import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axis=-1):
    return keras.ops.argsort(x=inputs, axis=axis)

# Create test input - eager tensor with shape (3, 4)
eager_input = tf.constant([[3, 1, 4, 2], 
                          [2, 5, 1, 3], 
                          [4, 2, 3, 1]], dtype=tf.float32)

# Test with eager tensor and axis=None
print("Testing with eager tensor:")
try:
    dynamic_result = call_func(eager_input, axis=None)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test with Keras Input placeholder of same shape
print("\nTesting with Keras Input placeholder:")
try:
    placeholder_input = keras.Input(shape=(3, 4))
    static_result = call_func(placeholder_input, axis=None)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static execution error: {e}")

# Additional verification
print(f"\nEager input shape: {eager_input.shape}")
print(f"Eager input type: {type(eager_input)}")