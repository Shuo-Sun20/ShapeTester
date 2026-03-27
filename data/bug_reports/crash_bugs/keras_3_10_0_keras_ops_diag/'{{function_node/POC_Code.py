import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, k=0):
    x = inputs[0]
    return keras.ops.diag(x, k=k)

# Create test input - eager tensor with shape (4, 4)
eager_input = tf.constant([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]], dtype=tf.float32)

# Create static input - Keras Input placeholder with same shape
static_input = keras.Input(shape=(4, 4))

# Test with k=-4 (the problematic case)
k_value = -4

print("Testing with eager tensor:")
try:
    eager_result = call_func([eager_input], k=k_value)
    print(f"Dynamic output shape: {eager_result.shape}")
except Exception as e:
    print(f"Dynamic output error: {e}")

print("\nTesting with static placeholder:")
try:
    static_result = call_func([static_input], k=k_value)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output error: {e}")

# Additional test to show the inconsistency
print("\nShape comparison:")
print(f"Eager tensor input shape: {eager_input.shape}")
print(f"Static input shape: {static_input.shape}")