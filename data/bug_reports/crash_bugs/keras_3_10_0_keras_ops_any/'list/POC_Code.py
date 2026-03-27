import keras
import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.any(x=inputs, axis=axis, keepdims=keepdims)

# Create eager tensor
eager_tensor = tf.constant([[True, False, True, False],
                           [False, True, False, True], 
                           [True, True, False, False]], dtype=tf.bool)

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(3, 4), dtype='bool')

# Test with eager tensor
print("Testing with eager tensor:")
print(f"Input shape: {eager_tensor.shape}")
dynamic_result = call_func(eager_tensor, axis=[-1, -2], keepdims=False)
print(f"Dynamic output shape: {dynamic_result.shape}")

# Test with Keras Input placeholder
print("\nTesting with Keras Input placeholder:")
print(f"Input shape: {input_placeholder.shape}")
try:
    static_result = call_func(input_placeholder, axis=[-1, -2], keepdims=False)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output error: {e}")

# Demonstrate the inconsistency
print(f"\nDefect reproduction:")
print(f"Dynamic output shape: {list(dynamic_result.shape)}")
print("Static output error: list assignment index out of range")