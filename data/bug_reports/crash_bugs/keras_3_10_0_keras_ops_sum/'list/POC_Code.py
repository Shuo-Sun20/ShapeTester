import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.sum(x=inputs, axis=axis, keepdims=keepdims)

# Create eager tensor with shape (3, 4, 5)
eager_tensor = tf.constant([[[1.0] * 5 for _ in range(4)] for _ in range(3)])
print(f"Eager tensor shape: {eager_tensor.shape}")

# Create Keras Input placeholder with same shape
input_placeholder = keras.Input(shape=(4, 5))
print(f"Input placeholder shape: {input_placeholder.shape}")

# Test parameters
axis = [-1, -3]
keepdims = False

# Call with eager tensor
print("\n--- Testing with eager tensor ---")
try:
    result_eager = call_func(eager_tensor, axis=axis, keepdims=keepdims)
    print(f"Dynamic output shape: {result_eager.shape}")
except Exception as e:
    print(f"Error with eager tensor: {e}")

# Call with Keras Input placeholder
print("\n--- Testing with Keras Input placeholder ---")
try:
    result_static = call_func(input_placeholder, axis=axis, keepdims=keepdims)
    print(f"Static output shape: {result_static.shape}")
except Exception as e:
    print(f"Error with input placeholder: {e}")

# Compare shapes
print("\n--- Shape comparison ---")
try:
    eager_shape = result_eager.shape
    static_shape = result_static.shape
    print(f"Eager tensor result shape: {eager_shape}")
    print(f"Static placeholder result shape: {static_shape}")
    print(f"Shapes match: {eager_shape == static_shape}")
except:
    print("Unable to compare shapes due to errors")