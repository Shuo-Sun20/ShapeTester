import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras

def call_func(inputs, kth, axis=-1):
    return keras.ops.argpartition(inputs, kth, axis)

# Create test input - eager tensor with shape (4, 4, 4)
eager_tensor = tf.random.normal((4, 4, 4))

# Create static input - Keras Input placeholder with same shape
static_input = keras.Input(shape=(4, 4, 4))

# Test parameters
kth = 0
axis = None

print("Testing keras.ops.argpartition defect:")
print(f"Input shape: (4, 4, 4)")
print(f"kth: {kth}")
print(f"axis: {axis}")
print()

# Test with eager tensor (dynamic)
try:
    dynamic_result = call_func(eager_tensor, kth, axis)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test with static input (Keras Input placeholder)
try:
    static_result = call_func(static_input, kth, axis)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {e}")