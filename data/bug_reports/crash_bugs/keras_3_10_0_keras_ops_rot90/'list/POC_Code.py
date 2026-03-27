import keras
import numpy as np
import tensorflow as tf
import keras
import numpy as np

def call_func(inputs, k=1, axes=(0, 1)):
    return keras.ops.rot90(array=inputs, k=k, axes=axes)

# Test input parameters
input_shape = (3, 4, 5)
k = 8
axes = [2, 3]

# Test with eager tensors (dynamic execution)
print("Testing with eager tensors:")
eager_input = np.random.random(input_shape)
try:
    dynamic_result = call_func(eager_input, k=k, axes=axes)
    dynamic_shape = dynamic_result.shape
    print(f"Dynamic output shape: {dynamic_shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Test with Keras.Input placeholders (static execution)
print("\nTesting with Keras.Input placeholders:")
try:
    static_input = keras.Input(shape=input_shape)
    static_result = call_func(static_input, k=k, axes=axes)
    static_shape = static_result.shape
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static execution error: {e}")

# Compare shapes
print(f"\nComparison:")
print(f"Input shape: {input_shape}")
print(f"k: {k}")
print(f"axes: {axes}")