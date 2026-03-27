import keras
import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras
import numpy as np

def call_func(axis, epsilon, inputs):
    layer = keras.layers.RMSNormalization(axis=axis, epsilon=epsilon)
    return layer(inputs)

# Test input parameters
axis = 0
epsilon = 1e-06

# Create test input tensor
input_shape = (1, 10)
test_input = np.random.rand(*input_shape).astype(np.float32)

print("=== Testing with eager tensor ===")
# Test with eager tensor
eager_tensor = tf.constant(test_input)
try:
    dynamic_output = call_func(axis, epsilon, eager_tensor)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

print("\n=== Testing with Keras.Input placeholder ===")
# Test with Keras.Input placeholder
try:
    input_placeholder = keras.Input(shape=input_shape[1:])  # Remove batch dimension for Input
    static_output = call_func(axis, epsilon, input_placeholder)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static execution error: {e}")

print("\n=== Reproducing the defect ===")
print(f"Input shape: {input_shape}")
print(f"Axis: {axis}")
print(f"Epsilon: {epsilon}")
print("Expected behavior: Both dynamic and static execution should work consistently")
print("Actual behavior: Static execution fails due to None dimensions in variable initialization")